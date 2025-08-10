"""
LLM Fallback and Error Handling System - Comprehensive strategies for handling LLM failures.

This module provides:
- Multi-tier fallback strategies (expensive -> cheap -> local -> static)
- Circuit breaker patterns for failing services
- Graceful degradation with partial results
- Error classification and handling
- Recovery strategies and retry logic
- Health monitoring and automatic switching
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict
import logging
from contextlib import asynccontextmanager

from ..core.config import Settings, get_settings
from ..services.llm_service import LLMService, LLMResponse, PromptTemplate, ModelType
from ..services.llm_cost_manager import get_cost_manager, CostCategory
from ..db.redis import RedisManager, get_redis_manager
from ..utils.logger import get_logger
from ..utils.exceptions import ValidationError

logger = get_logger(__name__)


class ErrorSeverity(Enum):
    """Severity levels for LLM errors."""
    LOW = "low"           # Minor issues, retry with same service
    MEDIUM = "medium"     # Service issues, try fallback
    HIGH = "high"         # Critical failure, skip to local
    CRITICAL = "critical" # System failure, use static response


class FallbackStrategy(Enum):
    """Types of fallback strategies."""
    MODEL_DOWNGRADE = "model_downgrade"      # Use cheaper model from same provider
    PROVIDER_SWITCH = "provider_switch"      # Switch to different provider
    LOCAL_FALLBACK = "local_fallback"        # Use local model
    STATIC_RESPONSE = "static_response"      # Return pre-computed response
    PARTIAL_DEGRADATION = "partial_degradation"  # Return partial results
    SKIP_ANALYSIS = "skip_analysis"          # Skip analysis entirely


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"     # Normal operation
    OPEN = "open"         # Failing, skip requests
    HALF_OPEN = "half_open"  # Testing if recovered


@dataclass
class ErrorContext:
    """Context information for an error."""
    error_type: str
    error_message: str
    model: str
    provider: str
    severity: ErrorSeverity
    timestamp: datetime
    request_context: Dict[str, Any]
    retry_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data['severity'] = self.severity.value
        data['timestamp'] = self.timestamp.isoformat()
        return data


@dataclass
class FallbackResult:
    """Result of fallback processing."""
    success: bool
    response: Optional[LLMResponse] = None
    fallback_strategy: Optional[FallbackStrategy] = None
    error_message: Optional[str] = None
    degraded: bool = False
    cost_savings: float = 0.0
    fallback_model: Optional[str] = None


@dataclass
class CircuitBreakerState:
    """State of a circuit breaker."""
    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    last_failure_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    failure_threshold: int = 5
    recovery_timeout: int = 300  # 5 minutes
    half_open_max_calls: int = 3
    half_open_calls: int = 0


class LLMErrorClassifier:
    """Classifies LLM errors and determines appropriate handling strategies."""
    
    def __init__(self):
        # Error patterns and their classifications
        self.error_patterns = {
            # Authentication/Authorization errors
            'authentication': {
                'patterns': ['unauthorized', 'invalid api key', 'authentication failed', '401'],
                'severity': ErrorSeverity.HIGH,
                'strategies': [FallbackStrategy.PROVIDER_SWITCH, FallbackStrategy.LOCAL_FALLBACK]
            },
            
            # Rate limiting errors
            'rate_limit': {
                'patterns': ['rate limit', 'too many requests', '429', 'quota exceeded'],
                'severity': ErrorSeverity.MEDIUM,
                'strategies': [FallbackStrategy.MODEL_DOWNGRADE, FallbackStrategy.PROVIDER_SWITCH]
            },
            
            # Service unavailable
            'service_unavailable': {
                'patterns': ['service unavailable', '503', '502', '500', 'internal server error'],
                'severity': ErrorSeverity.HIGH,
                'strategies': [FallbackStrategy.PROVIDER_SWITCH, FallbackStrategy.LOCAL_FALLBACK]
            },
            
            # Timeout errors
            'timeout': {
                'patterns': ['timeout', 'timed out', 'request timeout'],
                'severity': ErrorSeverity.MEDIUM,
                'strategies': [FallbackStrategy.MODEL_DOWNGRADE, FallbackStrategy.PROVIDER_SWITCH]
            },
            
            # Content filtering
            'content_filter': {
                'patterns': ['content filter', 'content policy', 'inappropriate content'],
                'severity': ErrorSeverity.LOW,
                'strategies': [FallbackStrategy.MODEL_DOWNGRADE, FallbackStrategy.STATIC_RESPONSE]
            },
            
            # Token limit exceeded
            'token_limit': {
                'patterns': ['token limit', 'context length', 'input too long', 'max tokens'],
                'severity': ErrorSeverity.MEDIUM,
                'strategies': [FallbackStrategy.PARTIAL_DEGRADATION, FallbackStrategy.MODEL_DOWNGRADE]
            },
            
            # Network errors
            'network': {
                'patterns': ['connection error', 'network error', 'dns error', 'connection timeout'],
                'severity': ErrorSeverity.MEDIUM,
                'strategies': [FallbackStrategy.PROVIDER_SWITCH, FallbackStrategy.LOCAL_FALLBACK]
            },
            
            # Budget/billing errors
            'budget': {
                'patterns': ['insufficient funds', 'billing', 'payment', 'account suspended'],
                'severity': ErrorSeverity.HIGH,
                'strategies': [FallbackStrategy.LOCAL_FALLBACK, FallbackStrategy.SKIP_ANALYSIS]
            }
        }
    
    def classify_error(self, error: Exception, context: Dict[str, Any]) -> ErrorContext:
        """Classify an error and determine handling strategy."""
        error_str = str(error).lower()
        error_type = type(error).__name__
        
        # Find matching error pattern
        severity = ErrorSeverity.MEDIUM  # Default
        classified_type = 'unknown'
        
        for error_category, config in self.error_patterns.items():
            if any(pattern in error_str for pattern in config['patterns']):
                severity = config['severity']
                classified_type = error_category
                break
        
        return ErrorContext(
            error_type=classified_type,
            error_message=str(error),
            model=context.get('model', 'unknown'),
            provider=context.get('provider', 'unknown'),
            severity=severity,
            timestamp=datetime.now(),
            request_context=context
        )
    
    def get_fallback_strategies(self, error_context: ErrorContext) -> List[FallbackStrategy]:
        """Get recommended fallback strategies for an error."""
        if error_context.error_type in self.error_patterns:
            return self.error_patterns[error_context.error_type]['strategies']
        
        # Default fallback strategies
        if error_context.severity == ErrorSeverity.HIGH:
            return [FallbackStrategy.LOCAL_FALLBACK, FallbackStrategy.STATIC_RESPONSE]
        elif error_context.severity == ErrorSeverity.MEDIUM:
            return [FallbackStrategy.MODEL_DOWNGRADE, FallbackStrategy.PROVIDER_SWITCH]
        else:
            return [FallbackStrategy.MODEL_DOWNGRADE]


class LLMFallbackService:
    """
    Comprehensive fallback service for LLM operations with circuit breakers and degradation strategies.
    """
    
    def __init__(
        self,
        settings: Optional[Settings] = None,
        redis_manager: Optional[RedisManager] = None
    ):
        self.settings = settings or get_settings()
        self.redis_manager = redis_manager
        self.error_classifier = LLMErrorClassifier()
        
        # Circuit breaker states for each provider
        self._circuit_breakers: Dict[str, CircuitBreakerState] = {}
        
        # Error tracking
        self._error_counts: Dict[str, int] = defaultdict(int)
        self._last_error_reset = datetime.now()
        
        # Fallback configurations
        self.fallback_models = {
            ModelType.SUMMARIZATION: [
                "claude-3-haiku-20240307",    # Fast and cheap
                "gpt-3.5-turbo",              # Reliable fallback
                "ollama/llama3",              # Local fallback
            ],
            ModelType.ANALYSIS: [
                "gpt-4-1106-preview",         # Primary
                "claude-3-sonnet-20240229",   # Alternative
                "gpt-3.5-turbo",              # Cheaper fallback
                "ollama/llama3",              # Local fallback
            ]
        }
        
        # Static responses for critical failures
        self.static_responses = {
            "paper_summary": "Unable to generate detailed analysis due to service limitations. Please try again later.",
            "citation_analysis": "Citation analysis temporarily unavailable. Basic relationship detected.",
            "trajectory_analysis": "Research trajectory analysis is currently unavailable.",
        }
        
        self._initialized = False
    
    async def initialize(self):
        """Initialize the fallback service."""
        if self._initialized:
            return
        
        if not self.redis_manager:
            self.redis_manager = await get_redis_manager()
        
        # Load circuit breaker states from Redis
        await self._load_circuit_breaker_states()
        
        self._initialized = True
        logger.info("LLM Fallback Service initialized")
    
    async def execute_with_fallback(
        self,
        operation: Callable,
        template: PromptTemplate,
        context: Dict[str, Any],
        max_fallback_attempts: int = 3,
        **kwargs
    ) -> FallbackResult:
        """
        Execute an LLM operation with comprehensive fallback strategies.
        
        Args:
            operation: The LLM operation to execute
            template: Prompt template
            context: Request context
            max_fallback_attempts: Maximum fallback attempts
            **kwargs: Additional arguments for the operation
            
        Returns:
            FallbackResult with response or error information
        """
        await self.initialize()
        
        primary_model = self._get_model_for_task(template.model_type, False)
        provider = self._get_provider_from_model(primary_model)
        
        # Check circuit breaker
        if not self._can_call_provider(provider):
            logger.warning(f"Circuit breaker OPEN for provider {provider}, using fallback")
            return await self._execute_fallback_chain(operation, template, context, kwargs)
        
        # Try primary operation
        try:
            response = await operation(template, **kwargs)
            
            # Record success
            await self._record_success(provider)
            
            return FallbackResult(
                success=True,
                response=response,
                fallback_strategy=None
            )
            
        except Exception as e:
            # Classify error
            error_context = self.error_classifier.classify_error(e, {
                'model': primary_model,
                'provider': provider,
                'operation': operation.__name__ if hasattr(operation, '__name__') else 'unknown',
                **context
            })
            
            # Record failure
            await self._record_failure(provider, error_context)
            
            # Log error
            logger.warning(f"Primary LLM operation failed: {error_context.error_message}")
            
            # Execute fallback chain
            return await self._execute_fallback_chain(
                operation, template, context, kwargs, error_context, max_fallback_attempts
            )
    
    async def _execute_fallback_chain(
        self,
        operation: Callable,
        template: PromptTemplate,
        context: Dict[str, Any],
        kwargs: Dict[str, Any],
        initial_error: Optional[ErrorContext] = None,
        max_attempts: int = 3
    ) -> FallbackResult:
        """Execute fallback chain based on error classification."""
        
        fallback_strategies = []
        if initial_error:
            fallback_strategies = self.error_classifier.get_fallback_strategies(initial_error)
        else:
            # Circuit breaker triggered, use default strategies
            fallback_strategies = [
                FallbackStrategy.PROVIDER_SWITCH,
                FallbackStrategy.LOCAL_FALLBACK,
                FallbackStrategy.STATIC_RESPONSE
            ]
        
        for attempt, strategy in enumerate(fallback_strategies[:max_attempts]):
            try:
                result = await self._execute_fallback_strategy(
                    strategy, operation, template, context, kwargs
                )
                
                if result.success:
                    logger.info(f"Fallback successful using strategy: {strategy.value}")
                    return result
                
            except Exception as e:
                logger.warning(f"Fallback strategy {strategy.value} failed: {e}")
                continue
        
        # All fallback strategies failed
        return FallbackResult(
            success=False,
            error_message="All fallback strategies exhausted",
            fallback_strategy=FallbackStrategy.SKIP_ANALYSIS
        )
    
    async def _execute_fallback_strategy(
        self,
        strategy: FallbackStrategy,
        operation: Callable,
        template: PromptTemplate,
        context: Dict[str, Any],
        kwargs: Dict[str, Any]
    ) -> FallbackResult:
        """Execute a specific fallback strategy."""
        
        if strategy == FallbackStrategy.MODEL_DOWNGRADE:
            return await self._try_model_downgrade(operation, template, context, kwargs)
        
        elif strategy == FallbackStrategy.PROVIDER_SWITCH:
            return await self._try_provider_switch(operation, template, context, kwargs)
        
        elif strategy == FallbackStrategy.LOCAL_FALLBACK:
            return await self._try_local_fallback(operation, template, context, kwargs)
        
        elif strategy == FallbackStrategy.STATIC_RESPONSE:
            return await self._generate_static_response(template, context)
        
        elif strategy == FallbackStrategy.PARTIAL_DEGRADATION:
            return await self._try_partial_degradation(operation, template, context, kwargs)
        
        elif strategy == FallbackStrategy.SKIP_ANALYSIS:
            return FallbackResult(
                success=True,
                response=None,
                fallback_strategy=strategy,
                degraded=True,
                error_message="Analysis skipped due to service limitations"
            )
        
        else:
            raise ValueError(f"Unknown fallback strategy: {strategy}")
    
    async def _try_model_downgrade(
        self,
        operation: Callable,
        template: PromptTemplate,
        context: Dict[str, Any],
        kwargs: Dict[str, Any]
    ) -> FallbackResult:
        """Try using a cheaper/faster model."""
        
        fallback_models = self.fallback_models.get(template.model_type, [])
        
        for model in fallback_models[1:]:  # Skip first (primary) model
            provider = self._get_provider_from_model(model)
            
            if not self._can_call_provider(provider):
                continue
            
            try:
                # Modify template to use fallback model
                fallback_template = PromptTemplate(
                    system_prompt=template.system_prompt,
                    user_template=template.user_template,
                    max_tokens=min(template.max_tokens, 1000),  # Reduce tokens for cheaper models
                    temperature=template.temperature,
                    model_type=template.model_type
                )
                
                # Override model preference
                kwargs['prefer_local'] = 'ollama' in model
                
                response = await operation(fallback_template, **kwargs)
                
                # Calculate cost savings (rough estimate)
                original_cost = 0.05  # Estimated
                fallback_cost = response.usage.cost if response else 0.02
                cost_savings = max(0, original_cost - fallback_cost)
                
                return FallbackResult(
                    success=True,
                    response=response,
                    fallback_strategy=FallbackStrategy.MODEL_DOWNGRADE,
                    cost_savings=cost_savings,
                    fallback_model=model,
                    degraded=True  # Potentially lower quality
                )
                
            except Exception as e:
                logger.warning(f"Model downgrade to {model} failed: {e}")
                continue
        
        return FallbackResult(
            success=False,
            error_message="All model downgrades failed"
        )
    
    async def _try_provider_switch(
        self,
        operation: Callable,
        template: PromptTemplate,
        context: Dict[str, Any],
        kwargs: Dict[str, Any]
    ) -> FallbackResult:
        """Try switching to a different provider."""
        
        # Get alternative models from different providers
        current_model = self._get_model_for_task(template.model_type, False)
        current_provider = self._get_provider_from_model(current_model)
        
        alternative_models = [
            model for model in self.fallback_models.get(template.model_type, [])
            if self._get_provider_from_model(model) != current_provider
            and self._can_call_provider(self._get_provider_from_model(model))
        ]
        
        for model in alternative_models:
            try:
                # Override model preference
                if 'ollama' in model:
                    kwargs['prefer_local'] = True
                else:
                    kwargs['prefer_local'] = False
                
                response = await operation(template, **kwargs)
                
                return FallbackResult(
                    success=True,
                    response=response,
                    fallback_strategy=FallbackStrategy.PROVIDER_SWITCH,
                    fallback_model=model
                )
                
            except Exception as e:
                logger.warning(f"Provider switch to {model} failed: {e}")
                continue
        
        return FallbackResult(
            success=False,
            error_message="All provider switches failed"
        )
    
    async def _try_local_fallback(
        self,
        operation: Callable,
        template: PromptTemplate,
        context: Dict[str, Any],
        kwargs: Dict[str, Any]
    ) -> FallbackResult:
        """Try using local model (Ollama)."""
        
        if not self.settings.llm.enable_local_fallback:
            return FallbackResult(
                success=False,
                error_message="Local fallback disabled"
            )
        
        try:
            # Force local model usage
            kwargs['prefer_local'] = True
            
            # Reduce complexity for local models
            local_template = PromptTemplate(
                system_prompt=template.system_prompt[:500],  # Truncate if too long
                user_template=template.user_template,
                max_tokens=min(template.max_tokens, 800),    # Reduce output length
                temperature=template.temperature,
                model_type=template.model_type
            )
            
            response = await operation(local_template, **kwargs)
            
            # Local models have zero API cost
            return FallbackResult(
                success=True,
                response=response,
                fallback_strategy=FallbackStrategy.LOCAL_FALLBACK,
                cost_savings=0.05,  # Estimated savings
                fallback_model=self.settings.llm.default_local_model,
                degraded=True
            )
            
        except Exception as e:
            logger.warning(f"Local fallback failed: {e}")
            return FallbackResult(
                success=False,
                error_message=f"Local fallback failed: {str(e)}"
            )
    
    async def _try_partial_degradation(
        self,
        operation: Callable,
        template: PromptTemplate,
        context: Dict[str, Any],
        kwargs: Dict[str, Any]
    ) -> FallbackResult:
        """Try processing with reduced input size."""
        
        try:
            # Truncate input for token limit issues
            truncated_template = PromptTemplate(
                system_prompt=template.system_prompt[:300],
                user_template=template.user_template[:2000],  # Significant reduction
                max_tokens=template.max_tokens // 2,          # Reduce output
                temperature=template.temperature,
                model_type=template.model_type
            )
            
            response = await operation(truncated_template, **kwargs)
            
            return FallbackResult(
                success=True,
                response=response,
                fallback_strategy=FallbackStrategy.PARTIAL_DEGRADATION,
                degraded=True,
                error_message="Analysis performed with reduced context due to limitations"
            )
            
        except Exception as e:
            return FallbackResult(
                success=False,
                error_message=f"Partial degradation failed: {str(e)}"
            )
    
    async def _generate_static_response(
        self,
        template: PromptTemplate,
        context: Dict[str, Any]
    ) -> FallbackResult:
        """Generate a static fallback response."""
        
        # Determine response type based on template or context
        response_type = "paper_summary"  # Default
        
        if "citation" in str(template.user_template).lower():
            response_type = "citation_analysis"
        elif "trajectory" in str(template.user_template).lower() or "lineage" in str(template.user_template).lower():
            response_type = "trajectory_analysis"
        
        static_content = self.static_responses.get(response_type, self.static_responses["paper_summary"])
        
        # Create a minimal LLM response
        from ..services.llm_service import LLMUsage
        
        usage = LLMUsage(
            input_tokens=0,
            output_tokens=len(static_content.split()),
            total_tokens=len(static_content.split()),
            cost=0.0,
            model="static_fallback",
            provider="fallback_service",
            timestamp=datetime.now()
        )
        
        response = LLMResponse(
            content=static_content,
            usage=usage,
            model="static_fallback",
            provider="fallback_service",
            cached=False
        )
        
        return FallbackResult(
            success=True,
            response=response,
            fallback_strategy=FallbackStrategy.STATIC_RESPONSE,
            degraded=True,
            cost_savings=0.05  # Estimated savings
        )
    
    def _get_model_for_task(self, model_type: ModelType, prefer_local: bool = False) -> str:
        """Get the primary model for a task type."""
        models = self.fallback_models.get(model_type, [])
        
        if prefer_local:
            for model in models:
                if 'ollama' in model:
                    return model
        
        return models[0] if models else "gpt-3.5-turbo"
    
    def _get_provider_from_model(self, model: str) -> str:
        """Extract provider from model name."""
        if 'gpt' in model.lower():
            return 'openai'
        elif 'claude' in model.lower():
            return 'anthropic'
        elif 'ollama' in model.lower():
            return 'ollama'
        else:
            return 'unknown'
    
    def _can_call_provider(self, provider: str) -> bool:
        """Check if provider can be called (circuit breaker check)."""
        if provider not in self._circuit_breakers:
            self._circuit_breakers[provider] = CircuitBreakerState()
        
        circuit = self._circuit_breakers[provider]
        now = datetime.now()
        
        if circuit.state == CircuitState.CLOSED:
            return True
        
        elif circuit.state == CircuitState.OPEN:
            # Check if we should transition to half-open
            if (circuit.last_failure_time and 
                now - circuit.last_failure_time > timedelta(seconds=circuit.recovery_timeout)):
                circuit.state = CircuitState.HALF_OPEN
                circuit.half_open_calls = 0
                return True
            return False
        
        elif circuit.state == CircuitState.HALF_OPEN:
            return circuit.half_open_calls < circuit.half_open_max_calls
        
        return False
    
    async def _record_success(self, provider: str):
        """Record successful operation for circuit breaker."""
        if provider not in self._circuit_breakers:
            self._circuit_breakers[provider] = CircuitBreakerState()
        
        circuit = self._circuit_breakers[provider]
        circuit.last_success_time = datetime.now()
        circuit.failure_count = 0
        
        if circuit.state == CircuitState.HALF_OPEN:
            circuit.state = CircuitState.CLOSED
        
        # Persist state
        await self._save_circuit_breaker_state(provider, circuit)
    
    async def _record_failure(self, provider: str, error_context: ErrorContext):
        """Record failed operation for circuit breaker."""
        if provider not in self._circuit_breakers:
            self._circuit_breakers[provider] = CircuitBreakerState()
        
        circuit = self._circuit_breakers[provider]
        circuit.failure_count += 1
        circuit.last_failure_time = datetime.now()
        
        # Only count severe failures towards circuit breaker
        if error_context.severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
            if circuit.failure_count >= circuit.failure_threshold:
                circuit.state = CircuitState.OPEN
                logger.warning(f"Circuit breaker OPENED for provider {provider}")
        
        # Track error statistics
        error_key = f"{provider}:{error_context.error_type}"
        self._error_counts[error_key] += 1
        
        # Persist state
        await self._save_circuit_breaker_state(provider, circuit)
    
    async def _load_circuit_breaker_states(self):
        """Load circuit breaker states from Redis."""
        if not self.redis_manager:
            return
        
        try:
            keys = await self.redis_manager.keys("circuit_breaker:*")
            
            for key in keys:
                data = await self.redis_manager.get(key)
                if data:
                    provider = key.split(":", 1)[1]
                    state_data = json.loads(data)
                    
                    circuit = CircuitBreakerState(
                        state=CircuitState(state_data['state']),
                        failure_count=state_data['failure_count'],
                        last_failure_time=datetime.fromisoformat(state_data['last_failure_time']) if state_data.get('last_failure_time') else None,
                        last_success_time=datetime.fromisoformat(state_data['last_success_time']) if state_data.get('last_success_time') else None
                    )
                    
                    self._circuit_breakers[provider] = circuit
                    
        except Exception as e:
            logger.warning(f"Failed to load circuit breaker states: {e}")
    
    async def _save_circuit_breaker_state(self, provider: str, circuit: CircuitBreakerState):
        """Save circuit breaker state to Redis."""
        if not self.redis_manager:
            return
        
        try:
            state_data = {
                'state': circuit.state.value,
                'failure_count': circuit.failure_count,
                'last_failure_time': circuit.last_failure_time.isoformat() if circuit.last_failure_time else None,
                'last_success_time': circuit.last_success_time.isoformat() if circuit.last_success_time else None
            }
            
            await self.redis_manager.setex(
                f"circuit_breaker:{provider}",
                86400,  # 24 hours TTL
                json.dumps(state_data)
            )
            
        except Exception as e:
            logger.warning(f"Failed to save circuit breaker state: {e}")
    
    async def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health status."""
        await self.initialize()
        
        health_status = {
            'overall_status': 'healthy',
            'providers': {},
            'error_summary': {},
            'fallback_stats': {
                'total_fallbacks_24h': 0,
                'successful_fallbacks_24h': 0,
                'cost_savings_24h': 0.0
            }
        }
        
        # Provider health
        for provider, circuit in self._circuit_breakers.items():
            health_status['providers'][provider] = {
                'status': circuit.state.value,
                'failure_count': circuit.failure_count,
                'last_failure': circuit.last_failure_time.isoformat() if circuit.last_failure_time else None,
                'last_success': circuit.last_success_time.isoformat() if circuit.last_success_time else None
            }
            
            if circuit.state == CircuitState.OPEN:
                health_status['overall_status'] = 'degraded'
        
        # Error summary
        health_status['error_summary'] = dict(self._error_counts)
        
        return health_status
    
    async def reset_circuit_breaker(self, provider: str):
        """Manually reset a circuit breaker."""
        if provider in self._circuit_breakers:
            circuit = self._circuit_breakers[provider]
            circuit.state = CircuitState.CLOSED
            circuit.failure_count = 0
            circuit.last_failure_time = None
            
            await self._save_circuit_breaker_state(provider, circuit)
            logger.info(f"Circuit breaker reset for provider {provider}")


# Global fallback service instance
_fallback_service: Optional[LLMFallbackService] = None


async def get_fallback_service() -> LLMFallbackService:
    """Get or create the global fallback service."""
    global _fallback_service
    
    if _fallback_service is None:
        _fallback_service = LLMFallbackService()
        await _fallback_service.initialize()
    
    return _fallback_service


# Context manager for fallback-aware operations
@asynccontextmanager
async def fallback_context(
    operation_name: str,
    context: Dict[str, Any],
    max_fallback_attempts: int = 3
):
    """Context manager for operations with automatic fallback handling."""
    fallback_service = await get_fallback_service()
    
    start_time = time.time()
    try:
        yield fallback_service
        
        # Record successful operation
        elapsed = time.time() - start_time
        logger.debug(f"Operation {operation_name} completed successfully in {elapsed:.2f}s")
        
    except Exception as e:
        # Log operation failure
        elapsed = time.time() - start_time
        logger.warning(f"Operation {operation_name} failed after {elapsed:.2f}s: {e}")
        raise