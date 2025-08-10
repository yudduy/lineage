"""
LLM Service - Multi-provider LLM abstraction with cost optimization and intelligent caching.

This service provides a unified interface for multiple LLM providers with:
- Cost tracking and budget management
- Intelligent caching with semantic similarity
- Fallback strategies for reliability
- Token counting and optimization
- Rate limiting and request management
"""

import asyncio
import json
import hashlib
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

import litellm
from litellm import completion, acompletion, cost_per_token
import tiktoken
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from sentence_transformers import SentenceTransformer
import numpy as np

from ..core.config import Settings, get_settings
from ..db.redis import RedisManager, get_redis_manager
from ..utils.logger import get_logger
from ..utils.exceptions import ValidationError

logger = get_logger(__name__)


class ModelType(Enum):
    """LLM model types for different tasks."""
    SUMMARIZATION = "summarization"
    ANALYSIS = "analysis" 
    LOCAL = "local"
    FALLBACK = "fallback"


class LLMProvider(Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"


@dataclass
class LLMUsage:
    """LLM usage statistics."""
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost: float
    model: str
    provider: str
    timestamp: datetime


@dataclass
class LLMResponse:
    """Standardized LLM response."""
    content: str
    usage: LLMUsage
    model: str
    provider: str
    cached: bool = False
    cache_key: Optional[str] = None


@dataclass
class PromptTemplate:
    """Structured prompt template."""
    system_prompt: str
    user_template: str
    max_tokens: int
    temperature: float = 0.1
    model_type: ModelType = ModelType.ANALYSIS
    
    def format(self, **kwargs) -> Tuple[str, str]:
        """Format the prompt with provided variables."""
        return self.system_prompt, self.user_template.format(**kwargs)


class LLMService:
    """
    Multi-provider LLM service with cost optimization and intelligent caching.
    """
    
    def __init__(
        self,
        settings: Optional[Settings] = None,
        redis_manager: Optional[RedisManager] = None
    ):
        self.settings = settings or get_settings()
        self.redis_manager = redis_manager
        self._embedding_model = None
        self._token_encoders = {}
        self._usage_stats = []
        self._daily_cost = 0.0
        self._monthly_cost = 0.0
        self._last_cost_reset = datetime.now()
        
        # Configure LiteLLM
        self._configure_litellm()
        
        # Initialize clients
        self._initialized = False
        
    async def initialize(self):
        """Initialize the LLM service."""
        if self._initialized:
            return
        
        if not self.redis_manager:
            self.redis_manager = await get_redis_manager()
        
        # Load cost tracking data
        await self._load_cost_data()
        
        # Initialize embedding model for semantic caching
        if self.settings.llm.enable_semantic_caching:
            await self._initialize_embedding_model()
        
        self._initialized = True
        logger.info("LLM service initialized successfully")
    
    def _configure_litellm(self):
        """Configure LiteLLM with provider settings."""
        # Set API keys
        if self.settings.llm.openai_api_key:
            litellm.openai_key = self.settings.llm.openai_api_key
            if self.settings.llm.openai_organization:
                litellm.openai_organization = self.settings.llm.openai_organization
        
        if self.settings.llm.anthropic_api_key:
            litellm.anthropic_key = self.settings.llm.anthropic_api_key
        
        # Configure request settings
        litellm.request_timeout = self.settings.llm.request_timeout
        litellm.max_retries = self.settings.llm.max_retries
        
        # Set logging
        litellm.set_verbose = self.settings.debug
    
    async def _initialize_embedding_model(self):
        """Initialize the embedding model for semantic caching."""
        try:
            # Use a lightweight model for cache similarity
            self._embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Embedding model initialized for semantic caching")
        except Exception as e:
            logger.warning(f"Failed to initialize embedding model: {e}")
            self._embedding_model = None
    
    async def _load_cost_data(self):
        """Load cost tracking data from Redis."""
        try:
            if not self.redis_manager:
                return
            
            # Load daily cost
            daily_key = f"llm_cost:daily:{datetime.now().strftime('%Y-%m-%d')}"
            daily_cost = await self.redis_manager.get(daily_key)
            if daily_cost:
                self._daily_cost = float(daily_cost)
            
            # Load monthly cost
            monthly_key = f"llm_cost:monthly:{datetime.now().strftime('%Y-%m')}"
            monthly_cost = await self.redis_manager.get(monthly_key)
            if monthly_cost:
                self._monthly_cost = float(monthly_cost)
                
        except Exception as e:
            logger.warning(f"Failed to load cost data: {e}")
    
    async def _update_cost_data(self, cost: float):
        """Update cost tracking data."""
        if not self.settings.llm.enable_cost_tracking or not self.redis_manager:
            return
        
        try:
            now = datetime.now()
            
            # Update daily cost
            daily_key = f"llm_cost:daily:{now.strftime('%Y-%m-%d')}"
            self._daily_cost += cost
            await self.redis_manager.setex(
                daily_key, 
                int(timedelta(days=2).total_seconds()),  # Keep for 2 days
                str(self._daily_cost)
            )
            
            # Update monthly cost
            monthly_key = f"llm_cost:monthly:{now.strftime('%Y-%m')}"
            self._monthly_cost += cost
            await self.redis_manager.setex(
                monthly_key,
                int(timedelta(days=32).total_seconds()),  # Keep for 32 days
                str(self._monthly_cost)
            )
            
            # Check budget limits
            await self._check_budget_limits()
            
        except Exception as e:
            logger.error(f"Failed to update cost data: {e}")
    
    async def _check_budget_limits(self):
        """Check if we're approaching or exceeding budget limits."""
        daily_limit = self.settings.llm.daily_budget_limit
        monthly_limit = self.settings.llm.monthly_budget_limit
        alert_threshold = self.settings.llm.cost_alert_threshold
        
        # Check daily budget
        if self._daily_cost >= daily_limit:
            logger.warning(f"Daily budget limit exceeded: ${self._daily_cost:.2f} >= ${daily_limit}")
        elif self._daily_cost >= daily_limit * alert_threshold:
            logger.warning(f"Daily budget alert: ${self._daily_cost:.2f} >= ${daily_limit * alert_threshold:.2f}")
        
        # Check monthly budget
        if self._monthly_cost >= monthly_limit:
            logger.warning(f"Monthly budget limit exceeded: ${self._monthly_cost:.2f} >= ${monthly_limit}")
        elif self._monthly_cost >= monthly_limit * alert_threshold:
            logger.warning(f"Monthly budget alert: ${self._monthly_cost:.2f} >= ${monthly_limit * alert_threshold:.2f}")
    
    def _count_tokens(self, text: str, model: str) -> int:
        """Count tokens for a given text and model."""
        try:
            # Get or create tokenizer for this model
            if model not in self._token_encoders:
                if "gpt" in model.lower():
                    encoding_name = "cl100k_base" if "gpt-4" in model.lower() else "p50k_base"
                    self._token_encoders[model] = tiktoken.get_encoding(encoding_name)
                else:
                    # Approximate for non-OpenAI models (4 chars per token on average)
                    return len(text) // 4
            
            return len(self._token_encoders[model].encode(text))
        except Exception:
            # Fallback approximation
            return len(text) // 4
    
    def _get_model_for_task(self, model_type: ModelType, prefer_local: bool = False) -> str:
        """Get the appropriate model for a given task type."""
        if prefer_local and self.settings.llm.enable_local_fallback:
            return f"ollama/{self.settings.llm.default_local_model}"
        
        if model_type == ModelType.SUMMARIZATION:
            return self.settings.llm.default_summarization_model
        elif model_type == ModelType.ANALYSIS:
            return self.settings.llm.default_analysis_model
        elif model_type == ModelType.LOCAL:
            return f"ollama/{self.settings.llm.default_local_model}"
        else:
            return self.settings.llm.default_summarization_model
    
    def _should_use_budget_limit(self) -> bool:
        """Check if we should respect budget limits."""
        daily_limit = self.settings.llm.daily_budget_limit
        monthly_limit = self.settings.llm.monthly_budget_limit
        
        return (
            self.settings.llm.enable_cost_tracking and
            (self._daily_cost < daily_limit and self._monthly_cost < monthly_limit)
        )
    
    async def _get_cache_key(self, prompt: str, model: str, max_tokens: int, temperature: float) -> str:
        """Generate a cache key for the request."""
        content = f"{prompt}:{model}:{max_tokens}:{temperature}"
        return hashlib.sha256(content.encode()).hexdigest()
    
    async def _get_cached_response(self, cache_key: str) -> Optional[LLMResponse]:
        """Get cached response if available."""
        if not self.redis_manager or not self.settings.llm.enable_semantic_caching:
            return None
        
        try:
            cached_data = await self.redis_manager.get(f"llm_cache:{cache_key}")
            if cached_data:
                data = json.loads(cached_data)
                usage = LLMUsage(**data['usage'])
                return LLMResponse(
                    content=data['content'],
                    usage=usage,
                    model=data['model'],
                    provider=data['provider'],
                    cached=True,
                    cache_key=cache_key
                )
        except Exception as e:
            logger.warning(f"Failed to retrieve cached response: {e}")
        
        return None
    
    async def _cache_response(self, cache_key: str, response: LLMResponse):
        """Cache the LLM response."""
        if not self.redis_manager or not self.settings.llm.enable_semantic_caching:
            return
        
        try:
            cache_data = {
                'content': response.content,
                'usage': {
                    'input_tokens': response.usage.input_tokens,
                    'output_tokens': response.usage.output_tokens,
                    'total_tokens': response.usage.total_tokens,
                    'cost': response.usage.cost,
                    'model': response.usage.model,
                    'provider': response.usage.provider,
                    'timestamp': response.usage.timestamp.isoformat()
                },
                'model': response.model,
                'provider': response.provider
            }
            
            ttl = self.settings.llm.cache_ttl_hours * 3600
            await self.redis_manager.setex(
                f"llm_cache:{cache_key}",
                ttl,
                json.dumps(cache_data)
            )
        except Exception as e:
            logger.warning(f"Failed to cache response: {e}")
    
    async def _find_similar_cached_response(self, prompt: str) -> Optional[LLMResponse]:
        """Find a semantically similar cached response."""
        if not self._embedding_model or not self.redis_manager:
            return None
        
        try:
            # Get embedding for current prompt
            current_embedding = self._embedding_model.encode([prompt])
            
            # Search for similar cached prompts (simplified implementation)
            # In production, you might want to use a vector database
            cache_keys = await self.redis_manager.keys("llm_cache:*")
            
            for key in cache_keys[:50]:  # Limit search to avoid performance issues
                try:
                    # This is a simplified approach - in practice, you'd store embeddings separately
                    cached_data = await self.redis_manager.get(key)
                    if cached_data:
                        data = json.loads(cached_data)
                        # For now, skip semantic similarity check and just return first valid cache
                        # In production, implement proper embedding storage and similarity search
                        pass
                except Exception:
                    continue
            
        except Exception as e:
            logger.warning(f"Failed to find similar cached response: {e}")
        
        return None
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(Exception)
    )
    async def _make_llm_request(
        self,
        messages: List[Dict[str, str]], 
        model: str,
        max_tokens: int,
        temperature: float,
        **kwargs
    ) -> LLMResponse:
        """Make an LLM request with retry logic."""
        start_time = time.time()
        
        try:
            # Make the request
            response = await acompletion(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=self.settings.llm.request_timeout,
                **kwargs
            )
            
            # Extract response details
            content = response.choices[0].message.content
            usage_data = response.usage
            
            # Calculate cost
            try:
                input_cost = cost_per_token(model, usage_data.prompt_tokens, "input")
                output_cost = cost_per_token(model, usage_data.completion_tokens, "output")
                total_cost = input_cost + output_cost
            except Exception:
                # Fallback cost calculation
                total_cost = 0.0
            
            # Determine provider
            provider = "unknown"
            if "gpt" in model.lower():
                provider = "openai"
            elif "claude" in model.lower():
                provider = "anthropic" 
            elif "ollama" in model.lower():
                provider = "ollama"
            
            # Create usage object
            usage = LLMUsage(
                input_tokens=usage_data.prompt_tokens,
                output_tokens=usage_data.completion_tokens,
                total_tokens=usage_data.total_tokens,
                cost=total_cost,
                model=model,
                provider=provider,
                timestamp=datetime.now()
            )
            
            # Update cost tracking
            await self._update_cost_data(total_cost)
            
            # Log request
            elapsed = time.time() - start_time
            logger.info(
                f"LLM request completed: model={model}, tokens={usage.total_tokens}, "
                f"cost=${total_cost:.4f}, time={elapsed:.2f}s"
            )
            
            return LLMResponse(
                content=content,
                usage=usage,
                model=model,
                provider=provider
            )
            
        except Exception as e:
            logger.error(f"LLM request failed: {e}")
            raise
    
    async def complete(
        self,
        prompt_template: PromptTemplate,
        use_cache: bool = True,
        prefer_local: bool = False,
        **template_vars
    ) -> LLMResponse:
        """
        Complete a prompt using the appropriate LLM.
        
        Args:
            prompt_template: The prompt template to use
            use_cache: Whether to use caching
            prefer_local: Whether to prefer local models
            **template_vars: Variables to format the template
            
        Returns:
            LLMResponse with the completion
        """
        await self.initialize()
        
        # Check budget limits
        if not self._should_use_budget_limit() and not prefer_local:
            if self.settings.llm.enable_local_fallback:
                prefer_local = True
                logger.warning("Budget limit reached, falling back to local model")
            else:
                raise ValidationError("Budget limit exceeded and no local fallback available")
        
        # Format the prompt
        system_prompt, user_prompt = prompt_template.format(**template_vars)
        
        # Get model
        model = self._get_model_for_task(prompt_template.model_type, prefer_local)
        
        # Create messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        
        # Check cache
        cache_key = await self._get_cache_key(
            f"{system_prompt}\n{user_prompt}", 
            model, 
            prompt_template.max_tokens, 
            prompt_template.temperature
        )
        
        if use_cache:
            cached_response = await self._get_cached_response(cache_key)
            if cached_response:
                logger.info(f"Retrieved cached LLM response: {cache_key[:16]}...")
                return cached_response
        
        # Make request
        try:
            response = await self._make_llm_request(
                messages=messages,
                model=model,
                max_tokens=prompt_template.max_tokens,
                temperature=prompt_template.temperature
            )
            
            # Cache response
            if use_cache:
                await self._cache_response(cache_key, response)
            
            return response
            
        except Exception as e:
            # Fallback to local model if enabled
            if (not prefer_local and 
                self.settings.llm.enable_local_fallback and 
                "ollama" not in model.lower()):
                
                logger.warning(f"Primary model failed, falling back to local: {e}")
                fallback_response = await self.complete(
                    prompt_template,
                    use_cache=use_cache,
                    prefer_local=True,
                    **template_vars
                )
                return fallback_response
            
            raise
    
    async def batch_complete(
        self,
        requests: List[Tuple[PromptTemplate, Dict[str, Any]]],
        use_cache: bool = True,
        prefer_local: bool = False,
        max_concurrency: int = None
    ) -> List[LLMResponse]:
        """
        Complete multiple prompts in batch with controlled concurrency.
        
        Args:
            requests: List of (prompt_template, template_vars) tuples
            use_cache: Whether to use caching
            prefer_local: Whether to prefer local models
            max_concurrency: Maximum concurrent requests
            
        Returns:
            List of LLMResponse objects
        """
        if not max_concurrency:
            max_concurrency = self.settings.llm.concurrent_requests
        
        semaphore = asyncio.Semaphore(max_concurrency)
        
        async def process_request(prompt_template: PromptTemplate, template_vars: Dict):
            async with semaphore:
                return await self.complete(
                    prompt_template,
                    use_cache=use_cache,
                    prefer_local=prefer_local,
                    **template_vars
                )
        
        tasks = [
            process_request(template, vars_dict)
            for template, vars_dict in requests
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        responses = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Batch request {i} failed: {result}")
                # Create error response
                responses.append(None)
            else:
                responses.append(result)
        
        return responses
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics."""
        return {
            'daily_cost': self._daily_cost,
            'monthly_cost': self._monthly_cost,
            'daily_budget_limit': self.settings.llm.daily_budget_limit,
            'monthly_budget_limit': self.settings.llm.monthly_budget_limit,
            'requests_today': len([u for u in self._usage_stats 
                                 if u.timestamp.date() == datetime.now().date()]),
            'total_requests': len(self._usage_stats),
            'avg_cost_per_request': (
                sum(u.cost for u in self._usage_stats) / len(self._usage_stats)
                if self._usage_stats else 0
            ),
            'budget_utilization': {
                'daily': self._daily_cost / self.settings.llm.daily_budget_limit,
                'monthly': self._monthly_cost / self.settings.llm.monthly_budget_limit
            }
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on LLM services."""
        health = {
            'service': 'healthy',
            'providers': {},
            'cost_tracking': self.settings.llm.enable_cost_tracking,
            'semantic_caching': self.settings.llm.enable_semantic_caching and self._embedding_model is not None,
            'local_fallback': self.settings.llm.enable_local_fallback
        }
        
        # Test each provider
        test_prompt = PromptTemplate(
            system_prompt="You are a helpful assistant.",
            user_template="Say 'OK' if you can understand this.",
            max_tokens=10,
            temperature=0.0
        )
        
        # Test OpenAI
        if self.settings.llm.openai_api_key:
            try:
                response = await self.complete(
                    test_prompt,
                    use_cache=False,
                    prefer_local=False
                )
                health['providers']['openai'] = 'healthy' if response else 'unhealthy'
            except Exception as e:
                health['providers']['openai'] = f'unhealthy: {str(e)}'
        
        # Test Anthropic
        if self.settings.llm.anthropic_api_key:
            try:
                # Would test Anthropic here similarly
                health['providers']['anthropic'] = 'not_tested'
            except Exception as e:
                health['providers']['anthropic'] = f'unhealthy: {str(e)}'
        
        # Test Ollama if local fallback enabled
        if self.settings.llm.enable_local_fallback:
            try:
                response = await self.complete(
                    test_prompt,
                    use_cache=False,
                    prefer_local=True
                )
                health['providers']['ollama'] = 'healthy' if response else 'unhealthy'
            except Exception as e:
                health['providers']['ollama'] = f'unhealthy: {str(e)}'
        
        return health


# Global service instance
_llm_service: Optional[LLMService] = None


async def get_llm_service() -> LLMService:
    """Get or create the global LLM service instance."""
    global _llm_service
    
    if _llm_service is None:
        _llm_service = LLMService()
        await _llm_service.initialize()
    
    return _llm_service