"""
Enhanced LLM Service with Fallback Integration - Production-ready LLM service with comprehensive error handling.

This module extends the base LLM service with:
- Integrated fallback strategies
- Circuit breaker patterns
- Graceful degradation
- Enhanced error handling
- Comprehensive monitoring
"""

import asyncio
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
import logging

from .llm_service import LLMService, LLMResponse, PromptTemplate, get_llm_service
from .llm_fallback import get_fallback_service, fallback_context, FallbackResult
from .llm_cost_manager import get_cost_manager, CostCategory
from ..utils.logger import get_logger
from ..utils.exceptions import ValidationError

logger = get_logger(__name__)


class EnhancedLLMService:
    """
    Enhanced LLM service with integrated fallback strategies and error handling.
    """
    
    def __init__(self, base_service: Optional[LLMService] = None):
        self.base_service = base_service
        self._fallback_service = None
        self._cost_manager = None
        self._initialized = False
        
        # Enhanced metrics
        self._fallback_stats = {
            'total_requests': 0,
            'fallback_used': 0,
            'fallback_success_rate': 0.0,
            'cost_savings_from_fallback': 0.0
        }
    
    async def initialize(self):
        """Initialize the enhanced service."""
        if self._initialized:
            return
        
        if not self.base_service:
            self.base_service = await get_llm_service()
        
        self._fallback_service = await get_fallback_service()
        self._cost_manager = await get_cost_manager()
        
        self._initialized = True
        logger.info("Enhanced LLM Service initialized with fallback support")
    
    async def complete_with_fallback(
        self,
        prompt_template: PromptTemplate,
        use_cache: bool = True,
        prefer_local: bool = False,
        max_fallback_attempts: int = 3,
        **template_vars
    ) -> LLMResponse:
        """
        Complete a prompt with comprehensive fallback support.
        
        Args:
            prompt_template: The prompt template to use
            use_cache: Whether to use caching
            prefer_local: Whether to prefer local models
            max_fallback_attempts: Maximum fallback attempts
            **template_vars: Variables to format the template
            
        Returns:
            LLMResponse with the completion
            
        Raises:
            ValidationError: If all strategies fail and no fallback possible
        """
        await self.initialize()
        
        self._fallback_stats['total_requests'] += 1
        start_time = time.time()
        
        # Prepare context for fallback service
        context = {
            'template_type': prompt_template.model_type.value,
            'use_cache': use_cache,
            'prefer_local': prefer_local,
            'template_vars': template_vars
        }
        
        # Define the primary operation
        async def primary_operation(template, **kwargs):
            return await self.base_service.complete(
                template,
                use_cache=kwargs.get('use_cache', True),
                prefer_local=kwargs.get('prefer_local', False),
                **{k: v for k, v in kwargs.items() if k not in ['use_cache', 'prefer_local']}
            )
        
        # Execute with fallback protection
        fallback_result = await self._fallback_service.execute_with_fallback(
            operation=primary_operation,
            template=prompt_template,
            context=context,
            max_fallback_attempts=max_fallback_attempts,
            use_cache=use_cache,
            prefer_local=prefer_local,
            **template_vars
        )
        
        # Update statistics
        if fallback_result.fallback_strategy is not None:
            self._fallback_stats['fallback_used'] += 1
            self._fallback_stats['cost_savings_from_fallback'] += fallback_result.cost_savings
        
        self._fallback_stats['fallback_success_rate'] = (
            self._fallback_stats['fallback_used'] / self._fallback_stats['total_requests']
        ) if self._fallback_stats['total_requests'] > 0 else 0.0
        
        # Handle result
        if fallback_result.success and fallback_result.response:
            # Log fallback usage
            elapsed = time.time() - start_time
            if fallback_result.fallback_strategy:
                logger.info(
                    f"LLM request completed with fallback ({fallback_result.fallback_strategy.value}): "
                    f"cost_savings=${fallback_result.cost_savings:.4f}, time={elapsed:.2f}s"
                )
            
            return fallback_result.response
        
        elif fallback_result.success and fallback_result.degraded:
            # Return degraded response (e.g., static response)
            from .llm_service import LLMUsage
            
            # Create a minimal response for degraded scenarios
            usage = LLMUsage(
                input_tokens=0,
                output_tokens=50,  # Estimated
                total_tokens=50,
                cost=0.0,
                model="fallback",
                provider="fallback_service",
                timestamp=datetime.now()
            )
            
            response = LLMResponse(
                content=fallback_result.error_message or "Analysis temporarily unavailable",
                usage=usage,
                model="fallback",
                provider="fallback_service",
                cached=False
            )
            
            logger.info(f"LLM request completed with degraded response: {fallback_result.fallback_strategy.value if fallback_result.fallback_strategy else 'unknown'}")
            
            return response
        
        else:
            # All strategies failed
            elapsed = time.time() - start_time
            error_msg = (
                f"LLM request failed after {elapsed:.2f}s: "
                f"{fallback_result.error_message or 'All fallback strategies exhausted'}"
            )
            logger.error(error_msg)
            raise ValidationError(error_msg)
    
    async def batch_complete_with_fallback(
        self,
        requests: List[Tuple[PromptTemplate, Dict[str, Any]]],
        use_cache: bool = True,
        prefer_local: bool = False,
        max_concurrency: int = None,
        error_handling: str = 'continue'  # 'continue', 'fail_fast', 'best_effort'
    ) -> List[Union[LLMResponse, Exception]]:
        """
        Complete multiple prompts with fallback support and error handling strategies.
        
        Args:
            requests: List of (prompt_template, template_vars) tuples
            use_cache: Whether to use caching
            prefer_local: Whether to prefer local models
            max_concurrency: Maximum concurrent requests
            error_handling: How to handle errors ('continue', 'fail_fast', 'best_effort')
            
        Returns:
            List of LLMResponse objects or exceptions
        """
        await self.initialize()
        
        if not max_concurrency:
            max_concurrency = min(5, len(requests))  # Conservative default
        
        semaphore = asyncio.Semaphore(max_concurrency)
        
        async def process_request(prompt_template: PromptTemplate, template_vars: Dict):
            async with semaphore:
                try:
                    return await self.complete_with_fallback(
                        prompt_template,
                        use_cache=use_cache,
                        prefer_local=prefer_local,
                        **template_vars
                    )
                except Exception as e:
                    if error_handling == 'fail_fast':
                        raise
                    elif error_handling == 'best_effort':
                        # Return a minimal response instead of exception
                        from .llm_service import LLMUsage
                        
                        usage = LLMUsage(
                            input_tokens=0,
                            output_tokens=0,
                            total_tokens=0,
                            cost=0.0,
                            model="error",
                            provider="error",
                            timestamp=datetime.now()
                        )
                        
                        return LLMResponse(
                            content=f"Analysis failed: {str(e)}",
                            usage=usage,
                            model="error",
                            provider="error",
                            cached=False
                        )
                    else:  # continue
                        return e
        
        # Process all requests
        tasks = [
            process_request(template, vars_dict)
            for template, vars_dict in requests
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=(error_handling != 'fail_fast'))
        
        # Log batch completion statistics
        successful = sum(1 for r in results if isinstance(r, LLMResponse))
        failed = len(results) - successful
        
        logger.info(f"Batch completion finished: {successful}/{len(results)} successful, "
                   f"{self._fallback_stats['fallback_used']} used fallback")
        
        return results
    
    async def health_check_with_fallback(self) -> Dict[str, Any]:
        """Comprehensive health check including fallback system status."""
        await self.initialize()
        
        # Get base service health
        base_health = await self.base_service.health_check()
        
        # Get fallback system health
        fallback_health = await self._fallback_service.get_system_health()
        
        # Get enhanced statistics
        enhanced_stats = {
            'fallback_statistics': self._fallback_stats,
            'base_service': base_health,
            'fallback_system': fallback_health
        }
        
        # Determine overall health status
        overall_status = 'healthy'
        if fallback_health['overall_status'] == 'degraded':
            overall_status = 'degraded'
        elif any(provider['status'] == 'open' for provider in fallback_health['providers'].values()):
            overall_status = 'degraded'
        
        return {
            'status': overall_status,
            'timestamp': datetime.now().isoformat(),
            'enhanced_service': True,
            'fallback_enabled': True,
            **enhanced_stats
        }
    
    async def reset_circuit_breaker(self, provider: str) -> bool:
        """Reset circuit breaker for a specific provider."""
        await self.initialize()
        
        try:
            await self._fallback_service.reset_circuit_breaker(provider)
            logger.info(f"Circuit breaker reset for provider {provider}")
            return True
        except Exception as e:
            logger.error(f"Failed to reset circuit breaker for {provider}: {e}")
            return False
    
    def get_enhanced_usage_stats(self) -> Dict[str, Any]:
        """Get comprehensive usage statistics including fallback metrics."""
        base_stats = self.base_service.get_usage_stats() if self.base_service else {}
        
        return {
            **base_stats,
            'enhanced_features': {
                'fallback_statistics': self._fallback_stats,
                'fallback_enabled': True,
                'circuit_breakers_active': len(self._fallback_service._circuit_breakers) if self._fallback_service else 0
            }
        }
    
    # Convenience methods that wrap base service functionality
    
    async def complete(self, *args, **kwargs) -> LLMResponse:
        """Standard complete method - uses fallback by default."""
        return await self.complete_with_fallback(*args, **kwargs)
    
    async def batch_complete(self, *args, **kwargs) -> List[Union[LLMResponse, Exception]]:
        """Standard batch complete method - uses fallback by default."""
        return await self.batch_complete_with_fallback(*args, **kwargs)


# Global enhanced service instance
_enhanced_service: Optional[EnhancedLLMService] = None


async def get_enhanced_llm_service() -> EnhancedLLMService:
    """Get or create the global enhanced LLM service."""
    global _enhanced_service
    
    if _enhanced_service is None:
        _enhanced_service = EnhancedLLMService()
        await _enhanced_service.initialize()
    
    return _enhanced_service


# Convenience function for backward compatibility
async def get_llm_service_with_fallback() -> EnhancedLLMService:
    """Alias for get_enhanced_llm_service for backward compatibility."""
    return await get_enhanced_llm_service()