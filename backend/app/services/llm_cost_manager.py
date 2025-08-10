"""
LLM Cost Management Service - Advanced cost tracking, optimization, and budget management.

This service provides:
- Real-time cost tracking across all LLM providers
- Budget management with alerts and enforcement
- Cost optimization through intelligent model selection
- Usage analytics and reporting
- Cost prediction and forecasting
"""

import asyncio
import json
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from decimal import Decimal, ROUND_HALF_UP
import logging

from ..core.config import Settings, get_settings
from ..db.redis import RedisManager, get_redis_manager
from ..utils.logger import get_logger

logger = get_logger(__name__)


class CostCategory(Enum):
    """Cost categories for different types of LLM usage."""
    PAPER_ANALYSIS = "paper_analysis"
    CITATION_ANALYSIS = "citation_analysis"
    SUMMARIZATION = "summarization"
    RESEARCH_TRAJECTORY = "research_trajectory"
    BATCH_PROCESSING = "batch_processing"
    API_REQUEST = "api_request"
    BACKGROUND_TASK = "background_task"


class AlertLevel(Enum):
    """Alert levels for budget notifications."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    BUDGET_EXCEEDED = "budget_exceeded"


@dataclass
class CostUsage:
    """Detailed cost usage record."""
    timestamp: datetime
    model: str
    provider: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost: float
    category: CostCategory
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    cached: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['category'] = self.category.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CostUsage':
        """Create from dictionary."""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        data['category'] = CostCategory(data['category'])
        return cls(**data)


@dataclass
class BudgetAlert:
    """Budget alert information."""
    level: AlertLevel
    message: str
    current_spend: float
    budget_limit: float
    utilization_percentage: float
    period: str  # 'daily' or 'monthly'
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data['level'] = self.level.value
        data['timestamp'] = self.timestamp.isoformat()
        return data


@dataclass
class CostAnalytics:
    """Cost analytics and insights."""
    period_start: datetime
    period_end: datetime
    total_cost: float
    total_requests: int
    avg_cost_per_request: float
    cost_by_category: Dict[CostCategory, float]
    cost_by_model: Dict[str, float]
    cost_by_provider: Dict[str, float]
    token_usage: Dict[str, int]
    cache_hit_rate: float
    cost_savings_from_cache: float
    top_expensive_requests: List[Dict[str, Any]]
    usage_trends: Dict[str, List[float]]  # Daily usage patterns
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data['period_start'] = self.period_start.isoformat()
        data['period_end'] = self.period_end.isoformat()
        data['cost_by_category'] = {k.value: v for k, v in self.cost_by_category.items()}
        return data


class ModelCostDatabase:
    """Database of model costs for accurate cost calculation."""
    
    # Cost per 1K tokens (input, output) in USD
    MODEL_COSTS = {
        # OpenAI models
        'gpt-4': (0.03, 0.06),
        'gpt-4-32k': (0.06, 0.12),
        'gpt-4-1106-preview': (0.01, 0.03),
        'gpt-4-0125-preview': (0.01, 0.03),
        'gpt-3.5-turbo': (0.001, 0.002),
        'gpt-3.5-turbo-16k': (0.003, 0.004),
        
        # Anthropic models
        'claude-3-opus-20240229': (0.015, 0.075),
        'claude-3-sonnet-20240229': (0.003, 0.015),
        'claude-3-haiku-20240307': (0.00025, 0.00125),
        'claude-2.1': (0.008, 0.024),
        'claude-2.0': (0.008, 0.024),
        'claude-instant-1.2': (0.0008, 0.0024),
        
        # Local models (Ollama) - essentially free but with compute costs
        'ollama/llama3': (0.0, 0.0),
        'ollama/codellama': (0.0, 0.0),
        'ollama/mistral': (0.0, 0.0),
    }
    
    @classmethod
    def get_cost_per_token(cls, model: str) -> Tuple[float, float]:
        """Get cost per token for input and output."""
        # Normalize model name
        model_key = model.lower()
        if model_key in cls.MODEL_COSTS:
            return cls.MODEL_COSTS[model_key]
        
        # Try partial matching
        for known_model, costs in cls.MODEL_COSTS.items():
            if known_model in model_key or any(part in model_key for part in known_model.split('-')):
                return costs
        
        # Default to GPT-4 pricing for unknown models (conservative estimate)
        logger.warning(f"Unknown model cost for {model}, using GPT-4 pricing as fallback")
        return cls.MODEL_COSTS['gpt-4']
    
    @classmethod
    def calculate_cost(cls, model: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate total cost for a request."""
        input_cost_per_k, output_cost_per_k = cls.get_cost_per_token(model)
        
        input_cost = (input_tokens / 1000.0) * input_cost_per_k
        output_cost = (output_tokens / 1000.0) * output_cost_per_k
        
        total_cost = input_cost + output_cost
        
        # Round to 6 decimal places for precision
        return float(Decimal(str(total_cost)).quantize(Decimal('0.000001'), rounding=ROUND_HALF_UP))


class LLMCostManager:
    """
    Advanced LLM cost management with tracking, budgets, and optimization.
    """
    
    def __init__(
        self,
        settings: Optional[Settings] = None,
        redis_manager: Optional[RedisManager] = None
    ):
        self.settings = settings or get_settings()
        self.redis_manager = redis_manager
        self._alerts_sent: Dict[str, datetime] = {}
        self._cost_cache: Dict[str, CostAnalytics] = {}
        self._initialized = False
    
    async def initialize(self):
        """Initialize the cost manager."""
        if self._initialized:
            return
        
        if not self.redis_manager:
            self.redis_manager = await get_redis_manager()
        
        self._initialized = True
        logger.info("LLM Cost Manager initialized")
    
    def _get_date_key(self, date_obj: date) -> str:
        """Get Redis key for a specific date."""
        return f"llm_cost:daily:{date_obj.strftime('%Y-%m-%d')}"
    
    def _get_month_key(self, date_obj: date) -> str:
        """Get Redis key for a specific month."""
        return f"llm_cost:monthly:{date_obj.strftime('%Y-%m')}"
    
    async def record_usage(
        self,
        model: str,
        provider: str,
        input_tokens: int,
        output_tokens: int,
        category: CostCategory,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        request_id: Optional[str] = None,
        cached: bool = False
    ) -> CostUsage:
        """Record LLM usage and calculate cost."""
        await self.initialize()
        
        # Calculate cost
        total_tokens = input_tokens + output_tokens
        cost = ModelCostDatabase.calculate_cost(model, input_tokens, output_tokens)
        
        # Create usage record
        usage = CostUsage(
            timestamp=datetime.now(),
            model=model,
            provider=provider,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            cost=cost,
            category=category,
            user_id=user_id,
            session_id=session_id,
            request_id=request_id,
            cached=cached
        )
        
        # Store in Redis
        await self._store_usage_record(usage)
        
        # Update daily and monthly totals
        await self._update_cost_totals(usage)
        
        # Check budget limits and send alerts if necessary
        await self._check_and_send_alerts()
        
        logger.debug(f"Recorded LLM usage: {model} - ${cost:.6f} ({total_tokens} tokens)")
        
        return usage
    
    async def _store_usage_record(self, usage: CostUsage):
        """Store individual usage record in Redis."""
        if not self.redis_manager:
            return
        
        try:
            # Store individual record
            record_key = f"llm_usage:{usage.timestamp.strftime('%Y%m%d')}:{usage.request_id or 'unknown'}"
            await self.redis_manager.setex(
                record_key,
                int(timedelta(days=90).total_seconds()),  # Keep for 90 days
                json.dumps(usage.to_dict())
            )
            
            # Add to daily list for analytics
            daily_list_key = f"llm_usage_list:{usage.timestamp.strftime('%Y-%m-%d')}"
            await self.redis_manager.lpush(daily_list_key, json.dumps(usage.to_dict()))
            await self.redis_manager.expire(daily_list_key, int(timedelta(days=90).total_seconds()))
            
        except Exception as e:
            logger.error(f"Failed to store usage record: {e}")
    
    async def _update_cost_totals(self, usage: CostUsage):
        """Update daily and monthly cost totals."""
        if not self.redis_manager:
            return
        
        try:
            today = usage.timestamp.date()
            
            # Update daily total
            daily_key = self._get_date_key(today)
            await self.redis_manager.incrbyfloat(daily_key, usage.cost)
            await self.redis_manager.expire(daily_key, int(timedelta(days=7).total_seconds()))
            
            # Update monthly total
            monthly_key = self._get_month_key(today)
            await self.redis_manager.incrbyfloat(monthly_key, usage.cost)
            await self.redis_manager.expire(monthly_key, int(timedelta(days=32).total_seconds()))
            
            # Update category totals
            category_key = f"llm_cost:category:{today.strftime('%Y-%m-%d')}:{usage.category.value}"
            await self.redis_manager.incrbyfloat(category_key, usage.cost)
            await self.redis_manager.expire(category_key, int(timedelta(days=7).total_seconds()))
            
        except Exception as e:
            logger.error(f"Failed to update cost totals: {e}")
    
    async def get_daily_cost(self, date_obj: Optional[date] = None) -> float:
        """Get total daily cost."""
        await self.initialize()
        
        if date_obj is None:
            date_obj = date.today()
        
        try:
            daily_key = self._get_date_key(date_obj)
            cost = await self.redis_manager.get(daily_key)
            return float(cost) if cost else 0.0
        except Exception as e:
            logger.error(f"Failed to get daily cost: {e}")
            return 0.0
    
    async def get_monthly_cost(self, date_obj: Optional[date] = None) -> float:
        """Get total monthly cost."""
        await self.initialize()
        
        if date_obj is None:
            date_obj = date.today()
        
        try:
            monthly_key = self._get_month_key(date_obj)
            cost = await self.redis_manager.get(monthly_key)
            return float(cost) if cost else 0.0
        except Exception as e:
            logger.error(f"Failed to get monthly cost: {e}")
            return 0.0
    
    async def _check_and_send_alerts(self):
        """Check budget limits and send alerts if necessary."""
        if not self.settings.llm.enable_cost_tracking:
            return
        
        daily_cost = await self.get_daily_cost()
        monthly_cost = await self.get_monthly_cost()
        
        daily_limit = self.settings.llm.daily_budget_limit
        monthly_limit = self.settings.llm.monthly_budget_limit
        alert_threshold = self.settings.llm.cost_alert_threshold
        
        alerts = []
        
        # Check daily budget
        daily_utilization = daily_cost / daily_limit if daily_limit > 0 else 0
        if daily_utilization >= 1.0:
            alerts.append(BudgetAlert(
                level=AlertLevel.BUDGET_EXCEEDED,
                message=f"Daily budget exceeded: ${daily_cost:.2f} / ${daily_limit:.2f}",
                current_spend=daily_cost,
                budget_limit=daily_limit,
                utilization_percentage=daily_utilization * 100,
                period="daily",
                timestamp=datetime.now()
            ))
        elif daily_utilization >= alert_threshold:
            alerts.append(BudgetAlert(
                level=AlertLevel.WARNING,
                message=f"Daily budget alert: ${daily_cost:.2f} / ${daily_limit:.2f} ({daily_utilization*100:.1f}%)",
                current_spend=daily_cost,
                budget_limit=daily_limit,
                utilization_percentage=daily_utilization * 100,
                period="daily",
                timestamp=datetime.now()
            ))
        
        # Check monthly budget
        monthly_utilization = monthly_cost / monthly_limit if monthly_limit > 0 else 0
        if monthly_utilization >= 1.0:
            alerts.append(BudgetAlert(
                level=AlertLevel.BUDGET_EXCEEDED,
                message=f"Monthly budget exceeded: ${monthly_cost:.2f} / ${monthly_limit:.2f}",
                current_spend=monthly_cost,
                budget_limit=monthly_limit,
                utilization_percentage=monthly_utilization * 100,
                period="monthly",
                timestamp=datetime.now()
            ))
        elif monthly_utilization >= alert_threshold:
            alerts.append(BudgetAlert(
                level=AlertLevel.WARNING,
                message=f"Monthly budget alert: ${monthly_cost:.2f} / ${monthly_limit:.2f} ({monthly_utilization*100:.1f}%)",
                current_spend=monthly_cost,
                budget_limit=monthly_limit,
                utilization_percentage=monthly_utilization * 100,
                period="monthly",
                timestamp=datetime.now()
            ))
        
        # Send alerts
        for alert in alerts:
            await self._send_alert(alert)
    
    async def _send_alert(self, alert: BudgetAlert):
        """Send budget alert."""
        # Prevent spam - only send same alert type once per hour
        alert_key = f"{alert.level.value}_{alert.period}"
        now = datetime.now()
        
        if alert_key in self._alerts_sent:
            time_since_last = now - self._alerts_sent[alert_key]
            if time_since_last < timedelta(hours=1):
                return
        
        # Log the alert
        if alert.level in [AlertLevel.BUDGET_EXCEEDED, AlertLevel.CRITICAL]:
            logger.error(alert.message)
        elif alert.level == AlertLevel.WARNING:
            logger.warning(alert.message)
        else:
            logger.info(alert.message)
        
        # Store alert in Redis for dashboard
        if self.redis_manager:
            alert_key_redis = f"llm_alert:{now.strftime('%Y%m%d%H%M%S')}"
            await self.redis_manager.setex(
                alert_key_redis,
                int(timedelta(days=7).total_seconds()),
                json.dumps(alert.to_dict())
            )
        
        self._alerts_sent[alert_key] = now
    
    async def is_budget_available(self, estimated_cost: float = 0.0) -> Tuple[bool, str]:
        """Check if budget is available for a request."""
        if not self.settings.llm.enable_cost_tracking:
            return True, "Cost tracking disabled"
        
        daily_cost = await self.get_daily_cost()
        monthly_cost = await self.get_monthly_cost()
        
        daily_limit = self.settings.llm.daily_budget_limit
        monthly_limit = self.settings.llm.monthly_budget_limit
        
        # Check if adding estimated cost would exceed limits
        if daily_cost + estimated_cost >= daily_limit:
            return False, f"Daily budget limit would be exceeded: ${daily_cost + estimated_cost:.2f} >= ${daily_limit:.2f}"
        
        if monthly_cost + estimated_cost >= monthly_limit:
            return False, f"Monthly budget limit would be exceeded: ${monthly_cost + estimated_cost:.2f} >= ${monthly_limit:.2f}"
        
        return True, "Budget available"
    
    async def get_cost_analytics(
        self,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None
    ) -> CostAnalytics:
        """Get comprehensive cost analytics for a date range."""
        await self.initialize()
        
        if end_date is None:
            end_date = date.today()
        if start_date is None:
            start_date = end_date - timedelta(days=30)
        
        # Get usage records for the period
        usage_records = await self._get_usage_records(start_date, end_date)
        
        if not usage_records:
            return CostAnalytics(
                period_start=datetime.combine(start_date, datetime.min.time()),
                period_end=datetime.combine(end_date, datetime.max.time()),
                total_cost=0.0,
                total_requests=0,
                avg_cost_per_request=0.0,
                cost_by_category={},
                cost_by_model={},
                cost_by_provider={},
                token_usage={},
                cache_hit_rate=0.0,
                cost_savings_from_cache=0.0,
                top_expensive_requests=[],
                usage_trends={}
            )
        
        # Calculate analytics
        total_cost = sum(record.cost for record in usage_records)
        total_requests = len(usage_records)
        avg_cost_per_request = total_cost / total_requests if total_requests > 0 else 0.0
        
        # Group by category
        cost_by_category = {}
        for record in usage_records:
            category = record.category
            cost_by_category[category] = cost_by_category.get(category, 0.0) + record.cost
        
        # Group by model
        cost_by_model = {}
        for record in usage_records:
            model = record.model
            cost_by_model[model] = cost_by_model.get(model, 0.0) + record.cost
        
        # Group by provider
        cost_by_provider = {}
        for record in usage_records:
            provider = record.provider
            cost_by_provider[provider] = cost_by_provider.get(provider, 0.0) + record.cost
        
        # Token usage
        token_usage = {
            'total_tokens': sum(record.total_tokens for record in usage_records),
            'input_tokens': sum(record.input_tokens for record in usage_records),
            'output_tokens': sum(record.output_tokens for record in usage_records),
        }
        
        # Cache analysis
        cached_requests = [r for r in usage_records if r.cached]
        cache_hit_rate = len(cached_requests) / total_requests if total_requests > 0 else 0.0
        
        # Estimate cost savings from cache (approximate)
        cost_savings_from_cache = sum(record.cost for record in cached_requests)
        
        # Top expensive requests
        top_expensive = sorted(usage_records, key=lambda x: x.cost, reverse=True)[:10]
        top_expensive_requests = [
            {
                'timestamp': record.timestamp.isoformat(),
                'model': record.model,
                'cost': record.cost,
                'tokens': record.total_tokens,
                'category': record.category.value
            }
            for record in top_expensive
        ]
        
        # Usage trends (daily aggregation)
        usage_trends = await self._calculate_usage_trends(start_date, end_date)
        
        return CostAnalytics(
            period_start=datetime.combine(start_date, datetime.min.time()),
            period_end=datetime.combine(end_date, datetime.max.time()),
            total_cost=total_cost,
            total_requests=total_requests,
            avg_cost_per_request=avg_cost_per_request,
            cost_by_category=cost_by_category,
            cost_by_model=cost_by_model,
            cost_by_provider=cost_by_provider,
            token_usage=token_usage,
            cache_hit_rate=cache_hit_rate,
            cost_savings_from_cache=cost_savings_from_cache,
            top_expensive_requests=top_expensive_requests,
            usage_trends=usage_trends
        )
    
    async def _get_usage_records(
        self,
        start_date: date,
        end_date: date
    ) -> List[CostUsage]:
        """Get usage records for a date range."""
        if not self.redis_manager:
            return []
        
        records = []
        current_date = start_date
        
        while current_date <= end_date:
            daily_list_key = f"llm_usage_list:{current_date.strftime('%Y-%m-%d')}"
            
            try:
                # Get all records for this day
                raw_records = await self.redis_manager.lrange(daily_list_key, 0, -1)
                
                for raw_record in raw_records:
                    try:
                        record_data = json.loads(raw_record)
                        record = CostUsage.from_dict(record_data)
                        records.append(record)
                    except Exception as e:
                        logger.warning(f"Failed to parse usage record: {e}")
                        
            except Exception as e:
                logger.warning(f"Failed to get usage records for {current_date}: {e}")
            
            current_date += timedelta(days=1)
        
        return records
    
    async def _calculate_usage_trends(
        self,
        start_date: date,
        end_date: date
    ) -> Dict[str, List[float]]:
        """Calculate daily usage trends."""
        trends = {
            'daily_costs': [],
            'daily_requests': [],
            'daily_tokens': []
        }
        
        current_date = start_date
        while current_date <= end_date:
            daily_cost = await self.get_daily_cost(current_date)
            
            # Get daily request count and tokens (would need to implement)
            # For now, use placeholder values
            daily_requests = 0
            daily_tokens = 0
            
            trends['daily_costs'].append(daily_cost)
            trends['daily_requests'].append(daily_requests)
            trends['daily_tokens'].append(daily_tokens)
            
            current_date += timedelta(days=1)
        
        return trends
    
    async def get_cost_optimization_recommendations(self) -> List[Dict[str, Any]]:
        """Get cost optimization recommendations based on usage patterns."""
        analytics = await self.get_cost_analytics()
        recommendations = []
        
        # Check for expensive models
        if analytics.cost_by_model:
            most_expensive_model = max(analytics.cost_by_model.items(), key=lambda x: x[1])
            if most_expensive_model[1] > analytics.total_cost * 0.5:
                recommendations.append({
                    'type': 'model_optimization',
                    'priority': 'high',
                    'message': f"Consider using cheaper models for simple tasks. {most_expensive_model[0]} accounts for {(most_expensive_model[1]/analytics.total_cost)*100:.1f}% of total cost.",
                    'estimated_savings': most_expensive_model[1] * 0.3  # Assume 30% savings
                })
        
        # Check cache hit rate
        if analytics.cache_hit_rate < 0.3:
            recommendations.append({
                'type': 'caching_optimization',
                'priority': 'medium',
                'message': f"Low cache hit rate ({analytics.cache_hit_rate*100:.1f}%). Consider enabling semantic caching or improving cache TTL.",
                'estimated_savings': analytics.total_cost * 0.2
            })
        
        # Check for token optimization
        avg_tokens_per_request = analytics.token_usage.get('total_tokens', 0) / analytics.total_requests if analytics.total_requests > 0 else 0
        if avg_tokens_per_request > 5000:
            recommendations.append({
                'type': 'token_optimization',
                'priority': 'medium',
                'message': f"High average token usage ({avg_tokens_per_request:.0f} tokens/request). Consider shorter prompts or response limits.",
                'estimated_savings': analytics.total_cost * 0.15
            })
        
        return recommendations
    
    def get_budget_status(self) -> Dict[str, Any]:
        """Get current budget status."""
        return {
            'daily_budget_limit': self.settings.llm.daily_budget_limit,
            'monthly_budget_limit': self.settings.llm.monthly_budget_limit,
            'cost_tracking_enabled': self.settings.llm.enable_cost_tracking,
            'alert_threshold': self.settings.llm.cost_alert_threshold
        }


# Global cost manager instance
_cost_manager: Optional[LLMCostManager] = None


async def get_cost_manager() -> LLMCostManager:
    """Get or create the global cost manager instance."""
    global _cost_manager
    
    if _cost_manager is None:
        _cost_manager = LLMCostManager()
        await _cost_manager.initialize()
    
    return _cost_manager