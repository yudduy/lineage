"""
Initialize Advanced Analytics System.

This script initializes the advanced analytics system components and verifies
that all services are properly configured and operational.
"""

import asyncio
import logging
from typing import Optional

from ..services.advanced_analytics import get_analytics_service
from ..services.research_intelligence import get_intelligence_engine  
from ..services.performance_optimizer import get_performance_optimizer
from ..core.config import get_settings
from ..utils.logger import get_logger

logger = get_logger(__name__)


async def initialize_analytics_system():
    """Initialize all components of the advanced analytics system."""
    
    logger.info("Starting Advanced Analytics System initialization...")
    
    try:
        # 1. Initialize Performance Optimizer (foundational layer)
        logger.info("Initializing Performance Optimizer...")
        performance_optimizer = await get_performance_optimizer()
        logger.info("✓ Performance Optimizer initialized")
        
        # 2. Initialize Advanced Analytics Service
        logger.info("Initializing Advanced Analytics Service...")
        analytics_service = await get_analytics_service()
        logger.info("✓ Advanced Analytics Service initialized")
        
        # 3. Initialize Research Intelligence Engine
        logger.info("Initializing Research Intelligence Engine...")
        intelligence_engine = await get_intelligence_engine()
        logger.info("✓ Research Intelligence Engine initialized")
        
        # 4. Verify system health
        logger.info("Verifying system health...")
        await verify_system_health(
            performance_optimizer,
            analytics_service, 
            intelligence_engine
        )
        logger.info("✓ System health verification passed")
        
        # 5. Warm up caches with sample data
        logger.info("Warming up caches...")
        await warm_up_caches(analytics_service)
        logger.info("✓ Cache warming completed")
        
        logger.info("🎉 Advanced Analytics System initialization completed successfully!")
        
        return {
            "status": "success",
            "components": {
                "performance_optimizer": True,
                "analytics_service": True,
                "intelligence_engine": True
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Advanced Analytics System initialization failed: {e}")
        raise


async def verify_system_health(
    performance_optimizer,
    analytics_service,
    intelligence_engine
):
    """Verify that all system components are healthy."""
    
    # Test performance optimizer
    try:
        metrics = await performance_optimizer.get_performance_report()
        assert metrics is not None, "Performance optimizer not responding"
        logger.info("  ✓ Performance optimizer health check passed")
    except Exception as e:
        logger.error(f"  ❌ Performance optimizer health check failed: {e}")
        raise
    
    # Test analytics service  
    try:
        perf_metrics = await analytics_service.get_performance_metrics()
        assert perf_metrics is not None, "Analytics service not responding"
        logger.info("  ✓ Analytics service health check passed")
    except Exception as e:
        logger.error(f"  ❌ Analytics service health check failed: {e}")
        raise
    
    # Test intelligence engine
    try:
        # Simple test - try to detect trends with minimal parameters
        trends = await intelligence_engine.detect_research_trends(
            domain=None,
            time_window=None,
            min_confidence=0.9  # High confidence to limit results
        )
        # Should return empty list or valid trends, not error
        logger.info(f"  ✓ Intelligence engine health check passed (found {len(trends)} trends)")
    except Exception as e:
        logger.error(f"  ❌ Intelligence engine health check failed: {e}")
        raise


async def warm_up_caches(analytics_service):
    """Warm up caches with sample data for better initial performance."""
    
    try:
        # Sample paper IDs for cache warming (these would be real IDs in production)
        sample_paper_ids = [
            "sample_paper_1",
            "sample_paper_2", 
            "sample_paper_3"
        ]
        
        # Queue cache warming for sample papers
        await analytics_service._queue_cache_warming(sample_paper_ids)
        logger.info(f"  ✓ Queued cache warming for {len(sample_paper_ids)} sample papers")
        
        # Wait a bit for cache warming to start
        await asyncio.sleep(2)
        
    except Exception as e:
        logger.warning(f"  ⚠️ Cache warming had issues (non-critical): {e}")


async def cleanup_analytics_system():
    """Clean up analytics system resources."""
    
    logger.info("Shutting down Advanced Analytics System...")
    
    try:
        # Get service instances if they exist
        performance_optimizer = await get_performance_optimizer()
        analytics_service = await get_analytics_service()
        intelligence_engine = await get_intelligence_engine()
        
        # Clean up in reverse order
        if intelligence_engine:
            await intelligence_engine.close()
            logger.info("✓ Research Intelligence Engine closed")
            
        if analytics_service:
            await analytics_service.close()
            logger.info("✓ Advanced Analytics Service closed")
            
        if performance_optimizer:
            await performance_optimizer.close()
            logger.info("✓ Performance Optimizer closed")
            
        logger.info("🔒 Advanced Analytics System shutdown completed")
        
    except Exception as e:
        logger.error(f"❌ Error during system shutdown: {e}")
        raise


def main():
    """Main function for direct script execution."""
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    async def run_initialization():
        try:
            result = await initialize_analytics_system()
            print(f"Initialization result: {result}")
            
            # Keep running for a bit to test the system
            logger.info("System running... (press Ctrl+C to stop)")
            await asyncio.sleep(30)
            
        except KeyboardInterrupt:
            logger.info("Received interrupt signal")
        finally:
            await cleanup_analytics_system()
    
    # Run the initialization
    asyncio.run(run_initialization())


if __name__ == "__main__":
    main()