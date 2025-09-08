"""
Simple Neo4j graph database initialization script.
Creates basic indexes and constraints needed for the citation network explorer.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from app.db.neo4j import get_neo4j_manager
from app.utils.logger import get_logger

logger = get_logger(__name__)


async def initialize_graph_database():
    """Initialize the Neo4j database with basic schema."""
    logger.info("Initializing Neo4j graph database...")
    
    neo4j_manager = get_neo4j_manager()
    
    try:
        # Test connection
        logger.info("Testing Neo4j connection...")
        await neo4j_manager.verify_connectivity()
        logger.info("✓ Neo4j connection successful")
        
        # Create basic indexes and constraints
        logger.info("Creating indexes and constraints...")
        
        # Paper constraints and indexes
        queries = [
            # Paper node constraints
            "CREATE CONSTRAINT paper_doi IF NOT EXISTS FOR (p:Paper) REQUIRE p.doi IS UNIQUE",
            "CREATE CONSTRAINT paper_id IF NOT EXISTS FOR (p:Paper) REQUIRE p.id IS UNIQUE",
            
            # Author node constraints
            "CREATE CONSTRAINT author_id IF NOT EXISTS FOR (a:Author) REQUIRE a.id IS UNIQUE",
            
            # Indexes for better query performance
            "CREATE INDEX paper_title IF NOT EXISTS FOR (p:Paper) ON (p.title)",
            "CREATE INDEX paper_year IF NOT EXISTS FOR (p:Paper) ON (p.publication_year)",
            "CREATE INDEX author_name IF NOT EXISTS FOR (a:Author) ON (a.name)",
        ]
        
        for query in queries:
            try:
                await neo4j_manager.execute_write(query)
                logger.info(f"✓ Executed: {query.split()[1:4]}")
            except Exception as e:
                if "already exists" in str(e).lower() or "equivalent" in str(e).lower():
                    logger.info(f"✓ Already exists: {query.split()[1:4]}")
                else:
                    logger.error(f"✗ Failed: {query} - {e}")
                    raise
        
        # Verify setup
        logger.info("Verifying database setup...")
        result = await neo4j_manager.execute_read("CALL db.schema.visualization()")
        logger.info("✓ Database schema created successfully")
        
        logger.info("🎉 Neo4j graph database initialized successfully!")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize database: {e}")
        sys.exit(1)
    
    finally:
        await neo4j_manager.close()


if __name__ == "__main__":
    asyncio.run(initialize_graph_database())
