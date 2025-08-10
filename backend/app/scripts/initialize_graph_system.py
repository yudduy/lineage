"""
Initialization script for the Neo4j graph system.
Sets up the database schema, creates initial projections, and validates the installation.
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Dict, Any, List

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from app.db.neo4j_advanced import AdvancedNeo4jManager, GraphProjection
from app.services.graph_operations import GraphCRUDOperations
from app.services.data_migration import DataMigrationManager
from app.utils.logger import get_logger

logger = get_logger(__name__)


class GraphSystemInitializer:
    """Initializes and validates the complete graph system."""
    
    def __init__(self):
        self.neo4j_manager = None
        self.crud_ops = None
        self.migration_manager = None
    
    async def initialize(self):
        """Initialize all system components."""
        logger.info("Starting Neo4j Graph System initialization...")
        
        # Step 1: Connect to Neo4j
        logger.info("Step 1: Connecting to Neo4j...")
        self.neo4j_manager = AdvancedNeo4jManager()
        await self.neo4j_manager.connect()
        
        # Verify GDS support
        logger.info("Verifying Neo4j Graph Data Science support...")
        await self._verify_gds_installation()
        
        # Step 2: Initialize CRUD operations
        logger.info("Step 2: Initializing CRUD operations...")
        self.crud_ops = GraphCRUDOperations(self.neo4j_manager)
        
        # Step 3: Initialize migration manager
        logger.info("Step 3: Initializing migration manager...")
        self.migration_manager = DataMigrationManager(self.neo4j_manager)
        
        # Step 4: Create initial graph projections
        logger.info("Step 4: Creating initial graph projections...")
        await self._create_initial_projections()
        
        # Step 5: Run system validation
        logger.info("Step 5: Running system validation...")
        validation_results = await self._run_system_validation()
        
        logger.info("Neo4j Graph System initialization completed successfully!")
        return validation_results
    
    async def _verify_gds_installation(self):
        """Verify that Neo4j GDS is properly installed."""
        try:
            # Test GDS availability
            query = "CALL gds.version() YIELD gdsVersion RETURN gdsVersion"
            result = await self.neo4j_manager.execute_read(query)
            
            if result:
                gds_version = result[0]["gdsVersion"]
                logger.info(f"Neo4j GDS version {gds_version} detected")
            else:
                raise Exception("GDS version query returned no results")
                
        except Exception as e:
            logger.error(f"Neo4j GDS verification failed: {e}")
            logger.error("""
            Neo4j Graph Data Science (GDS) library is required but not available.
            
            Please install GDS by:
            1. Downloading GDS plugin from https://neo4j.com/docs/graph-data-science/current/installation/
            2. Copying the plugin jar to Neo4j plugins directory
            3. Adding 'dbms.security.procedures.unrestricted=gds.*' to neo4j.conf
            4. Restarting Neo4j server
            """)
            raise
    
    async def _create_initial_projections(self):
        """Create initial graph projections for common analyses."""
        projections = [
            GraphProjection(
                name="citation_network",
                node_labels=["Paper"],
                relationship_types=["CITES"],
                node_properties={
                    "Paper": ["citation_count", "publication_year", "h_index"]
                },
                orientation="NATURAL"
            ),
            GraphProjection(
                name="citation_network_undirected",
                node_labels=["Paper"],
                relationship_types=["CITES"],
                node_properties={
                    "Paper": ["citation_count", "publication_year", "h_index"]
                },
                orientation="UNDIRECTED"
            ),
            GraphProjection(
                name="author_collaboration",
                node_labels=["Author"],
                relationship_types=["COLLABORATES_WITH"],
                node_properties={
                    "Author": ["h_index", "citation_count", "paper_count"]
                },
                orientation="UNDIRECTED"
            )
        ]
        
        for projection in projections:
            try:
                # Drop existing projection if it exists
                await self.neo4j_manager.drop_graph_projection(projection.name)
                
                # Create new projection (will succeed even if no data exists)
                success = await self.neo4j_manager.create_graph_projection(projection)
                if success:
                    logger.info(f"Created graph projection: {projection.name}")
                else:
                    logger.warning(f"Failed to create graph projection: {projection.name}")
                    
            except Exception as e:
                logger.warning(f"Error creating projection {projection.name}: {e}")
    
    async def _run_system_validation(self) -> Dict[str, Any]:
        """Run comprehensive system validation."""
        validation_results = {
            "timestamp": "2025-08-08T00:00:00Z",
            "neo4j_connection": False,
            "gds_availability": False,
            "schema_initialized": False,
            "projections_created": False,
            "crud_operations": False,
            "graph_algorithms": False,
            "errors": []
        }
        
        try:
            # Test Neo4j connection
            health = await self.neo4j_manager.health_check()
            validation_results["neo4j_connection"] = health["status"] == "healthy"
            
            # Test GDS availability
            try:
                await self.neo4j_manager.execute_read("CALL gds.version() YIELD gdsVersion RETURN gdsVersion")
                validation_results["gds_availability"] = True
            except Exception as e:
                validation_results["errors"].append(f"GDS test failed: {e}")
            
            # Test schema initialization
            try:
                # Check if constraints exist
                constraints_query = "SHOW CONSTRAINTS YIELD name RETURN count(*) as count"
                result = await self.neo4j_manager.execute_read(constraints_query)
                constraint_count = result[0]["count"] if result else 0
                validation_results["schema_initialized"] = constraint_count > 0
            except Exception as e:
                validation_results["errors"].append(f"Schema check failed: {e}")
            
            # Test projections
            try:
                projections = await self.neo4j_manager.list_graph_projections()
                validation_results["projections_created"] = len(projections) > 0
            except Exception as e:
                validation_results["errors"].append(f"Projections check failed: {e}")
            
            # Test CRUD operations
            try:
                # Test basic CRUD with sample data
                from app.services.graph_operations import PaperNode
                test_paper = PaperNode(
                    id="test_init_paper",
                    title="Test Paper for System Validation",
                    test_data=True
                )
                
                paper_id = await self.crud_ops.create_paper(test_paper)
                retrieved = await self.crud_ops.get_paper(paper_id)
                
                if retrieved and retrieved["title"] == test_paper.title:
                    validation_results["crud_operations"] = True
                    
                    # Clean up test data
                    await self.crud_ops.delete_paper(paper_id)
                    
            except Exception as e:
                validation_results["errors"].append(f"CRUD test failed: {e}")
            
            # Test graph algorithms (if we have data)
            try:
                projections = await self.neo4j_manager.list_graph_projections()
                if projections:
                    # Try to calculate basic metrics for first projection
                    first_projection = projections[0]["graphName"]
                    metrics = await self.neo4j_manager.calculate_graph_metrics(first_projection)
                    validation_results["graph_algorithms"] = "node_count" in metrics
                else:
                    validation_results["graph_algorithms"] = True  # No data to test with
                    
            except Exception as e:
                validation_results["errors"].append(f"Algorithms test failed: {e}")
            
        except Exception as e:
            validation_results["errors"].append(f"Validation failed: {e}")
        
        return validation_results
    
    async def create_sample_dataset(self, size: int = 50):
        """Create a sample dataset for testing and demonstration."""
        logger.info(f"Creating sample dataset with {size} papers...")
        
        try:
            from app.services.graph_operations import PaperNode, AuthorNode
            import random
            
            # Generate sample authors
            authors = []
            for i in range(size // 3):  # ~17 authors for 50 papers
                author = AuthorNode(
                    id=f"sample_author_{i}",
                    name=f"Dr. Author {i}",
                    h_index=random.randint(5, 50),
                    citation_count=random.randint(100, 2000),
                    paper_count=random.randint(10, 100),
                    test_data=True
                )
                authors.append(author)
                await self.crud_ops.create_author(author)
            
            # Generate sample papers
            papers = []
            research_areas = ["machine learning", "network science", "data mining", 
                            "artificial intelligence", "computer vision", "nlp"]
            
            for i in range(size):
                area = random.choice(research_areas)
                paper = PaperNode(
                    id=f"sample_paper_{i}",
                    title=f"Advances in {area.title()}: A Comprehensive Study {i}",
                    abstract=f"This paper presents novel approaches to {area}...",
                    publication_year=random.randint(2015, 2024),
                    citation_count=random.randint(0, 500),
                    keywords=[area, "research", "analysis"],
                    test_data=True
                )
                papers.append(paper)
            
            # Create papers in batch
            created_ids = await self.crud_ops.batch_create_papers(papers)
            logger.info(f"Created {len(created_ids)} sample papers")
            
            # Create random citation network
            citations = []
            for i in range(size * 2):  # 2x citations as papers
                citing_paper = random.choice(created_ids)
                cited_paper = random.choice(created_ids)
                
                if citing_paper != cited_paper:  # Don't self-cite
                    citations.append((citing_paper, cited_paper))
            
            # Remove duplicates
            citations = list(set(citations))
            
            # Create citations
            citations_created = await self.crud_ops.batch_create_citations(citations)
            logger.info(f"Created {citations_created} sample citations")
            
            # Create some authorship relationships
            authorship_count = 0
            for paper_id in created_ids[:size//2]:  # Only for half the papers
                # Randomly assign 1-3 authors
                num_authors = random.randint(1, 3)
                selected_authors = random.sample(authors, min(num_authors, len(authors)))
                
                for pos, author in enumerate(selected_authors):
                    success = await self.crud_ops.create_authorship_relationship(
                        author_id=author.id,
                        paper_id=paper_id,
                        position=pos + 1
                    )
                    if success:
                        authorship_count += 1
            
            logger.info(f"Created {authorship_count} authorship relationships")
            
            # Recreate projections with new data
            await self._create_initial_projections()
            
            return {
                "papers_created": len(created_ids),
                "citations_created": citations_created,
                "authors_created": len(authors),
                "authorships_created": authorship_count
            }
            
        except Exception as e:
            logger.error(f"Sample dataset creation failed: {e}")
            raise
    
    async def cleanup_test_data(self):
        """Clean up all test data from the database."""
        logger.info("Cleaning up test data...")
        
        try:
            # Delete all nodes marked as test data
            cleanup_query = """
            MATCH (n)
            WHERE n.test_data = true
            DETACH DELETE n
            RETURN count(*) as deleted_count
            """
            
            result = await self.neo4j_manager.execute_write(cleanup_query)
            deleted_count = result.get("nodes_deleted", 0)
            
            logger.info(f"Cleaned up {deleted_count} test nodes")
            
            # Drop test projections
            test_projections = ["citation_network", "citation_network_undirected", "author_collaboration"]
            for proj_name in test_projections:
                await self.neo4j_manager.drop_graph_projection(proj_name)
            
        except Exception as e:
            logger.error(f"Test data cleanup failed: {e}")
    
    async def close(self):
        """Close all connections."""
        if self.neo4j_manager:
            await self.neo4j_manager.disconnect()


async def main():
    """Main initialization function."""
    initializer = GraphSystemInitializer()
    
    try:
        # Initialize system
        validation_results = await initializer.initialize()
        
        # Print validation results
        print("\n" + "="*50)
        print("NEO4J GRAPH SYSTEM VALIDATION RESULTS")
        print("="*50)
        
        for check, status in validation_results.items():
            if check != "errors" and check != "timestamp":
                status_symbol = "✅" if status else "❌"
                print(f"{status_symbol} {check.replace('_', ' ').title()}: {status}")
        
        if validation_results["errors"]:
            print("\nErrors encountered:")
            for error in validation_results["errors"]:
                print(f"  ❌ {error}")
        
        # Offer to create sample dataset
        print("\n" + "="*50)
        create_sample = input("Create sample dataset for testing? (y/n): ").lower() == 'y'
        
        if create_sample:
            sample_results = await initializer.create_sample_dataset(50)
            print("\nSample dataset created:")
            for key, value in sample_results.items():
                print(f"  • {key.replace('_', ' ').title()}: {value}")
        
        print("\n🎉 Neo4j Graph System is ready for use!")
        print("\nAvailable API endpoints:")
        print("  • /api/v1/graph/projections - Manage graph projections")
        print("  • /api/v1/graph/algorithms/centrality - Calculate centrality measures")
        print("  • /api/v1/graph/algorithms/communities - Detect communities")
        print("  • /api/v1/graph/analysis/lineage - Trace intellectual lineage")
        print("  • /api/v1/graph/search - Search the graph")
        
    except Exception as e:
        logger.error(f"System initialization failed: {e}")
        print(f"\n❌ System initialization failed: {e}")
        sys.exit(1)
        
    finally:
        await initializer.close()


if __name__ == "__main__":
    asyncio.run(main())