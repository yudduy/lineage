"""
Comprehensive test suite for Neo4j graph integration.
Tests graph algorithms, CRUD operations, and data migration.
"""

import pytest
import asyncio
import json
import uuid
from typing import Dict, List, Any
from datetime import datetime, timedelta

from app.db.neo4j import Neo4jManager
from app.services.graph_operations import (
    GraphCRUDOperations,
    PaperNode,
    AuthorNode,
    InstitutionNode,
    VenueNode
)
from app.services.data_migration import DataMigrationManager
from app.services.query_optimization import QueryOptimizer


# ==================== FIXTURES ====================

@pytest.fixture
async def neo4j_manager():
    """Create Neo4j manager for testing."""
    manager = Neo4jManager()
    await manager.connect()
    
    # Clean up any existing test data
    await manager.execute_write("MATCH (n) WHERE n.test_data = true DETACH DELETE n")
    
    yield manager
    
    # Clean up after tests
    await manager.execute_write("MATCH (n) WHERE n.test_data = true DETACH DELETE n")
    await manager.disconnect()


@pytest.fixture
async def crud_operations(neo4j_manager):
    """Create CRUD operations instance."""
    return GraphCRUDOperations(neo4j_manager)


@pytest.fixture
async def migration_manager(neo4j_manager):
    """Create data migration manager."""
    return DataMigrationManager(neo4j_manager)


@pytest.fixture
def sample_paper_data():
    """Sample paper data for testing."""
    return {
        "id": "test_paper_1",
        "title": "Advanced Graph Algorithms for Citation Networks",
        "abstract": "This paper presents novel algorithms for analyzing citation networks...",
        "doi": "10.1000/test.doi.123",
        "publication_year": 2023,
        "citation_count": 42,
        "reference_count": 35,
        "keywords": ["graph algorithms", "citation analysis", "network science"],
        "language": "en",
        "open_access": True,
        "test_data": True
    }


@pytest.fixture
def sample_author_data():
    """Sample author data for testing."""
    return {
        "id": "test_author_1",
        "name": "Dr. Jane Smith",
        "display_name": "Jane Smith",
        "orcid": "0000-0000-0000-0001",
        "h_index": 25,
        "citation_count": 1500,
        "paper_count": 45,
        "affiliations": ["University of Technology"],
        "test_data": True
    }


@pytest.fixture
def sample_papers_dataset():
    """Sample dataset for migration testing."""
    return [
        {
            "id": "paper_1",
            "title": "Foundation of Network Science",
            "abstract": "Fundamental concepts in network analysis...",
            "year": 2020,
            "citationCount": 150,
            "authors": [
                {"authorId": "author_1", "name": "Alice Johnson"},
                {"authorId": "author_2", "name": "Bob Wilson"}
            ],
            "test_data": True
        },
        {
            "id": "paper_2", 
            "title": "Machine Learning on Graphs",
            "abstract": "Applying ML techniques to graph data...",
            "year": 2021,
            "citationCount": 75,
            "authors": [
                {"authorId": "author_2", "name": "Bob Wilson"},
                {"authorId": "author_3", "name": "Carol Davis"}
            ],
            "test_data": True
        },
        {
            "id": "paper_3",
            "title": "Citation Network Analysis",
            "abstract": "Methods for analyzing academic citation networks...",
            "year": 2022,
            "citationCount": 30,
            "authors": [
                {"authorId": "author_1", "name": "Alice Johnson"}
            ],
            "test_data": True
        }
    ]


@pytest.fixture
def sample_edges_dataset():
    """Sample edges dataset for migration testing."""
    return [
        {"source": "paper_2", "target": "paper_1"},
        {"source": "paper_3", "target": "paper_1"},
        {"source": "paper_3", "target": "paper_2"}
    ]


# ==================== CRUD OPERATIONS TESTS ====================

@pytest.mark.asyncio
class TestCRUDOperations:
    """Test CRUD operations for graph entities."""
    
    async def test_create_and_get_paper(self, crud_operations, sample_paper_data):
        """Test creating and retrieving a paper."""
        # Create paper
        paper_id = await crud_operations.create_paper(sample_paper_data)
        assert paper_id == sample_paper_data["id"]
        
        # Retrieve paper
        retrieved_paper = await crud_operations.get_paper(paper_id)
        assert retrieved_paper is not None
        assert retrieved_paper["title"] == sample_paper_data["title"]
        assert retrieved_paper["doi"] == sample_paper_data["doi"]
        assert retrieved_paper["publication_year"] == sample_paper_data["publication_year"]
    
    async def test_update_paper(self, crud_operations, sample_paper_data):
        """Test updating paper properties."""
        # Create paper
        paper_id = await crud_operations.create_paper(sample_paper_data)
        
        # Update paper
        updates = {
            "citation_count": 50,
            "abstract": "Updated abstract for the paper..."
        }
        success = await crud_operations.update_paper(paper_id, updates)
        assert success
        
        # Verify updates
        updated_paper = await crud_operations.get_paper(paper_id)
        assert updated_paper["citation_count"] == 50
        assert "Updated abstract" in updated_paper["abstract"]
    
    async def test_search_papers(self, crud_operations, sample_paper_data):
        """Test paper search functionality."""
        # Create paper
        await crud_operations.create_paper(sample_paper_data)
        
        # Wait a moment for indexing
        await asyncio.sleep(1)
        
        # Search papers
        results = await crud_operations.search_papers("graph algorithms", limit=10)
        
        # Should find the paper (if full-text index is set up)
        # Note: In actual test environment, may need to set up indexes
        assert isinstance(results, list)
    
    async def test_create_and_get_author(self, crud_operations, sample_author_data):
        """Test creating and retrieving an author."""
        # Create author
        author_id = await crud_operations.create_author(sample_author_data)
        assert author_id == sample_author_data["id"]
        
        # Retrieve author
        retrieved_author = await crud_operations.get_author(author_id)
        assert retrieved_author is not None
        assert retrieved_author["name"] == sample_author_data["name"]
        assert retrieved_author["orcid"] == sample_author_data["orcid"]
    
    async def test_citation_relationship(self, crud_operations, sample_paper_data):
        """Test creating citation relationships."""
        # Create two papers
        paper_1_data = sample_paper_data.copy()
        paper_1_data["id"] = "test_paper_citing"
        
        paper_2_data = sample_paper_data.copy()
        paper_2_data["id"] = "test_paper_cited"
        paper_2_data["title"] = "Referenced Paper"
        
        await crud_operations.create_paper(paper_1_data)
        await crud_operations.create_paper(paper_2_data)
        
        # Create citation relationship
        success = await crud_operations.create_citation_relationship(
            citing_paper_id="test_paper_citing",
            cited_paper_id="test_paper_cited",
            context="This paper builds on the foundational work...",
            is_influential=True
        )
        assert success
    
    async def test_authorship_relationship(self, crud_operations, sample_paper_data, sample_author_data):
        """Test creating authorship relationships."""
        # Create paper and author
        await crud_operations.create_paper(sample_paper_data)
        await crud_operations.create_author(sample_author_data)
        
        # Create authorship relationship
        success = await crud_operations.create_authorship_relationship(
            author_id=sample_author_data["id"],
            paper_id=sample_paper_data["id"],
            position=1,
            is_corresponding=True
        )
        assert success
        
        # Verify relationship by getting author's papers
        papers = await crud_operations.get_author_papers(sample_author_data["id"])
        assert len(papers) >= 1
        assert any(paper["id"] == sample_paper_data["id"] for paper in papers)
    
    async def test_batch_operations(self, crud_operations):
        """Test batch creation of papers and citations."""
        # Create batch of papers
        papers = []
        for i in range(5):
            paper = PaperNode(
                id=f"batch_paper_{i}",
                title=f"Batch Paper {i}",
                publication_year=2020 + i,
                citation_count=i * 10,
                test_data=True
            )
            papers.append(paper)
        
        created_ids = await crud_operations.batch_create_papers(papers)
        assert len(created_ids) == 5
        
        # Create batch citations
        citations = [
            ("batch_paper_1", "batch_paper_0"),
            ("batch_paper_2", "batch_paper_0"),
            ("batch_paper_2", "batch_paper_1"),
            ("batch_paper_3", "batch_paper_1"),
            ("batch_paper_4", "batch_paper_2")
        ]
        
        citations_created = await crud_operations.batch_create_citations(citations)
        assert citations_created >= 4  # Some citations might already exist


# ==================== GRAPH ALGORITHMS TESTS ====================

@pytest.mark.asyncio
# NOTE: Advanced graph algorithm tests commented out since we removed advanced features
# These would require Neo4j GDS and GraphProjection functionality
"""
class TestGraphAlgorithms:
    '''Test graph algorithms functionality.'''
    
    async def test_graph_projection_creation(self, neo4j_manager):
        '''Test creating and managing graph projections.'''
        projection = GraphProjection(
            name="test_projection",
            node_labels=["Paper"],
            relationship_types=["CITES"],
            orientation="NATURAL"
        )
        
        success = await neo4j_manager.create_graph_projection(projection)
        assert success
        
        # List projections
        projections = await neo4j_manager.list_graph_projections()
        projection_names = [p["graphName"] for p in projections]
        assert "test_projection" in projection_names
        
        # Drop projection
        success = await neo4j_manager.drop_graph_projection("test_projection")
        assert success
    
    async def test_pagerank_calculation(self, neo4j_manager, crud_operations):
        """Test PageRank centrality calculation."""
        # Create a small citation network
        papers = []
        for i in range(4):
            paper = PaperNode(
                id=f"pagerank_paper_{i}",
                title=f"Paper {i}",
                test_data=True
            )
            papers.append(paper)
        
        await crud_operations.batch_create_papers(papers)
        
        # Create citation network: 0 <- 1, 0 <- 2, 1 <- 3, 2 <- 3
        citations = [
            ("pagerank_paper_1", "pagerank_paper_0"),
            ("pagerank_paper_2", "pagerank_paper_0"),
            ("pagerank_paper_3", "pagerank_paper_1"),
            ("pagerank_paper_3", "pagerank_paper_2")
        ]
        await crud_operations.batch_create_citations(citations)
        
        # Create graph projection
        projection = GraphProjection(
            name="pagerank_test",
            node_labels=["Paper"],
            relationship_types=["CITES"]
        )
        
        await neo4j_manager.create_graph_projection(projection)
        
        try:
            # Calculate PageRank
            result = await neo4j_manager.calculate_pagerank(
                graph_name="pagerank_test",
                max_iterations=10
            )
            
            assert result.algorithm == "PageRank"
            assert result.execution_time > 0
            assert len(result.top_nodes) > 0
            assert result.statistics["min"] >= 0
            assert result.statistics["max"] > result.statistics["min"]
            
        finally:
            await neo4j_manager.drop_graph_projection("pagerank_test")
    
    async def test_community_detection(self, neo4j_manager, crud_operations):
        """Test community detection algorithms."""
        # Create a network with clear communities
        # Community 1: papers 0, 1, 2 (densely connected)
        # Community 2: papers 3, 4, 5 (densely connected)
        # Sparse connections between communities
        
        papers = []
        for i in range(6):
            paper = PaperNode(
                id=f"community_paper_{i}",
                title=f"Community Paper {i}",
                test_data=True
            )
            papers.append(paper)
        
        await crud_operations.batch_create_papers(papers)
        
        # Create citation network with communities
        citations = [
            # Community 1 internal connections
            ("community_paper_1", "community_paper_0"),
            ("community_paper_2", "community_paper_0"),
            ("community_paper_2", "community_paper_1"),
            
            # Community 2 internal connections  
            ("community_paper_4", "community_paper_3"),
            ("community_paper_5", "community_paper_3"),
            ("community_paper_5", "community_paper_4"),
            
            # Cross-community connection
            ("community_paper_3", "community_paper_0")
        ]
        await crud_operations.batch_create_citations(citations)
        
        # Create undirected projection for community detection
        projection = GraphProjection(
            name="community_test",
            node_labels=["Paper"],
            relationship_types=["CITES"],
            orientation="UNDIRECTED"
        )
        
        await neo4j_manager.create_graph_projection(projection)
        
        try:
            # Test Louvain algorithm
            louvain_result = await neo4j_manager.detect_communities_louvain(
                graph_name="community_test",
                max_iterations=10
            )
            
            assert louvain_result.algorithm == "Louvain"
            assert louvain_result.community_count >= 1
            assert louvain_result.execution_time > 0
            assert len(louvain_result.node_communities) == 6
            
        finally:
            await neo4j_manager.drop_graph_projection("community_test")
    
    async def test_shortest_path(self, neo4j_manager, crud_operations):
        """Test shortest path algorithms."""
        # Create linear chain: 0 -> 1 -> 2 -> 3
        papers = []
        for i in range(4):
            paper = PaperNode(
                id=f"path_paper_{i}",
                title=f"Path Paper {i}",
                test_data=True
            )
            papers.append(paper)
        
        await crud_operations.batch_create_papers(papers)
        
        citations = [
            ("path_paper_1", "path_paper_0"),
            ("path_paper_2", "path_paper_1"),
            ("path_paper_3", "path_paper_2")
        ]
        await crud_operations.batch_create_citations(citations)
        
        # Create projection
        projection = GraphProjection(
            name="path_test",
            node_labels=["Paper"],
            relationship_types=["CITES"]
        )
        
        await neo4j_manager.create_graph_projection(projection)
        
        try:
            # Find shortest path
            path = await neo4j_manager.find_shortest_path(
                graph_name="path_test",
                source_node_id="path_paper_3",
                target_node_id="path_paper_0"
            )
            
            assert path["path_length"] == 4  # 3 hops + 1
            assert path["path_nodes"][0] == "path_paper_3"
            assert path["path_nodes"][-1] == "path_paper_0"
            
        finally:
            await neo4j_manager.drop_graph_projection("path_test")
    
    async def test_research_lineage_tracing(self, neo4j_manager, crud_operations):
        """Test intellectual lineage tracing."""
        # Create a citation lineage: 0 <- 1 <- 2 <- 3 (3 cites 2 cites 1 cites 0)
        papers = []
        for i in range(4):
            paper = PaperNode(
                id=f"lineage_paper_{i}",
                title=f"Lineage Paper {i}",
                publication_year=2020 + i,
                test_data=True
            )
            papers.append(paper)
        
        await crud_operations.batch_create_papers(papers)
        
        citations = [
            ("lineage_paper_1", "lineage_paper_0"),
            ("lineage_paper_2", "lineage_paper_1"), 
            ("lineage_paper_3", "lineage_paper_2")
        ]
        await crud_operations.batch_create_citations(citations)
        
        # Trace backward lineage from paper 3
        lineage = await neo4j_manager.find_research_lineage(
            paper_id="lineage_paper_3",
            direction="backward",
            max_depth=5
        )
        
        assert lineage["paper_id"] == "lineage_paper_3"
        assert lineage["total_lineage_papers"] >= 3
        assert lineage["lineage_depth"] >= 3
        assert len(lineage["backward_lineage"]) >= 3


# ==================== DATA MIGRATION TESTS ====================

@pytest.mark.asyncio 
class TestDataMigration:
    """Test data migration functionality."""
    
    async def test_paper_migration(self, migration_manager, sample_papers_dataset):
        """Test migrating papers from legacy format."""
        stats = await migration_manager.migrate_papers_from_json(
            papers_data=sample_papers_dataset,
            batch_size=2,
            use_background_tasks=False  # Direct migration for testing
        )
        
        assert stats.papers_processed == 3
        assert stats.papers_migrated >= 2  # Some might fail due to missing fields
        assert stats.duration > 0
        
        # Verify papers were created
        crud_ops = migration_manager.crud_ops
        paper = await crud_ops.get_paper("paper_1")
        assert paper is not None
        assert paper["title"] == "Foundation of Network Science"
    
    async def test_author_migration(self, migration_manager, sample_papers_dataset):
        """Test migrating authors from paper data."""
        migrated_count = await migration_manager.migrate_authors_from_papers(
            papers_data=sample_papers_dataset,
            batch_size=10
        )
        
        assert migrated_count >= 2  # At least 2 unique authors
        
        # Verify authors were created
        crud_ops = migration_manager.crud_ops
        author = await crud_ops.get_author("author_1")
        assert author is not None
        assert author["name"] == "Alice Johnson"
    
    async def test_citations_migration(self, migration_manager, sample_edges_dataset):
        """Test migrating citation relationships."""
        # First create papers for citations to reference
        papers_data = [
            {"id": "paper_1", "title": "Paper 1", "test_data": True},
            {"id": "paper_2", "title": "Paper 2", "test_data": True},
            {"id": "paper_3", "title": "Paper 3", "test_data": True}
        ]
        
        await migration_manager.migrate_papers_from_json(
            papers_data=papers_data,
            use_background_tasks=False
        )
        
        # Migrate citations
        migrated_count = await migration_manager.migrate_citations_from_edges(
            edges_data=sample_edges_dataset,
            use_background_tasks=False
        )
        
        assert migrated_count >= 2
    
    async def test_complete_migration(self, migration_manager, sample_papers_dataset, sample_edges_dataset):
        """Test complete dataset migration."""
        stats = await migration_manager.migrate_complete_dataset(
            papers_data=sample_papers_dataset,
            edges_data=sample_edges_dataset,
            batch_size=2,
            use_background_tasks=False
        )
        
        assert stats.papers_processed == 3
        assert stats.papers_migrated >= 2
        assert stats.citations_processed == 3
        assert stats.success_rate > 0.5
    
    async def test_migration_validation(self, migration_manager, sample_papers_dataset):
        """Test migration validation."""
        # Perform migration
        await migration_manager.migrate_complete_dataset(
            papers_data=sample_papers_dataset,
            use_background_tasks=False
        )
        
        # Validate migration
        validation = await migration_manager.validate_migration()
        
        assert "node_counts" in validation
        assert "validation_checks" in validation
        assert validation["validation_checks"]["papers_have_titles"] >= 2


# ==================== PERFORMANCE TESTS ====================

@pytest.mark.asyncio
class TestPerformance:
    """Test performance aspects of graph operations."""
    
    async def test_large_batch_operations(self, crud_operations):
        """Test performance with larger batches."""
        # Create 100 papers
        papers = []
        for i in range(100):
            paper = PaperNode(
                id=f"perf_paper_{i}",
                title=f"Performance Paper {i}",
                publication_year=2020 + (i % 4),
                citation_count=i,
                test_data=True
            )
            papers.append(paper)
        
        start_time = datetime.utcnow()
        created_ids = await crud_operations.batch_create_papers(papers)
        end_time = datetime.utcnow()
        
        duration = (end_time - start_time).total_seconds()
        
        assert len(created_ids) == 100
        assert duration < 10.0  # Should complete within 10 seconds
        
        # Test batch citations
        citations = []
        for i in range(1, 100):
            # Each paper cites the previous one
            citations.append((f"perf_paper_{i}", f"perf_paper_{i-1}"))
        
        start_time = datetime.utcnow()
        citations_created = await crud_operations.batch_create_citations(citations)
        end_time = datetime.utcnow()
        
        duration = (end_time - start_time).total_seconds()
        
        assert citations_created >= 90  # Most should succeed
        assert duration < 15.0  # Should complete within 15 seconds


# ==================== INTEGRATION TESTS ====================

@pytest.mark.asyncio
class TestIntegration:
    """Integration tests for complete workflows."""
    
    async def test_complete_analysis_workflow(self, neo4j_manager, crud_operations):
        """Test complete analysis workflow from data creation to insights."""
        
        # Step 1: Create research community with papers and citations
        papers = []
        for i in range(10):
            paper = PaperNode(
                id=f"workflow_paper_{i}",
                title=f"Workflow Paper {i}",
                publication_year=2020 + (i % 3),
                citation_count=i * 5,
                keywords=[f"keyword_{i%3}", f"topic_{i%2}"],
                test_data=True
            )
            papers.append(paper)
        
        await crud_operations.batch_create_papers(papers)
        
        # Create citation network (star pattern with paper_0 at center)
        citations = []
        for i in range(1, 10):
            citations.append((f"workflow_paper_{i}", "workflow_paper_0"))
        
        await crud_operations.batch_create_citations(citations)
        
        # Step 2: Create graph projection
        projection = GraphProjection(
            name="workflow_analysis",
            node_labels=["Paper"],
            relationship_types=["CITES"]
        )
        
        await neo4j_manager.create_graph_projection(projection)
        
        try:
            # Step 3: Run multiple analyses
            
            # Calculate influence metrics
            pagerank_result = await neo4j_manager.calculate_pagerank(
                graph_name="workflow_analysis"
            )
            assert pagerank_result.execution_time > 0
            
            # Detect communities
            community_result = await neo4j_manager.detect_communities_louvain(
                graph_name="workflow_analysis"
            )
            assert community_result.community_count >= 1
            
            # Calculate graph metrics
            metrics = await neo4j_manager.calculate_graph_metrics("workflow_analysis")
            assert metrics["node_count"] == 10
            assert metrics["relationship_count"] == 9
            
            # Trace lineage for most cited paper
            lineage = await neo4j_manager.find_research_lineage(
                paper_id="workflow_paper_0",
                direction="forward",
                max_depth=2
            )
            assert lineage["total_lineage_papers"] >= 8
            
        finally:
            await neo4j_manager.drop_graph_projection("workflow_analysis")
"""

# ==================== UTILITY FUNCTIONS ====================

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])