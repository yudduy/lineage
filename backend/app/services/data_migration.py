"""
Data migration utilities for converting existing data structures to Neo4j graph format.
Handles migration from Papers[] and Edges[] to comprehensive graph database.
"""

import asyncio
import json
import uuid
from typing import Any, Dict, List, Optional, Tuple, Set
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path

from ..db.neo4j_advanced import AdvancedNeo4jManager, GraphProjection
from ..services.graph_operations import (
    GraphCRUDOperations,
    PaperNode,
    AuthorNode,
    InstitutionNode,
    VenueNode
)
from ..services.graph_tasks import (
    migrate_papers_batch_task,
    migrate_citations_batch_task
)
from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class MigrationStats:
    """Statistics for data migration process."""
    papers_processed: int = 0
    papers_migrated: int = 0
    papers_failed: int = 0
    authors_processed: int = 0
    authors_migrated: int = 0
    institutions_processed: int = 0
    institutions_migrated: int = 0
    venues_processed: int = 0
    venues_migrated: int = 0
    citations_processed: int = 0
    citations_migrated: int = 0
    authorships_processed: int = 0
    authorships_migrated: int = 0
    start_time: datetime = None
    end_time: datetime = None
    
    @property
    def duration(self) -> float:
        """Get migration duration in seconds."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0
    
    @property
    def success_rate(self) -> float:
        """Calculate overall success rate."""
        total_processed = (
            self.papers_processed + self.authors_processed + 
            self.institutions_processed + self.venues_processed + 
            self.citations_processed + self.authorships_processed
        )
        total_migrated = (
            self.papers_migrated + self.authors_migrated + 
            self.institutions_migrated + self.venues_migrated + 
            self.citations_migrated + self.authorships_migrated
        )
        
        return total_migrated / total_processed if total_processed > 0 else 0.0


class DataMigrationManager:
    """Manages migration from existing data structures to Neo4j."""
    
    def __init__(self, neo4j_manager: AdvancedNeo4jManager):
        self.neo4j = neo4j_manager
        self.crud_ops = GraphCRUDOperations(neo4j_manager)
        self.stats = MigrationStats()
        
    # ==================== LEGACY DATA PARSERS ====================
    
    def parse_legacy_paper(self, paper_data: Dict) -> PaperNode:
        """Parse legacy paper data into PaperNode format."""
        # Handle various legacy formats
        paper_id = paper_data.get('id') or paper_data.get('paperId') or str(uuid.uuid4())
        
        # Extract basic paper information
        title = paper_data.get('title', '')
        abstract = paper_data.get('abstract')
        doi = paper_data.get('doi')
        
        # Handle publication year from various fields
        pub_year = None
        if paper_data.get('year'):
            pub_year = int(paper_data['year'])
        elif paper_data.get('publication_year'):
            pub_year = int(paper_data['publication_year'])
        elif paper_data.get('publicationDate'):
            try:
                pub_year = int(paper_data['publicationDate'][:4])
            except (ValueError, TypeError):
                pass
        
        # Handle citation counts
        citation_count = paper_data.get('citationCount', 0)
        if isinstance(citation_count, str):
            try:
                citation_count = int(citation_count)
            except ValueError:
                citation_count = 0
        
        reference_count = paper_data.get('referenceCount', 0)
        influential_citation_count = paper_data.get('influentialCitationCount', 0)
        
        # Handle keywords/fields
        keywords = []
        if paper_data.get('keywords'):
            keywords = paper_data['keywords']
        elif paper_data.get('fieldsOfStudy'):
            keywords = paper_data['fieldsOfStudy']
        
        # Handle URLs
        pdf_urls = []
        if paper_data.get('openAccessPdf'):
            if isinstance(paper_data['openAccessPdf'], dict):
                if paper_data['openAccessPdf'].get('url'):
                    pdf_urls.append(paper_data['openAccessPdf']['url'])
            elif isinstance(paper_data['openAccessPdf'], str):
                pdf_urls.append(paper_data['openAccessPdf'])
        
        # Handle external IDs
        external_ids = {}
        if paper_data.get('externalIds'):
            external_ids = paper_data['externalIds']
        
        # Handle additional fields
        language = paper_data.get('language')
        open_access = paper_data.get('isOpenAccess', paper_data.get('openAccess'))
        
        return PaperNode(
            id=paper_id,
            doi=doi,
            title=title,
            abstract=abstract,
            publication_year=pub_year,
            citation_count=citation_count,
            reference_count=reference_count,
            influential_citation_count=influential_citation_count,
            keywords=keywords if keywords else None,
            language=language,
            open_access=open_access,
            pdf_urls=pdf_urls if pdf_urls else None,
            external_ids=external_ids if external_ids else None
        )
    
    def parse_legacy_author(self, author_data: Dict) -> AuthorNode:
        """Parse legacy author data into AuthorNode format."""
        author_id = author_data.get('authorId') or author_data.get('id') or str(uuid.uuid4())
        
        name = author_data.get('name', '')
        display_name = author_data.get('displayName', author_data.get('display_name'))
        
        # Handle ORCID
        orcid = author_data.get('orcid')
        
        # Handle metrics
        h_index = author_data.get('hIndex')
        citation_count = author_data.get('citationCount', 0)
        paper_count = author_data.get('paperCount', 0)
        
        # Handle affiliations
        affiliations = []
        if author_data.get('affiliations'):
            for affiliation in author_data['affiliations']:
                if isinstance(affiliation, dict):
                    affiliations.append(affiliation.get('name', str(affiliation)))
                else:
                    affiliations.append(str(affiliation))
        
        # Handle external IDs
        external_ids = author_data.get('externalIds', {})
        
        return AuthorNode(
            id=author_id,
            name=name,
            display_name=display_name,
            orcid=orcid,
            h_index=h_index,
            citation_count=citation_count,
            paper_count=paper_count,
            affiliations=affiliations if affiliations else None,
            external_ids=external_ids if external_ids else None
        )
    
    def parse_legacy_venue(self, venue_data: Dict) -> VenueNode:
        """Parse legacy venue data into VenueNode format."""
        venue_id = venue_data.get('id') or str(uuid.uuid4())
        
        name = venue_data.get('name', '')
        display_name = venue_data.get('displayName', venue_data.get('display_name'))
        
        venue_type = venue_data.get('type')
        issn = venue_data.get('issn')
        publisher = venue_data.get('publisher')
        
        # Handle metrics
        impact_factor = venue_data.get('impactFactor')
        h_index = venue_data.get('hIndex')
        
        # Handle URLs
        homepage_url = venue_data.get('homepageUrl', venue_data.get('homepage_url'))
        
        # Handle external IDs
        external_ids = venue_data.get('externalIds', {})
        
        return VenueNode(
            id=venue_id,
            name=name,
            display_name=display_name,
            issn=issn,
            type=venue_type,
            publisher=publisher,
            impact_factor=impact_factor,
            h_index=h_index,
            homepage_url=homepage_url,
            external_ids=external_ids if external_ids else None
        )
    
    # ==================== MIGRATION METHODS ====================
    
    async def migrate_papers_from_json(
        self, 
        papers_data: List[Dict],
        batch_size: int = 100,
        use_background_tasks: bool = True
    ) -> MigrationStats:
        """Migrate papers from JSON data."""
        logger.info(f"Starting migration of {len(papers_data)} papers")
        self.stats.start_time = datetime.utcnow()
        self.stats.papers_processed = len(papers_data)
        
        try:
            # Parse legacy paper data
            paper_nodes = []
            for paper_data in papers_data:
                try:
                    paper_node = self.parse_legacy_paper(paper_data)
                    paper_nodes.append(paper_node)
                except Exception as e:
                    logger.error(f"Failed to parse paper {paper_data.get('id', 'unknown')}: {e}")
                    self.stats.papers_failed += 1
            
            if use_background_tasks:
                # Use background tasks for large datasets
                tasks = []
                for i in range(0, len(paper_nodes), batch_size):
                    batch = paper_nodes[i:i + batch_size]
                    batch_dicts = [paper.__dict__ for paper in batch]
                    task = migrate_papers_batch_task.delay(batch_dicts, batch_size)
                    tasks.append(task)
                
                # Wait for all tasks to complete (simplified - in production, use proper task monitoring)
                for task in tasks:
                    result = task.get(timeout=300)  # 5 minutes timeout per batch
                    self.stats.papers_migrated += result.get('migrated_count', 0)
                    self.stats.papers_failed += result.get('failed_count', 0)
            
            else:
                # Direct migration for smaller datasets
                migrated_ids = await self.crud_ops.batch_create_papers(paper_nodes)
                self.stats.papers_migrated = len(migrated_ids)
        
        except Exception as e:
            logger.error(f"Papers migration failed: {e}")
            raise
        
        finally:
            self.stats.end_time = datetime.utcnow()
            
        logger.info(f"Papers migration completed: {self.stats.papers_migrated}/{self.stats.papers_processed} successful")
        return self.stats
    
    async def migrate_authors_from_papers(
        self, 
        papers_data: List[Dict],
        batch_size: int = 100
    ) -> int:
        """Extract and migrate authors from paper data."""
        logger.info("Extracting and migrating authors from papers")
        
        authors_map = {}  # Use map to deduplicate
        
        for paper_data in papers_data:
            authors = paper_data.get('authors', [])
            for author_data in authors:
                try:
                    if isinstance(author_data, dict):
                        author_id = author_data.get('authorId') or author_data.get('id')
                        if author_id and author_id not in authors_map:
                            author_node = self.parse_legacy_author(author_data)
                            authors_map[author_id] = author_node
                except Exception as e:
                    logger.error(f"Failed to parse author {author_data}: {e}")
        
        self.stats.authors_processed = len(authors_map)
        
        # Create authors in batches
        author_nodes = list(authors_map.values())
        migrated_count = 0
        
        for i in range(0, len(author_nodes), batch_size):
            batch = author_nodes[i:i + batch_size]
            try:
                for author in batch:
                    await self.crud_ops.create_author(author)
                    migrated_count += 1
            except Exception as e:
                logger.error(f"Failed to migrate author batch {i//batch_size + 1}: {e}")
        
        self.stats.authors_migrated = migrated_count
        logger.info(f"Authors migration completed: {migrated_count}/{len(authors_map)} successful")
        
        return migrated_count
    
    async def migrate_citations_from_edges(
        self,
        edges_data: List[Dict],
        batch_size: int = 500,
        use_background_tasks: bool = True
    ) -> int:
        """Migrate citation relationships from edges data."""
        logger.info(f"Starting migration of {len(edges_data)} citation edges")
        
        # Parse edges into citation pairs
        citations = []
        for edge in edges_data:
            try:
                source = edge.get('source') or edge.get('citing_paper_id')
                target = edge.get('target') or edge.get('cited_paper_id')
                
                if source and target:
                    citations.append((source, target))
                else:
                    logger.warning(f"Invalid edge data: {edge}")
            except Exception as e:
                logger.error(f"Failed to parse edge {edge}: {e}")
        
        self.stats.citations_processed = len(citations)
        
        if use_background_tasks:
            # Use background tasks for large datasets
            tasks = []
            for i in range(0, len(citations), batch_size):
                batch = citations[i:i + batch_size]
                task = migrate_citations_batch_task.delay(batch, batch_size)
                tasks.append(task)
            
            migrated_count = 0
            for task in tasks:
                result = task.get(timeout=300)
                migrated_count += result.get('migrated_count', 0)
            
            self.stats.citations_migrated = migrated_count
        else:
            # Direct migration
            migrated_count = await self.crud_ops.batch_create_citations(citations)
            self.stats.citations_migrated = migrated_count
        
        logger.info(f"Citations migration completed: {migrated_count}/{len(citations)} successful")
        return migrated_count
    
    async def migrate_authorships_from_papers(self, papers_data: List[Dict]) -> int:
        """Create authorship relationships from paper data."""
        logger.info("Creating authorship relationships")
        
        authorship_count = 0
        
        for paper_data in papers_data:
            paper_id = paper_data.get('id') or paper_data.get('paperId')
            authors = paper_data.get('authors', [])
            
            if not paper_id:
                continue
            
            for position, author_data in enumerate(authors):
                try:
                    if isinstance(author_data, dict):
                        author_id = author_data.get('authorId') or author_data.get('id')
                        if author_id:
                            success = await self.crud_ops.create_authorship_relationship(
                                author_id=author_id,
                                paper_id=paper_id,
                                position=position + 1  # 1-based indexing
                            )
                            if success:
                                authorship_count += 1
                except Exception as e:
                    logger.error(f"Failed to create authorship {author_data} -> {paper_id}: {e}")
        
        self.stats.authorships_processed = authorship_count
        self.stats.authorships_migrated = authorship_count
        
        logger.info(f"Authorships migration completed: {authorship_count} relationships created")
        return authorship_count
    
    # ==================== FULL MIGRATION ORCHESTRATION ====================
    
    async def migrate_complete_dataset(
        self,
        papers_data: List[Dict],
        edges_data: List[Dict] = None,
        batch_size: int = 100,
        use_background_tasks: bool = True
    ) -> MigrationStats:
        """Perform complete dataset migration."""
        logger.info("Starting complete dataset migration")
        
        self.stats = MigrationStats()
        self.stats.start_time = datetime.utcnow()
        
        try:
            # Step 1: Migrate papers
            logger.info("Step 1: Migrating papers")
            await self.migrate_papers_from_json(
                papers_data=papers_data,
                batch_size=batch_size,
                use_background_tasks=use_background_tasks
            )
            
            # Step 2: Migrate authors
            logger.info("Step 2: Migrating authors")
            await self.migrate_authors_from_papers(
                papers_data=papers_data,
                batch_size=batch_size
            )
            
            # Step 3: Create authorship relationships
            logger.info("Step 3: Creating authorship relationships")
            await self.migrate_authorships_from_papers(papers_data)
            
            # Step 4: Migrate citations if provided
            if edges_data:
                logger.info("Step 4: Migrating citations")
                await self.migrate_citations_from_edges(
                    edges_data=edges_data,
                    batch_size=batch_size * 5,  # Larger batch size for citations
                    use_background_tasks=use_background_tasks
                )
            
            # Step 5: Create initial graph projection for analysis
            logger.info("Step 5: Creating initial graph projection")
            projection = GraphProjection(
                name="migration_analysis",
                node_labels=["Paper", "Author"],
                relationship_types=["CITES", "AUTHORED"],
                orientation="NATURAL"
            )
            
            try:
                await self.neo4j.create_graph_projection(projection)
                logger.info("Initial graph projection created successfully")
            except Exception as e:
                logger.warning(f"Failed to create initial projection: {e}")
        
        except Exception as e:
            logger.error(f"Complete migration failed: {e}")
            raise
        
        finally:
            self.stats.end_time = datetime.utcnow()
        
        # Log migration summary
        logger.info(f"""
Migration Summary:
- Duration: {self.stats.duration:.2f} seconds
- Papers: {self.stats.papers_migrated}/{self.stats.papers_processed}
- Authors: {self.stats.authors_migrated}/{self.stats.authors_processed}
- Citations: {self.stats.citations_migrated}/{self.stats.citations_processed}
- Authorships: {self.stats.authorships_migrated}/{self.stats.authorships_processed}
- Overall Success Rate: {self.stats.success_rate:.2%}
        """)
        
        return self.stats
    
    # ==================== UTILITY METHODS ====================
    
    async def validate_migration(self) -> Dict[str, Any]:
        """Validate the migrated data."""
        logger.info("Validating migrated data")
        
        stats = await self.crud_ops.get_node_statistics()
        
        # Check for orphaned nodes
        orphaned = await self.crud_ops.cleanup_orphaned_nodes()
        
        # Basic validation checks
        validation_results = {
            "node_counts": stats,
            "orphaned_nodes_cleaned": orphaned,
            "validation_checks": {
                "papers_have_titles": await self._count_papers_with_titles(),
                "authors_have_names": await self._count_authors_with_names(),
                "citations_are_valid": await self._count_valid_citations(),
            }
        }
        
        return validation_results
    
    async def _count_papers_with_titles(self) -> int:
        """Count papers that have titles."""
        query = """
        MATCH (p:Paper)
        WHERE p.title IS NOT NULL AND p.title <> ""
        RETURN count(p) as count
        """
        
        result = await self.neo4j.execute_read(query)
        return result[0]["count"] if result else 0
    
    async def _count_authors_with_names(self) -> int:
        """Count authors that have names."""
        query = """
        MATCH (a:Author)
        WHERE a.name IS NOT NULL AND a.name <> ""
        RETURN count(a) as count
        """
        
        result = await self.neo4j.execute_read(query)
        return result[0]["count"] if result else 0
    
    async def _count_valid_citations(self) -> int:
        """Count valid citation relationships."""
        query = """
        MATCH (p1:Paper)-[r:CITES]->(p2:Paper)
        WHERE p1.id IS NOT NULL AND p2.id IS NOT NULL
        RETURN count(r) as count
        """
        
        result = await self.neo4j.execute_read(query)
        return result[0]["count"] if result else 0
    
    async def export_migration_report(self, file_path: str):
        """Export migration statistics to a JSON file."""
        report = {
            "migration_stats": {
                "papers_processed": self.stats.papers_processed,
                "papers_migrated": self.stats.papers_migrated,
                "papers_failed": self.stats.papers_failed,
                "authors_processed": self.stats.authors_processed,
                "authors_migrated": self.stats.authors_migrated,
                "citations_processed": self.stats.citations_processed,
                "citations_migrated": self.stats.citations_migrated,
                "duration": self.stats.duration,
                "success_rate": self.stats.success_rate
            },
            "timestamp": datetime.utcnow().isoformat(),
            "validation": await self.validate_migration()
        }
        
        with open(file_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Migration report exported to {file_path}")


# ==================== CONVENIENCE FUNCTIONS ====================

async def migrate_from_json_files(
    papers_json_path: str,
    edges_json_path: str = None,
    neo4j_manager: AdvancedNeo4jManager = None,
    batch_size: int = 100,
    use_background_tasks: bool = True
) -> MigrationStats:
    """Convenience function to migrate from JSON files."""
    
    if neo4j_manager is None:
        neo4j_manager = AdvancedNeo4jManager()
        await neo4j_manager.connect()
    
    migration_manager = DataMigrationManager(neo4j_manager)
    
    # Load papers data
    papers_data = []
    papers_path = Path(papers_json_path)
    if papers_path.exists():
        with open(papers_path, 'r') as f:
            papers_data = json.load(f)
        logger.info(f"Loaded {len(papers_data)} papers from {papers_json_path}")
    else:
        raise FileNotFoundError(f"Papers file not found: {papers_json_path}")
    
    # Load edges data if provided
    edges_data = None
    if edges_json_path:
        edges_path = Path(edges_json_path)
        if edges_path.exists():
            with open(edges_path, 'r') as f:
                edges_data = json.load(f)
            logger.info(f"Loaded {len(edges_data)} edges from {edges_json_path}")
        else:
            logger.warning(f"Edges file not found: {edges_json_path}")
    
    # Perform migration
    stats = await migration_manager.migrate_complete_dataset(
        papers_data=papers_data,
        edges_data=edges_data,
        batch_size=batch_size,
        use_background_tasks=use_background_tasks
    )
    
    return stats