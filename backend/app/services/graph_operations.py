"""
Comprehensive CRUD operations for the Neo4j citation network graph.
Handles papers, authors, institutions, and relationships with data validation.
"""

from typing import Any, Dict, List, Optional, Union, Tuple, Set
import uuid
import json
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum

from ..db.neo4j_advanced import AdvancedNeo4jManager, GraphProjection
from ..utils.logger import get_logger

logger = get_logger(__name__)


class NodeType(str, Enum):
    """Graph node types."""
    PAPER = "Paper"
    AUTHOR = "Author"
    INSTITUTION = "Institution"
    VENUE = "Venue"
    FIELD = "Field"
    CONCEPT = "Concept"


class RelationshipType(str, Enum):
    """Graph relationship types."""
    CITES = "CITES"
    AUTHORED = "AUTHORED"
    AFFILIATED_WITH = "AFFILIATED_WITH"
    PUBLISHED_IN = "PUBLISHED_IN"
    HAS_FIELD = "HAS_FIELD"
    HAS_CONCEPT = "HAS_CONCEPT"
    COLLABORATES_WITH = "COLLABORATES_WITH"
    CO_AUTHORED = "CO_AUTHORED"


@dataclass
class PaperNode:
    """Paper node structure."""
    id: str
    doi: Optional[str] = None
    title: str = ""
    abstract: Optional[str] = None
    publication_year: Optional[int] = None
    publication_date: Optional[str] = None
    citation_count: int = 0
    reference_count: int = 0
    influential_citation_count: int = 0
    h_index: Optional[float] = None
    keywords: Optional[List[str]] = None
    language: Optional[str] = None
    open_access: Optional[bool] = None
    pdf_urls: Optional[List[str]] = None
    external_ids: Optional[Dict[str, str]] = None
    created_at: str = None
    updated_at: str = None
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())
        if not self.created_at:
            self.created_at = datetime.utcnow().isoformat()
        self.updated_at = datetime.utcnow().isoformat()


@dataclass
class AuthorNode:
    """Author node structure."""
    id: str
    name: str
    display_name: Optional[str] = None
    orcid: Optional[str] = None
    h_index: Optional[int] = None
    citation_count: int = 0
    paper_count: int = 0
    affiliations: Optional[List[str]] = None
    fields_of_study: Optional[List[str]] = None
    external_ids: Optional[Dict[str, str]] = None
    created_at: str = None
    updated_at: str = None
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())
        if not self.created_at:
            self.created_at = datetime.utcnow().isoformat()
        self.updated_at = datetime.utcnow().isoformat()


@dataclass
class InstitutionNode:
    """Institution node structure."""
    id: str
    name: str
    display_name: Optional[str] = None
    country: Optional[str] = None
    type: Optional[str] = None
    homepage_url: Optional[str] = None
    image_url: Optional[str] = None
    geo_coordinates: Optional[Dict[str, float]] = None
    external_ids: Optional[Dict[str, str]] = None
    created_at: str = None
    updated_at: str = None
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())
        if not self.created_at:
            self.created_at = datetime.utcnow().isoformat()
        self.updated_at = datetime.utcnow().isoformat()


@dataclass
class VenueNode:
    """Venue/Journal node structure."""
    id: str
    name: str
    display_name: Optional[str] = None
    issn: Optional[str] = None
    type: Optional[str] = None  # journal, conference, etc.
    publisher: Optional[str] = None
    impact_factor: Optional[float] = None
    h_index: Optional[int] = None
    homepage_url: Optional[str] = None
    external_ids: Optional[Dict[str, str]] = None
    created_at: str = None
    updated_at: str = None
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())
        if not self.created_at:
            self.created_at = datetime.utcnow().isoformat()
        self.updated_at = datetime.utcnow().isoformat()


class GraphCRUDOperations:
    """Comprehensive CRUD operations for the citation network graph."""
    
    def __init__(self, neo4j_manager: AdvancedNeo4jManager):
        self.neo4j = neo4j_manager
    
    # ==================== PAPER OPERATIONS ====================
    
    async def create_paper(self, paper: Union[PaperNode, Dict]) -> str:
        """Create a paper node."""
        if isinstance(paper, dict):
            paper = PaperNode(**paper)
        
        paper_dict = asdict(paper)
        
        # Handle list fields that need to be stored as arrays
        if paper_dict.get("keywords"):
            paper_dict["keywords"] = json.dumps(paper_dict["keywords"])
        if paper_dict.get("pdf_urls"):
            paper_dict["pdf_urls"] = json.dumps(paper_dict["pdf_urls"])
        if paper_dict.get("external_ids"):
            paper_dict["external_ids"] = json.dumps(paper_dict["external_ids"])
        
        query = """
        MERGE (p:Paper {id: $id})
        SET p += $properties
        RETURN p.id as paper_id
        """
        
        async with self.neo4j.get_session() as session:
            result = await session.run(query, {
                "id": paper.id,
                "properties": paper_dict
            })
            record = await result.single()
            
        logger.info(f"Created/updated paper: {paper.id}")
        return record["paper_id"] if record else paper.id
    
    async def get_paper(self, paper_id: str) -> Optional[Dict]:
        """Get paper by ID."""
        query = """
        MATCH (p:Paper {id: $paper_id})
        RETURN p
        """
        
        async with self.neo4j.get_session() as session:
            result = await session.run(query, {"paper_id": paper_id})
            record = await result.single()
            
        if record:
            paper_data = dict(record["p"])
            # Parse JSON fields back to lists/dicts
            for field in ["keywords", "pdf_urls", "external_ids"]:
                if paper_data.get(field):
                    try:
                        paper_data[field] = json.loads(paper_data[field])
                    except json.JSONDecodeError:
                        pass
            return paper_data
        return None
    
    async def update_paper(self, paper_id: str, updates: Dict) -> bool:
        """Update paper properties."""
        # Handle JSON fields
        for field in ["keywords", "pdf_urls", "external_ids"]:
            if field in updates and isinstance(updates[field], (list, dict)):
                updates[field] = json.dumps(updates[field])
        
        updates["updated_at"] = datetime.utcnow().isoformat()
        
        query = """
        MATCH (p:Paper {id: $paper_id})
        SET p += $updates
        RETURN p.id as updated_id
        """
        
        async with self.neo4j.get_session() as session:
            result = await session.run(query, {
                "paper_id": paper_id,
                "updates": updates
            })
            record = await result.single()
            
        success = bool(record)
        if success:
            logger.info(f"Updated paper: {paper_id}")
        return success
    
    async def delete_paper(self, paper_id: str) -> bool:
        """Delete paper and all its relationships."""
        query = """
        MATCH (p:Paper {id: $paper_id})
        DETACH DELETE p
        RETURN count(*) as deleted_count
        """
        
        async with self.neo4j.get_session() as session:
            result = await session.run(query, {"paper_id": paper_id})
            record = await result.single()
            
        deleted = record["deleted_count"] > 0
        if deleted:
            logger.info(f"Deleted paper: {paper_id}")
        return deleted
    
    async def search_papers(
        self,
        query_text: str,
        limit: int = 20,
        skip: int = 0,
        filters: Optional[Dict] = None
    ) -> List[Dict]:
        """Search papers with full-text search and filters."""
        # Build filter conditions
        filter_conditions = []
        filter_params = {"query": query_text, "limit": limit, "skip": skip}
        
        if filters:
            if filters.get("publication_year"):
                filter_conditions.append("p.publication_year = $year")
                filter_params["year"] = filters["publication_year"]
            
            if filters.get("min_citations"):
                filter_conditions.append("p.citation_count >= $min_citations")
                filter_params["min_citations"] = filters["min_citations"]
            
            if filters.get("language"):
                filter_conditions.append("p.language = $language")
                filter_params["language"] = filters["language"]
        
        where_clause = ""
        if filter_conditions:
            where_clause = f"WHERE {' AND '.join(filter_conditions)}"
        
        cypher_query = f"""
        CALL db.index.fulltext.queryNodes('paperFulltext', $query)
        YIELD node as p, score
        {where_clause}
        RETURN p, score
        ORDER BY score DESC
        SKIP $skip
        LIMIT $limit
        """
        
        async with self.neo4j.get_session() as session:
            result = await session.run(cypher_query, filter_params)
            records = await result.data()
            
        papers = []
        for record in records:
            paper_data = dict(record["p"])
            # Parse JSON fields
            for field in ["keywords", "pdf_urls", "external_ids"]:
                if paper_data.get(field):
                    try:
                        paper_data[field] = json.loads(paper_data[field])
                    except json.JSONDecodeError:
                        pass
            paper_data["search_score"] = record["score"]
            papers.append(paper_data)
        
        return papers
    
    # ==================== AUTHOR OPERATIONS ====================
    
    async def create_author(self, author: Union[AuthorNode, Dict]) -> str:
        """Create an author node."""
        if isinstance(author, dict):
            author = AuthorNode(**author)
        
        author_dict = asdict(author)
        
        # Handle list/dict fields
        for field in ["affiliations", "fields_of_study", "external_ids"]:
            if author_dict.get(field):
                author_dict[field] = json.dumps(author_dict[field])
        
        query = """
        MERGE (a:Author {id: $id})
        SET a += $properties
        RETURN a.id as author_id
        """
        
        async with self.neo4j.get_session() as session:
            result = await session.run(query, {
                "id": author.id,
                "properties": author_dict
            })
            record = await result.single()
            
        logger.info(f"Created/updated author: {author.id}")
        return record["author_id"] if record else author.id
    
    async def get_author(self, author_id: str) -> Optional[Dict]:
        """Get author by ID."""
        query = """
        MATCH (a:Author {id: $author_id})
        RETURN a
        """
        
        async with self.neo4j.get_session() as session:
            result = await session.run(query, {"author_id": author_id})
            record = await result.single()
            
        if record:
            author_data = dict(record["a"])
            # Parse JSON fields
            for field in ["affiliations", "fields_of_study", "external_ids"]:
                if author_data.get(field):
                    try:
                        author_data[field] = json.loads(author_data[field])
                    except json.JSONDecodeError:
                        pass
            return author_data
        return None
    
    async def get_author_papers(
        self,
        author_id: str,
        limit: int = 100,
        sort_by: str = "publication_year"
    ) -> List[Dict]:
        """Get papers authored by a specific author."""
        sort_clause = f"ORDER BY p.{sort_by} DESC" if sort_by in ["publication_year", "citation_count"] else ""
        
        query = f"""
        MATCH (a:Author {{id: $author_id}})-[:AUTHORED]->(p:Paper)
        RETURN p
        {sort_clause}
        LIMIT $limit
        """
        
        async with self.neo4j.get_session() as session:
            result = await session.run(query, {
                "author_id": author_id,
                "limit": limit
            })
            records = await result.data()
            
        papers = []
        for record in records:
            paper_data = dict(record["p"])
            for field in ["keywords", "pdf_urls", "external_ids"]:
                if paper_data.get(field):
                    try:
                        paper_data[field] = json.loads(paper_data[field])
                    except json.JSONDecodeError:
                        pass
            papers.append(paper_data)
        
        return papers
    
    # ==================== INSTITUTION OPERATIONS ====================
    
    async def create_institution(self, institution: Union[InstitutionNode, Dict]) -> str:
        """Create an institution node."""
        if isinstance(institution, dict):
            institution = InstitutionNode(**institution)
        
        institution_dict = asdict(institution)
        
        # Handle dict fields
        for field in ["geo_coordinates", "external_ids"]:
            if institution_dict.get(field):
                institution_dict[field] = json.dumps(institution_dict[field])
        
        query = """
        MERGE (i:Institution {id: $id})
        SET i += $properties
        RETURN i.id as institution_id
        """
        
        async with self.neo4j.get_session() as session:
            result = await session.run(query, {
                "id": institution.id,
                "properties": institution_dict
            })
            record = await result.single()
            
        logger.info(f"Created/updated institution: {institution.id}")
        return record["institution_id"] if record else institution.id
    
    # ==================== VENUE OPERATIONS ====================
    
    async def create_venue(self, venue: Union[VenueNode, Dict]) -> str:
        """Create a venue node."""
        if isinstance(venue, dict):
            venue = VenueNode(**venue)
        
        venue_dict = asdict(venue)
        
        # Handle dict fields
        if venue_dict.get("external_ids"):
            venue_dict["external_ids"] = json.dumps(venue_dict["external_ids"])
        
        query = """
        MERGE (v:Venue {id: $id})
        SET v += $properties
        RETURN v.id as venue_id
        """
        
        async with self.neo4j.get_session() as session:
            result = await session.run(query, {
                "id": venue.id,
                "properties": venue_dict
            })
            record = await result.single()
            
        logger.info(f"Created/updated venue: {venue.id}")
        return record["venue_id"] if record else venue.id
    
    # ==================== RELATIONSHIP OPERATIONS ====================
    
    async def create_citation_relationship(
        self,
        citing_paper_id: str,
        cited_paper_id: str,
        context: Optional[str] = None,
        citation_intent: Optional[str] = None,
        is_influential: bool = False
    ) -> bool:
        """Create citation relationship between papers."""
        relationship_props = {
            "created_at": datetime.utcnow().isoformat(),
            "is_influential": is_influential
        }
        
        if context:
            relationship_props["context"] = context
        if citation_intent:
            relationship_props["citation_intent"] = citation_intent
        
        query = """
        MATCH (citing:Paper {id: $citing_id})
        MATCH (cited:Paper {id: $cited_id})
        MERGE (citing)-[r:CITES]->(cited)
        SET r += $properties
        RETURN r
        """
        
        async with self.neo4j.get_session() as session:
            result = await session.run(query, {
                "citing_id": citing_paper_id,
                "cited_id": cited_paper_id,
                "properties": relationship_props
            })
            record = await result.single()
            
        success = bool(record)
        if success:
            logger.info(f"Created citation: {citing_paper_id} -> {cited_paper_id}")
        return success
    
    async def create_authorship_relationship(
        self,
        author_id: str,
        paper_id: str,
        position: Optional[int] = None,
        is_corresponding: bool = False
    ) -> bool:
        """Create authorship relationship."""
        relationship_props = {
            "created_at": datetime.utcnow().isoformat(),
            "is_corresponding": is_corresponding
        }
        
        if position is not None:
            relationship_props["position"] = position
        
        query = """
        MATCH (a:Author {id: $author_id})
        MATCH (p:Paper {id: $paper_id})
        MERGE (a)-[r:AUTHORED]->(p)
        SET r += $properties
        RETURN r
        """
        
        async with self.neo4j.get_session() as session:
            result = await session.run(query, {
                "author_id": author_id,
                "paper_id": paper_id,
                "properties": relationship_props
            })
            record = await result.single()
            
        success = bool(record)
        if success:
            logger.info(f"Created authorship: {author_id} -> {paper_id}")
        return success
    
    async def create_affiliation_relationship(
        self,
        author_id: str,
        institution_id: str,
        start_year: Optional[int] = None,
        end_year: Optional[int] = None,
        position: Optional[str] = None
    ) -> bool:
        """Create affiliation relationship."""
        relationship_props = {
            "created_at": datetime.utcnow().isoformat()
        }
        
        if start_year:
            relationship_props["start_year"] = start_year
        if end_year:
            relationship_props["end_year"] = end_year
        if position:
            relationship_props["position"] = position
        
        query = """
        MATCH (a:Author {id: $author_id})
        MATCH (i:Institution {id: $institution_id})
        MERGE (a)-[r:AFFILIATED_WITH]->(i)
        SET r += $properties
        RETURN r
        """
        
        async with self.neo4j.get_session() as session:
            result = await session.run(query, {
                "author_id": author_id,
                "institution_id": institution_id,
                "properties": relationship_props
            })
            record = await result.single()
            
        success = bool(record)
        if success:
            logger.info(f"Created affiliation: {author_id} -> {institution_id}")
        return success
    
    # ==================== BATCH OPERATIONS ====================
    
    async def batch_create_papers(self, papers: List[Union[PaperNode, Dict]]) -> List[str]:
        """Create multiple papers in batch."""
        created_ids = []
        
        # Convert all to dictionaries and prepare for batch insert
        paper_dicts = []
        for paper in papers:
            if isinstance(paper, dict):
                paper_obj = PaperNode(**paper)
            else:
                paper_obj = paper
            
            paper_dict = asdict(paper_obj)
            
            # Handle JSON fields
            for field in ["keywords", "pdf_urls", "external_ids"]:
                if paper_dict.get(field):
                    paper_dict[field] = json.dumps(paper_dict[field])
            
            paper_dicts.append(paper_dict)
        
        query = """
        UNWIND $papers as paper
        MERGE (p:Paper {id: paper.id})
        SET p += paper
        RETURN p.id as paper_id
        """
        
        async with self.neo4j.get_session() as session:
            result = await session.run(query, {"papers": paper_dicts})
            records = await result.data()
            
        created_ids = [record["paper_id"] for record in records]
        logger.info(f"Batch created {len(created_ids)} papers")
        
        return created_ids
    
    async def batch_create_citations(self, citations: List[Tuple[str, str]]) -> int:
        """Create multiple citations in batch."""
        citation_data = []
        for citing_id, cited_id in citations:
            citation_data.append({
                "citing_id": citing_id,
                "cited_id": cited_id,
                "created_at": datetime.utcnow().isoformat(),
                "is_influential": False
            })
        
        query = """
        UNWIND $citations as citation
        MATCH (citing:Paper {id: citation.citing_id})
        MATCH (cited:Paper {id: citation.cited_id})
        MERGE (citing)-[r:CITES]->(cited)
        SET r.created_at = citation.created_at,
            r.is_influential = citation.is_influential
        RETURN count(r) as created_count
        """
        
        async with self.neo4j.get_session() as session:
            result = await session.run(query, {"citations": citation_data})
            record = await result.single()
            
        created_count = record["created_count"] if record else 0
        logger.info(f"Batch created {created_count} citations")
        
        return created_count
    
    # ==================== UTILITY OPERATIONS ====================
    
    async def get_node_statistics(self) -> Dict[str, int]:
        """Get counts of different node types."""
        queries = {
            "papers": "MATCH (p:Paper) RETURN count(p) as count",
            "authors": "MATCH (a:Author) RETURN count(a) as count",
            "institutions": "MATCH (i:Institution) RETURN count(i) as count",
            "venues": "MATCH (v:Venue) RETURN count(v) as count",
            "citations": "MATCH ()-[r:CITES]->() RETURN count(r) as count",
            "authorships": "MATCH ()-[r:AUTHORED]->() RETURN count(r) as count",
            "affiliations": "MATCH ()-[r:AFFILIATED_WITH]->() RETURN count(r) as count"
        }
        
        statistics = {}
        async with self.neo4j.get_session() as session:
            for stat_name, query in queries.items():
                try:
                    result = await session.run(query)
                    record = await result.single()
                    statistics[stat_name] = record["count"] if record else 0
                except Exception as e:
                    logger.error(f"Error getting {stat_name} count: {e}")
                    statistics[stat_name] = 0
        
        return statistics
    
    async def cleanup_orphaned_nodes(self) -> Dict[str, int]:
        """Clean up nodes that have no relationships."""
        queries = {
            "orphaned_papers": """
                MATCH (p:Paper)
                WHERE NOT (p)--()
                DELETE p
                RETURN count(*) as deleted_count
            """,
            "orphaned_authors": """
                MATCH (a:Author)
                WHERE NOT (a)--()
                DELETE a
                RETURN count(*) as deleted_count
            """,
            "orphaned_institutions": """
                MATCH (i:Institution)
                WHERE NOT (i)--()
                DELETE i
                RETURN count(*) as deleted_count
            """
        }
        
        cleanup_results = {}
        async with self.neo4j.get_session() as session:
            for node_type, query in queries.items():
                try:
                    result = await session.run(query)
                    record = await result.single()
                    cleanup_results[node_type] = record["deleted_count"] if record else 0
                except Exception as e:
                    logger.error(f"Error cleaning up {node_type}: {e}")
                    cleanup_results[node_type] = 0
        
        total_cleaned = sum(cleanup_results.values())
        if total_cleaned > 0:
            logger.info(f"Cleaned up {total_cleaned} orphaned nodes")
        
        return cleanup_results