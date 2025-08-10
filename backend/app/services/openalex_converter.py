"""
Data conversion utilities for transforming OpenAlex data to internal models.

Converts OpenAlex API responses to our internal Paper, Author, and Journal models
while preserving important metadata and relationships.
"""

from typing import Any, Dict, List, Optional
from datetime import datetime, date
from urllib.parse import urlparse

from ..models.openalex import OpenAlexWork, OpenAlexAuthor, OpenAlexAuthorship, OpenAlexInstitution
from ..models.paper import (
    Paper,
    Author,
    Journal,
    CitationCount,
    PaperEdge
)
from ..utils.logger import get_logger

logger = get_logger(__name__)


class OpenAlexConverter:
    """Converter for OpenAlex data to internal models."""
    
    @staticmethod
    def convert_openalex_work_to_paper(work: OpenAlexWork) -> Paper:
        """
        Convert OpenAlex work to internal Paper model.
        
        Args:
            work: OpenAlexWork object
            
        Returns:
            Paper object with converted data
        """
        try:
            # Extract identifiers
            paper_id = work.id.split('/')[-1] if work.id else None
            doi = work.ids.doi
            pmid = work.ids.pmid
            arxiv_id = work.ids.arxiv
            openalex_id = work.id
            
            # Convert authors
            authors = OpenAlexConverter._convert_authorships_to_authors(work.authorships)
            
            # Convert journal information
            journal = OpenAlexConverter._convert_location_to_journal(work.primary_location)
            
            # Extract publication date and year
            publication_date = work.publication_date
            publication_year = work.publication_year
            
            # Get URLs
            url = None
            pdf_url = None
            open_access_url = None
            
            if work.best_oa_location:
                if isinstance(work.best_oa_location, dict):
                    open_access_url = work.best_oa_location.get('landing_page_url')
                    pdf_url = work.best_oa_location.get('pdf_url')
            
            if work.primary_location:
                if isinstance(work.primary_location, dict):
                    url = work.primary_location.get('landing_page_url')
            
            # Convert citation information
            citation_count = CitationCount(
                total=work.cited_by_count,
                openalex=work.cited_by_count,
                last_updated=datetime.utcnow()
            )
            
            # Get references and citations
            references = work.referenced_works or []
            # Note: cited_by would need a separate API call to get full list
            cited_by = []  # This would be populated through citation network traversal
            
            # Extract subjects/concepts
            subjects = [concept.display_name for concept in work.concepts[:10]]  # Top 10 concepts
            
            # Extract keywords from concepts and MeSH terms
            keywords = []
            keywords.extend([concept.display_name for concept in work.concepts[:5]])
            keywords.extend([mesh.descriptor_name for mesh in work.mesh[:5]])
            keywords.extend([keyword.display_name for keyword in work.keywords[:10]])
            keywords = list(set(keywords))  # Remove duplicates
            
            # Convert concepts to our format
            concepts = []
            for concept in work.concepts:
                concepts.append({
                    "id": concept.id,
                    "name": concept.display_name,
                    "level": concept.level,
                    "score": concept.score
                })
            
            # Extract bibliographic details
            volume = None
            issue = None
            pages = None
            
            if work.biblio:
                volume = work.biblio.volume
                issue = work.biblio.issue
                if work.biblio.first_page and work.biblio.last_page:
                    pages = f"{work.biblio.first_page}-{work.biblio.last_page}"
                elif work.biblio.first_page:
                    pages = work.biblio.first_page
            
            # Determine paper type
            paper_type = work.type or work.type_crossref
            
            # Language
            language = work.language
            
            # Open access status
            is_open_access = work.is_open_access()
            
            # Get abstract text
            abstract = work.get_abstract_text()
            
            # Create Paper object
            paper = Paper(
                id=paper_id,
                doi=doi,
                pmid=pmid,
                arxiv_id=arxiv_id,
                openalex_id=openalex_id,
                title=work.title or work.display_name,
                abstract=abstract,
                authors=authors,
                journal=journal,
                publication_date=publication_date,
                publication_year=publication_year,
                volume=volume,
                issue=issue,
                pages=pages,
                url=url,
                pdf_url=pdf_url,
                open_access_url=open_access_url,
                citation_count=citation_count,
                references=[ref.split('/')[-1] if ref.startswith('https://') else ref for ref in references],
                cited_by=cited_by,
                subjects=subjects,
                keywords=keywords,
                concepts=concepts,
                language=language,
                paper_type=paper_type,
                is_open_access=is_open_access,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            
            return paper
            
        except Exception as e:
            logger.error(f"Error converting OpenAlex work to Paper: {e}")
            logger.error(f"Work data: {work.dict() if hasattr(work, 'dict') else str(work)}")
            raise
    
    @staticmethod
    def _convert_authorships_to_authors(authorships: List[OpenAlexAuthorship]) -> List[Author]:
        """Convert OpenAlex authorships to Author objects."""
        authors = []
        
        for authorship in authorships:
            try:
                # Get primary affiliation
                affiliation = None
                if authorship.institutions:
                    affiliation = authorship.institutions[0].display_name
                
                author = Author(
                    name=authorship.author.display_name,
                    orcid=authorship.author.orcid,
                    affiliation=affiliation
                )
                
                authors.append(author)
                
            except Exception as e:
                logger.warning(f"Error converting authorship to Author: {e}")
                # Create minimal author with just name
                if hasattr(authorship, 'author') and hasattr(authorship.author, 'display_name'):
                    authors.append(Author(name=authorship.author.display_name))
        
        return authors
    
    @staticmethod
    def _convert_location_to_journal(location: Optional[Dict[str, Any]]) -> Optional[Journal]:
        """Convert OpenAlex location to Journal object."""
        if not location or not isinstance(location, dict):
            return None
        
        source = location.get('source')
        if not source or not isinstance(source, dict):
            return None
        
        try:
            journal_name = source.get('display_name')
            if not journal_name:
                return None
            
            # Extract ISSN
            issn = None
            if source.get('issn'):
                issn = source['issn'][0] if isinstance(source['issn'], list) else source['issn']
            elif source.get('issn_l'):
                issn = source['issn_l']
            
            # Extract publisher
            publisher = source.get('host_organization_name')
            
            journal = Journal(
                name=journal_name,
                issn=issn,
                publisher=publisher
            )
            
            return journal
            
        except Exception as e:
            logger.warning(f"Error converting location to Journal: {e}")
            return None
    
    @staticmethod
    def convert_citation_relationships_to_edges(
        center_paper_id: str,
        cited_works: List[OpenAlexWork],
        citing_works: List[OpenAlexWork]
    ) -> List[PaperEdge]:
        """
        Convert citation relationships to PaperEdge objects.
        
        Args:
            center_paper_id: ID of the center paper
            cited_works: Works that the center paper cites
            citing_works: Works that cite the center paper
            
        Returns:
            List of PaperEdge objects
        """
        edges = []
        
        # Add edges for works that the center paper cites (outgoing edges)
        for cited_work in cited_works:
            try:
                cited_id = cited_work.id.split('/')[-1] if cited_work.id else None
                if cited_id:
                    edge = PaperEdge(
                        source_id=center_paper_id,
                        target_id=cited_id,
                        edge_type="cites",
                        weight=1.0,
                        created_at=datetime.utcnow()
                    )
                    edges.append(edge)
            except Exception as e:
                logger.warning(f"Error creating citation edge: {e}")
        
        # Add edges for works that cite the center paper (incoming edges)
        for citing_work in citing_works:
            try:
                citing_id = citing_work.id.split('/')[-1] if citing_work.id else None
                if citing_id:
                    edge = PaperEdge(
                        source_id=citing_id,
                        target_id=center_paper_id,
                        edge_type="cites",
                        weight=1.0,
                        created_at=datetime.utcnow()
                    )
                    edges.append(edge)
            except Exception as e:
                logger.warning(f"Error creating citing edge: {e}")
        
        return edges
    
    @staticmethod
    def batch_convert_works_to_papers(works: List[OpenAlexWork]) -> List[Paper]:
        """
        Convert multiple OpenAlex works to Paper objects.
        
        Args:
            works: List of OpenAlexWork objects
            
        Returns:
            List of Paper objects
        """
        papers = []
        
        for work in works:
            try:
                paper = OpenAlexConverter.convert_openalex_work_to_paper(work)
                papers.append(paper)
            except Exception as e:
                logger.error(f"Error converting work {work.id if hasattr(work, 'id') else 'unknown'}: {e}")
                continue
        
        return papers
    
    @staticmethod
    def extract_paper_metadata_summary(paper: Paper) -> Dict[str, Any]:
        """
        Extract summary metadata from a Paper object for quick access.
        
        Args:
            paper: Paper object
            
        Returns:
            Dictionary with key metadata fields
        """
        return {
            "id": paper.id,
            "title": paper.title,
            "authors": [author.name for author in paper.authors],
            "journal": paper.journal.name if paper.journal else None,
            "publication_year": paper.publication_year,
            "citation_count": paper.citation_count.total,
            "doi": paper.doi,
            "openalex_id": paper.openalex_id,
            "is_open_access": paper.is_open_access,
            "concepts": [concept["name"] for concept in paper.concepts[:5]],
            "language": paper.language
        }
    
    @staticmethod
    def create_paper_from_minimal_data(
        title: str,
        authors: List[str],
        doi: Optional[str] = None,
        year: Optional[int] = None,
        journal_name: Optional[str] = None
    ) -> Paper:
        """
        Create a Paper object from minimal data (useful for imports/user input).
        
        Args:
            title: Paper title
            authors: List of author names
            doi: DOI if available
            year: Publication year if available
            journal_name: Journal name if available
            
        Returns:
            Paper object with minimal data
        """
        # Convert author names to Author objects
        author_objects = [Author(name=name) for name in authors]
        
        # Create journal object if name provided
        journal = Journal(name=journal_name) if journal_name else None
        
        # Create basic citation count
        citation_count = CitationCount(total=0)
        
        paper = Paper(
            title=title,
            authors=author_objects,
            doi=doi,
            publication_year=year,
            journal=journal,
            citation_count=citation_count,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        return paper
    
    @staticmethod
    def merge_paper_data(existing_paper: Paper, openalex_paper: Paper) -> Paper:
        """
        Merge data from OpenAlex with existing paper data, preferring more complete information.
        
        Args:
            existing_paper: Existing paper with potentially incomplete data
            openalex_paper: Paper data from OpenAlex
            
        Returns:
            Merged Paper object
        """
        # Start with existing paper data
        merged_data = existing_paper.dict()
        openalex_data = openalex_paper.dict()
        
        # Merge fields, preferring non-empty values from OpenAlex
        merge_fields = [
            'abstract', 'publication_date', 'publication_year', 'volume', 'issue', 'pages',
            'url', 'pdf_url', 'open_access_url', 'subjects', 'keywords', 'concepts',
            'language', 'paper_type', 'is_open_access'
        ]
        
        for field in merge_fields:
            if openalex_data.get(field) and not merged_data.get(field):
                merged_data[field] = openalex_data[field]
        
        # Merge identifiers
        if openalex_data.get('openalex_id') and not merged_data.get('openalex_id'):
            merged_data['openalex_id'] = openalex_data['openalex_id']
        
        if openalex_data.get('pmid') and not merged_data.get('pmid'):
            merged_data['pmid'] = openalex_data['pmid']
        
        if openalex_data.get('arxiv_id') and not merged_data.get('arxiv_id'):
            merged_data['arxiv_id'] = openalex_data['arxiv_id']
        
        # Merge authors if existing has fewer
        if len(openalex_data.get('authors', [])) > len(merged_data.get('authors', [])):
            merged_data['authors'] = openalex_data['authors']
        
        # Merge journal if not present
        if openalex_data.get('journal') and not merged_data.get('journal'):
            merged_data['journal'] = openalex_data['journal']
        
        # Update citation count with OpenAlex data
        if openalex_data.get('citation_count'):
            merged_data['citation_count'] = openalex_data['citation_count']
        
        # Update references and citations
        if openalex_data.get('references'):
            merged_data['references'] = list(set(
                merged_data.get('references', []) + openalex_data.get('references', [])
            ))
        
        if openalex_data.get('cited_by'):
            merged_data['cited_by'] = list(set(
                merged_data.get('cited_by', []) + openalex_data.get('cited_by', [])
            ))
        
        # Update timestamps
        merged_data['updated_at'] = datetime.utcnow()
        
        return Paper(**merged_data)
    
    @staticmethod
    def validate_paper_completeness(paper: Paper) -> Dict[str, Any]:
        """
        Validate paper data completeness and return a report.
        
        Args:
            paper: Paper object to validate
            
        Returns:
            Dictionary with completeness metrics and missing fields
        """
        required_fields = ['title', 'authors']
        important_fields = ['doi', 'publication_year', 'abstract', 'journal']
        optional_fields = ['volume', 'issue', 'pages', 'subjects', 'keywords']
        
        missing_required = []
        missing_important = []
        missing_optional = []
        
        # Check required fields
        for field in required_fields:
            value = getattr(paper, field, None)
            if not value or (isinstance(value, list) and len(value) == 0):
                missing_required.append(field)
        
        # Check important fields
        for field in important_fields:
            value = getattr(paper, field, None)
            if not value or (isinstance(value, list) and len(value) == 0):
                missing_important.append(field)
        
        # Check optional fields
        for field in optional_fields:
            value = getattr(paper, field, None)
            if not value or (isinstance(value, list) and len(value) == 0):
                missing_optional.append(field)
        
        # Calculate completeness score
        total_fields = len(required_fields) + len(important_fields) + len(optional_fields)
        missing_fields = len(missing_required) + len(missing_important) + len(missing_optional)
        completeness_score = (total_fields - missing_fields) / total_fields * 100
        
        # Determine data quality
        if missing_required:
            quality = "poor"
        elif len(missing_important) > 2:
            quality = "fair"
        elif len(missing_important) > 0:
            quality = "good"
        else:
            quality = "excellent"
        
        return {
            "completeness_score": round(completeness_score, 1),
            "quality": quality,
            "missing_required": missing_required,
            "missing_important": missing_important,
            "missing_optional": missing_optional,
            "has_identifiers": bool(paper.doi or paper.openalex_id or paper.pmid),
            "has_citations": paper.citation_count.total > 0 if paper.citation_count else False,
            "has_abstract": bool(paper.abstract),
            "author_count": len(paper.authors) if paper.authors else 0
        }