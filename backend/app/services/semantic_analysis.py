"""
Semantic analysis service for advanced citation and research analysis.

Provides high-level semantic analysis capabilities including citation intent
classification, influential citation detection, semantic similarity computation,
and research trend analysis using both OpenAlex and Semantic Scholar data.
"""

import asyncio
import statistics
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from datetime import datetime, timedelta
from collections import defaultdict, Counter

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

from ..models.semantic_scholar import (
    SemanticScholarPaper,
    SemanticScholarCitationNetwork,
    SemanticScholarInfluentialCitation,
    SemanticScholarSimilarityResult,
    SemanticScholarEmbedding,
    CitationIntent,
    EnrichedPaper
)
from ..models.openalex import OpenAlexWork
from ..services.semantic_scholar import SemanticScholarClient, get_semantic_scholar_client
from ..services.openalex import OpenAlexClient, get_openalex_client
from ..db.redis import RedisManager, get_redis_manager
from ..utils.logger import get_logger
from ..utils.exceptions import ValidationError

logger = get_logger(__name__)


class SemanticAnalysisService:
    """
    High-level semantic analysis service combining multiple data sources
    and providing advanced research intelligence capabilities.
    """
    
    def __init__(
        self,
        semantic_scholar_client: Optional[SemanticScholarClient] = None,
        openalex_client: Optional[OpenAlexClient] = None,
        redis_manager: Optional[RedisManager] = None
    ):
        self.semantic_scholar_client = semantic_scholar_client
        self.openalex_client = openalex_client
        self.redis_manager = redis_manager
        self._clients_initialized = False
    
    async def _ensure_clients(self):
        """Ensure all clients are initialized."""
        if not self._clients_initialized:
            if not self.semantic_scholar_client:
                self.semantic_scholar_client = await get_semantic_scholar_client(self.redis_manager)
            if not self.openalex_client:
                self.openalex_client = await get_openalex_client(self.redis_manager)
            if not self.redis_manager:
                self.redis_manager = await get_redis_manager()
            
            self._clients_initialized = True
    
    async def enrich_paper_with_semantic_features(
        self,
        paper_identifier: str,
        use_cache: bool = True
    ) -> Optional[EnrichedPaper]:
        """
        Enrich a paper with comprehensive semantic features from multiple sources.
        
        Args:
            paper_identifier: Paper DOI, OpenAlex ID, or Semantic Scholar ID
            use_cache: Whether to use caching
            
        Returns:
            EnrichedPaper with combined data from all sources
        """
        await self._ensure_clients()
        
        enriched_paper = EnrichedPaper(
            enrichment_timestamp=datetime.utcnow(),
            enrichment_sources=[]
        )
        
        # Try to get data from Semantic Scholar
        try:
            semantic_paper = await self.semantic_scholar_client.get_paper_by_id(
                paper_identifier,
                use_cache=use_cache
            )
            
            if semantic_paper:
                enriched_paper.semantic_scholar_data = semantic_paper
                enriched_paper.semantic_scholar_id = semantic_paper.paper_id
                enriched_paper.title = enriched_paper.title or semantic_paper.title
                enriched_paper.enrichment_sources.append("semantic_scholar")
                
                # Extract DOI if available
                if semantic_paper.external_ids and semantic_paper.external_ids.doi:
                    enriched_paper.doi = semantic_paper.external_ids.doi
                
        except Exception as e:
            logger.warning(f"Failed to get Semantic Scholar data for {paper_identifier}: {e}")
        
        # Try to get data from OpenAlex
        try:
            openalex_paper = await self.openalex_client.get_work_by_id(
                paper_identifier,
                use_cache=use_cache
            )
            
            if openalex_paper:
                enriched_paper.openalex_data = openalex_paper.dict()
                enriched_paper.openalex_id = openalex_paper.id
                enriched_paper.title = enriched_paper.title or openalex_paper.title
                enriched_paper.enrichment_sources.append("openalex")
                
                # Extract DOI if available
                if openalex_paper.ids and openalex_paper.ids.doi:
                    enriched_paper.doi = openalex_paper.ids.doi
                
        except Exception as e:
            logger.warning(f"Failed to get OpenAlex data for {paper_identifier}: {e}")
        
        if not enriched_paper.enrichment_sources:
            return None
        
        # Perform semantic analysis if Semantic Scholar data is available
        if enriched_paper.semantic_scholar_data:
            await self._analyze_citation_patterns(enriched_paper, use_cache)
            await self._analyze_influential_citations(enriched_paper, use_cache)
            await self._compute_semantic_similarities(enriched_paper, use_cache)
        
        return enriched_paper
    
    async def _analyze_citation_patterns(
        self,
        enriched_paper: EnrichedPaper,
        use_cache: bool = True
    ):
        """Analyze citation intent patterns for the paper."""
        if not enriched_paper.semantic_scholar_id:
            return
        
        try:
            # Get citations with intent data
            citations = await self.semantic_scholar_client.get_paper_citations(
                enriched_paper.semantic_scholar_id,
                limit=200,
                use_cache=use_cache
            )
            
            # Analyze citation intents
            intent_analysis = {
                "total_citations": len(citations),
                "intent_distribution": {},
                "intent_trends": {},
                "context_analysis": {}
            }
            
            # Count intent types
            intent_counts = Counter()
            context_lengths = []
            
            for citation in citations:
                if hasattr(citation, 'intents') and citation.intents:
                    for intent in citation.intents:
                        intent_counts[intent] += 1
                
                if hasattr(citation, 'contexts') and citation.contexts:
                    for context in citation.contexts:
                        context_lengths.append(len(context))
            
            # Calculate intent distribution
            total_intents = sum(intent_counts.values())
            if total_intents > 0:
                for intent, count in intent_counts.items():
                    intent_analysis["intent_distribution"][intent.value] = {
                        "count": count,
                        "percentage": (count / total_intents) * 100
                    }
            
            # Analyze context characteristics
            if context_lengths:
                intent_analysis["context_analysis"] = {
                    "avg_context_length": statistics.mean(context_lengths),
                    "median_context_length": statistics.median(context_lengths),
                    "total_contexts": len(context_lengths)
                }
            
            enriched_paper.citation_intent_analysis = intent_analysis
            
        except Exception as e:
            logger.warning(f"Citation pattern analysis failed: {e}")
    
    async def _analyze_influential_citations(
        self,
        enriched_paper: EnrichedPaper,
        use_cache: bool = True
    ):
        """Analyze influential citation patterns for the paper."""
        if not enriched_paper.semantic_scholar_id:
            return
        
        try:
            # Get influential citations
            influential_citations = await self.semantic_scholar_client.get_influential_citations(
                enriched_paper.semantic_scholar_id,
                limit=100,
                use_cache=use_cache
            )
            
            # Get total citations for ratio calculation
            total_citations = 0
            if enriched_paper.semantic_scholar_data:
                total_citations = enriched_paper.semantic_scholar_data.citation_count or 0
            
            influence_analysis = {
                "total_citations": total_citations,
                "influential_citations": len(influential_citations),
                "influence_ratio": 0.0,
                "temporal_influence": {},
                "intent_influence_correlation": {}
            }
            
            if total_citations > 0:
                influence_analysis["influence_ratio"] = len(influential_citations) / total_citations
            
            # Analyze temporal patterns of influential citations
            citation_years = [cite.citation_year for cite in influential_citations if cite.citation_year]
            if citation_years:
                year_counts = Counter(citation_years)
                influence_analysis["temporal_influence"] = dict(year_counts)
            
            # Analyze correlation between intents and influence
            intent_influence = defaultdict(list)
            for citation in influential_citations:
                if citation.intents:
                    for intent in citation.intents:
                        intent_influence[intent.value].append(citation.is_influential)
            
            # Calculate influence rates by intent
            for intent, influences in intent_influence.items():
                if influences:
                    influence_rate = sum(influences) / len(influences)
                    influence_analysis["intent_influence_correlation"][intent] = {
                        "total_citations": len(influences),
                        "influence_rate": influence_rate
                    }
            
            enriched_paper.influential_citation_analysis = influence_analysis
            
        except Exception as e:
            logger.warning(f"Influential citation analysis failed: {e}")
    
    async def _compute_semantic_similarities(
        self,
        enriched_paper: EnrichedPaper,
        use_cache: bool = True
    ):
        """Compute semantic similarities to related papers."""
        if not enriched_paper.semantic_scholar_id:
            return
        
        try:
            # Get paper embedding
            embedding = await self.semantic_scholar_client.get_paper_embedding(
                enriched_paper.semantic_scholar_id,
                use_cache=use_cache
            )
            
            if not embedding:
                return
            
            # Get some related papers (citations and references) for similarity comparison
            citations = await self.semantic_scholar_client.get_paper_citations(
                enriched_paper.semantic_scholar_id,
                limit=20,
                use_cache=use_cache
            )
            
            references = await self.semantic_scholar_client.get_paper_references(
                enriched_paper.semantic_scholar_id,
                limit=20,
                use_cache=use_cache
            )
            
            related_papers = citations + references
            related_paper_ids = [paper.paper_id for paper in related_papers if paper.paper_id]
            
            # Compute similarities
            similarity_results = await self.semantic_scholar_client.find_similar_papers(
                enriched_paper.semantic_scholar_id,
                related_paper_ids,
                similarity_threshold=0.3,
                use_cache=use_cache
            )
            
            # Convert to dictionary for storage
            similarity_scores = {}
            for result in similarity_results:
                similarity_scores[result.paper_id_2] = result.similarity_score
            
            enriched_paper.semantic_similarity_scores = similarity_scores
            
        except Exception as e:
            logger.warning(f"Semantic similarity computation failed: {e}")
    
    async def analyze_research_trajectory(
        self,
        author_papers: List[str],
        time_window_years: int = 5,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Analyze research trajectory using semantic embeddings and citation patterns.
        
        Args:
            author_papers: List of paper identifiers for an author
            time_window_years: Time window for trend analysis
            use_cache: Whether to use caching
            
        Returns:
            Research trajectory analysis results
        """
        await self._ensure_clients()
        
        if len(author_papers) < 3:
            raise ValidationError("Need at least 3 papers for trajectory analysis")
        
        # Get papers with full metadata
        papers = []
        for paper_id in author_papers:
            paper = await self.semantic_scholar_client.get_paper_by_id(
                paper_id,
                use_cache=use_cache
            )
            if paper and paper.year:
                papers.append(paper)
        
        # Sort papers by publication year
        papers.sort(key=lambda p: p.year or 0)
        
        # Extract embeddings and temporal data
        embeddings = []
        years = []
        citations_over_time = []
        
        for paper in papers:
            embedding = await self.semantic_scholar_client.get_paper_embedding(
                paper.paper_id,
                use_cache=use_cache
            )
            
            if embedding:
                embeddings.append(embedding.vector)
                years.append(paper.year)
                citations_over_time.append(paper.citation_count or 0)
        
        if len(embeddings) < 3:
            raise ValidationError("Need at least 3 papers with embeddings for trajectory analysis")
        
        # Compute trajectory metrics
        trajectory_analysis = {
            "paper_count": len(papers),
            "time_span": max(years) - min(years) if years else 0,
            "semantic_evolution": self._analyze_semantic_evolution(embeddings, years),
            "citation_trajectory": self._analyze_citation_trajectory(citations_over_time, years),
            "research_focus_shifts": await self._identify_research_focus_shifts(papers),
            "collaboration_patterns": self._analyze_collaboration_patterns(papers),
            "venue_diversity": self._analyze_venue_diversity(papers)
        }
        
        return trajectory_analysis
    
    def _analyze_semantic_evolution(
        self,
        embeddings: List[List[float]],
        years: List[int]
    ) -> Dict[str, Any]:
        """Analyze how research focus evolves semantically over time."""
        if len(embeddings) < 3:
            return {}
        
        embeddings_array = np.array(embeddings)
        
        # Compute pairwise similarities
        similarities = cosine_similarity(embeddings_array)
        
        # Analyze temporal similarity patterns
        temporal_similarities = []
        for i in range(1, len(embeddings)):
            similarity = similarities[i-1][i]  # Similarity to previous paper
            temporal_similarities.append(similarity)
        
        # Detect shifts (low similarity with previous work)
        shift_threshold = 0.7
        potential_shifts = [
            {"year": years[i+1], "similarity": sim, "is_shift": sim < shift_threshold}
            for i, sim in enumerate(temporal_similarities)
        ]
        
        # Cluster papers to identify research phases
        if len(embeddings) >= 5:
            kmeans = KMeans(n_clusters=min(3, len(embeddings)//2), random_state=42)
            clusters = kmeans.fit_predict(embeddings_array)
            
            # Group papers by cluster and time
            cluster_timeline = defaultdict(list)
            for i, (year, cluster) in enumerate(zip(years, clusters)):
                cluster_timeline[int(cluster)].append({"year": year, "paper_index": i})
        else:
            cluster_timeline = {}
        
        return {
            "avg_temporal_similarity": statistics.mean(temporal_similarities) if temporal_similarities else 0,
            "similarity_variance": statistics.variance(temporal_similarities) if len(temporal_similarities) > 1 else 0,
            "potential_shifts": potential_shifts,
            "research_phases": dict(cluster_timeline)
        }
    
    def _analyze_citation_trajectory(
        self,
        citation_counts: List[int],
        years: List[int]
    ) -> Dict[str, Any]:
        """Analyze citation trajectory over time."""
        if not citation_counts:
            return {}
        
        # Calculate growth metrics
        total_citations = sum(citation_counts)
        avg_citations_per_paper = statistics.mean(citation_counts)
        
        # Calculate citation growth rate
        growth_rates = []
        for i in range(1, len(citation_counts)):
            if citation_counts[i-1] > 0:
                growth_rate = (citation_counts[i] - citation_counts[i-1]) / citation_counts[i-1]
                growth_rates.append(growth_rate)
        
        return {
            "total_citations": total_citations,
            "avg_citations_per_paper": avg_citations_per_paper,
            "max_citations": max(citation_counts),
            "citation_growth_rates": growth_rates,
            "avg_growth_rate": statistics.mean(growth_rates) if growth_rates else 0,
            "citation_timeline": list(zip(years, citation_counts))
        }
    
    async def _identify_research_focus_shifts(
        self,
        papers: List[SemanticScholarPaper]
    ) -> Dict[str, Any]:
        """Identify major shifts in research focus based on fields of study."""
        if not papers:
            return {}
        
        # Extract fields of study over time
        field_timeline = []
        for paper in papers:
            fields = []
            if paper.s2_fields_of_study:
                fields.extend([field.get("category", "") for field in paper.s2_fields_of_study 
                              if isinstance(field, dict)])
            if paper.fields_of_study:
                fields.extend(paper.fields_of_study)
            
            field_timeline.append({
                "year": paper.year,
                "fields": list(set(fields)),  # Remove duplicates
                "primary_field": paper.get_primary_field()
            })
        
        # Identify field transitions
        transitions = []
        for i in range(1, len(field_timeline)):
            prev_fields = set(field_timeline[i-1]["fields"])
            curr_fields = set(field_timeline[i]["fields"])
            
            # Calculate field overlap
            overlap = len(prev_fields.intersection(curr_fields))
            total_fields = len(prev_fields.union(curr_fields))
            
            if total_fields > 0:
                overlap_ratio = overlap / total_fields
                
                transitions.append({
                    "from_year": field_timeline[i-1]["year"],
                    "to_year": field_timeline[i]["year"],
                    "field_overlap": overlap_ratio,
                    "new_fields": list(curr_fields - prev_fields),
                    "dropped_fields": list(prev_fields - curr_fields)
                })
        
        return {
            "field_timeline": field_timeline,
            "field_transitions": transitions,
            "field_diversity_over_time": [len(entry["fields"]) for entry in field_timeline]
        }
    
    def _analyze_collaboration_patterns(
        self,
        papers: List[SemanticScholarPaper]
    ) -> Dict[str, Any]:
        """Analyze collaboration patterns over time."""
        collaboration_data = []
        all_collaborators = set()
        
        for paper in papers:
            authors = paper.get_author_names()
            collaborators = set(authors[1:]) if len(authors) > 1 else set()  # Exclude main author
            
            collaboration_data.append({
                "year": paper.year,
                "total_authors": len(authors),
                "collaborators": list(collaborators)
            })
            
            all_collaborators.update(collaborators)
        
        # Calculate collaboration metrics
        collaboration_counts = [len(entry["collaborators"]) for entry in collaboration_data]
        author_counts = [entry["total_authors"] for entry in collaboration_data]
        
        return {
            "total_unique_collaborators": len(all_collaborators),
            "avg_collaborators_per_paper": statistics.mean(collaboration_counts) if collaboration_counts else 0,
            "avg_authors_per_paper": statistics.mean(author_counts) if author_counts else 0,
            "collaboration_timeline": collaboration_data,
            "collaboration_growth": collaboration_counts
        }
    
    def _analyze_venue_diversity(
        self,
        papers: List[SemanticScholarPaper]
    ) -> Dict[str, Any]:
        """Analyze publication venue diversity."""
        venues = []
        venue_types = []
        
        for paper in papers:
            if paper.venue:
                venues.append(paper.venue.name or "Unknown")
                venue_types.append(paper.venue.type or "Unknown")
        
        venue_counts = Counter(venues)
        venue_type_counts = Counter(venue_types)
        
        return {
            "total_venues": len(set(venues)),
            "venue_distribution": dict(venue_counts),
            "venue_type_distribution": dict(venue_type_counts),
            "venue_diversity_score": len(set(venues)) / len(papers) if papers else 0
        }
    
    async def identify_emerging_research_trends(
        self,
        field_of_study: str,
        time_window_months: int = 12,
        min_papers: int = 10,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Identify emerging research trends in a specific field using semantic analysis.
        
        Args:
            field_of_study: Research field to analyze
            time_window_months: Time window for trend detection
            min_papers: Minimum papers required for trend identification
            use_cache: Whether to use caching
            
        Returns:
            Emerging trends analysis
        """
        await self._ensure_clients()
        
        # Search for recent papers in the field
        end_date = datetime.now()
        start_date = end_date - timedelta(days=time_window_months * 30)
        
        filters = {
            "fieldsOfStudy": [field_of_study],
            "publicationDateOrYear": f"{start_date.year}-{end_date.year}"
        }
        
        # Get recent papers
        recent_papers = []
        offset = 0
        limit = 100
        
        while len(recent_papers) < min_papers * 5:  # Get more papers for better analysis
            try:
                results = await self.semantic_scholar_client.search_papers(
                    query=f"field:{field_of_study}",
                    filters=filters,
                    limit=limit,
                    offset=offset,
                    use_cache=use_cache
                )
                
                if not results.data:
                    break
                
                recent_papers.extend(results.data)
                offset += limit
                
                if offset >= 500:  # Limit total papers analyzed
                    break
                    
            except Exception as e:
                logger.warning(f"Error searching papers for trend analysis: {e}")
                break
        
        if len(recent_papers) < min_papers:
            raise ValidationError(f"Insufficient papers found for trend analysis. Found {len(recent_papers)}, need {min_papers}")
        
        # Analyze trends
        trend_analysis = {
            "field": field_of_study,
            "analysis_period": {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "papers_analyzed": len(recent_papers)
            },
            "publication_trends": self._analyze_publication_trends(recent_papers),
            "citation_trends": self._analyze_recent_citation_trends(recent_papers),
            "collaboration_trends": self._analyze_collaboration_trends(recent_papers),
            "venue_trends": self._analyze_venue_trends(recent_papers),
            "semantic_clusters": await self._identify_semantic_clusters(recent_papers, use_cache)
        }
        
        return trend_analysis
    
    def _analyze_publication_trends(
        self,
        papers: List[SemanticScholarPaper]
    ) -> Dict[str, Any]:
        """Analyze publication volume trends."""
        # Group papers by month
        monthly_counts = defaultdict(int)
        
        for paper in papers:
            if paper.year and paper.publication_date:
                month_key = f"{paper.year}-{paper.publication_date.month:02d}"
                monthly_counts[month_key] += 1
        
        # Calculate growth rate
        sorted_months = sorted(monthly_counts.keys())
        growth_rates = []
        
        for i in range(1, len(sorted_months)):
            prev_count = monthly_counts[sorted_months[i-1]]
            curr_count = monthly_counts[sorted_months[i]]
            
            if prev_count > 0:
                growth_rate = (curr_count - prev_count) / prev_count
                growth_rates.append(growth_rate)
        
        return {
            "monthly_publication_counts": dict(monthly_counts),
            "total_papers": len(papers),
            "avg_monthly_growth_rate": statistics.mean(growth_rates) if growth_rates else 0,
            "publication_acceleration": len(growth_rates) > 0 and growth_rates[-1] > statistics.mean(growth_rates)
        }
    
    def _analyze_recent_citation_trends(
        self,
        papers: List[SemanticScholarPaper]
    ) -> Dict[str, Any]:
        """Analyze citation patterns in recent papers."""
        citation_counts = [paper.citation_count or 0 for paper in papers if paper.citation_count is not None]
        influential_counts = [paper.influential_citation_count or 0 for paper in papers if paper.influential_citation_count is not None]
        
        if not citation_counts:
            return {}
        
        # Calculate influence ratios
        influence_ratios = []
        for i, paper in enumerate(papers):
            if paper.citation_count and paper.citation_count > 0:
                ratio = (paper.influential_citation_count or 0) / paper.citation_count
                influence_ratios.append(ratio)
        
        return {
            "avg_citations_per_paper": statistics.mean(citation_counts),
            "median_citations": statistics.median(citation_counts),
            "highly_cited_papers": sum(1 for count in citation_counts if count > 50),
            "avg_influence_ratio": statistics.mean(influence_ratios) if influence_ratios else 0,
            "total_citations": sum(citation_counts),
            "total_influential_citations": sum(influential_counts)
        }
    
    def _analyze_collaboration_trends(
        self,
        papers: List[SemanticScholarPaper]
    ) -> Dict[str, Any]:
        """Analyze collaboration trends in recent research."""
        author_counts = []
        institution_diversity = []
        
        for paper in papers:
            authors = paper.get_author_names()
            author_counts.append(len(authors))
            
            # Analyze institution diversity (if available)
            if hasattr(paper, 'authors') and paper.authors:
                institutions = set()
                for author in paper.authors:
                    if hasattr(author, 'affiliations') and author.affiliations:
                        institutions.update(author.affiliations)
                institution_diversity.append(len(institutions))
        
        return {
            "avg_authors_per_paper": statistics.mean(author_counts) if author_counts else 0,
            "median_authors_per_paper": statistics.median(author_counts) if author_counts else 0,
            "single_author_papers": sum(1 for count in author_counts if count == 1),
            "highly_collaborative_papers": sum(1 for count in author_counts if count > 5),
            "avg_institution_diversity": statistics.mean(institution_diversity) if institution_diversity else 0
        }
    
    def _analyze_venue_trends(
        self,
        papers: List[SemanticScholarPaper]
    ) -> Dict[str, Any]:
        """Analyze venue publication trends."""
        venue_counts = Counter()
        venue_types = Counter()
        
        for paper in papers:
            if paper.venue and paper.venue.name:
                venue_counts[paper.venue.name] += 1
                if paper.venue.type:
                    venue_types[paper.venue.type] += 1
        
        return {
            "top_venues": dict(venue_counts.most_common(10)),
            "venue_type_distribution": dict(venue_types),
            "venue_diversity": len(venue_counts),
            "top_venue_concentration": venue_counts.most_common(5)[0][1] / len(papers) if venue_counts else 0
        }
    
    async def _identify_semantic_clusters(
        self,
        papers: List[SemanticScholarPaper],
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """Identify semantic clusters in recent research."""
        # Get embeddings for papers
        embeddings = []
        paper_metadata = []
        
        for paper in papers:
            embedding = await self.semantic_scholar_client.get_paper_embedding(
                paper.paper_id,
                use_cache=use_cache
            )
            
            if embedding:
                embeddings.append(embedding.vector)
                paper_metadata.append({
                    "paper_id": paper.paper_id,
                    "title": paper.title,
                    "year": paper.year,
                    "citation_count": paper.citation_count or 0
                })
        
        if len(embeddings) < 10:
            return {"error": "Insufficient papers with embeddings for clustering"}
        
        # Perform clustering
        n_clusters = min(8, len(embeddings) // 5)  # Adaptive cluster count
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(embeddings)
        
        # Analyze clusters
        clusters = defaultdict(list)
        for i, label in enumerate(cluster_labels):
            clusters[int(label)].append(paper_metadata[i])
        
        # Characterize each cluster
        cluster_analysis = {}
        for cluster_id, cluster_papers in clusters.items():
            cluster_analysis[f"cluster_{cluster_id}"] = {
                "paper_count": len(cluster_papers),
                "avg_citations": statistics.mean([p["citation_count"] for p in cluster_papers]),
                "year_range": {
                    "min": min([p["year"] for p in cluster_papers if p["year"]]),
                    "max": max([p["year"] for p in cluster_papers if p["year"]])
                },
                "top_papers": sorted(cluster_papers, key=lambda x: x["citation_count"], reverse=True)[:5]
            }
        
        return {
            "total_clusters": n_clusters,
            "clustered_papers": len(embeddings),
            "cluster_analysis": cluster_analysis
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get service performance metrics."""
        metrics = {
            "clients_initialized": self._clients_initialized
        }
        
        if self.semantic_scholar_client:
            metrics["semantic_scholar"] = self.semantic_scholar_client.get_metrics()
        
        if self.openalex_client:
            metrics["openalex"] = self.openalex_client.get_metrics()
        
        return metrics


# Global service instance
_semantic_analysis_service: Optional[SemanticAnalysisService] = None


async def get_semantic_analysis_service() -> SemanticAnalysisService:
    """Get or create semantic analysis service instance."""
    global _semantic_analysis_service
    
    if _semantic_analysis_service is None:
        _semantic_analysis_service = SemanticAnalysisService()
    
    return _semantic_analysis_service