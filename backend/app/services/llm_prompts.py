"""
LLM Prompt Templates - Comprehensive templates for research paper analysis and enrichment.

This module contains carefully crafted prompt templates for:
- Paper summarization and key contribution extraction
- Citation relationship analysis and context explanation
- Research trajectory and intellectual lineage analysis
- Community theme identification and trend analysis
"""

from typing import Dict, List, Any
from enum import Enum

from .llm_service import PromptTemplate, ModelType


class PromptType(Enum):
    """Types of prompts for different tasks."""
    PAPER_SUMMARY = "paper_summary"
    CITATION_ANALYSIS = "citation_analysis"
    RESEARCH_TRAJECTORY = "research_trajectory"
    THEME_IDENTIFICATION = "theme_identification"
    COMPARATIVE_ANALYSIS = "comparative_analysis"
    TIMELINE_NARRATIVE = "timeline_narrative"


class LLMPrompts:
    """Collection of prompt templates for research analysis tasks."""
    
    @staticmethod
    def get_paper_summary_template() -> PromptTemplate:
        """Template for paper summarization and key contribution extraction."""
        return PromptTemplate(
            system_prompt="""You are an expert research analyst specializing in academic paper analysis. 
Your task is to provide comprehensive, accurate summaries that capture the essence of research papers.

Focus on:
1. Core contributions and novelty
2. Methodology and approach
3. Key findings and results
4. Significance and impact
5. Limitations and future work

Provide structured, concise analysis that would be valuable to researchers in the field.""",
            
            user_template="""Analyze the following research paper and provide a comprehensive summary:

**Paper Title:** {title}

**Abstract:** {abstract}

**Publication Year:** {year}

**Authors:** {authors}

**Venue:** {venue}

{additional_content}

Please provide a structured analysis with the following sections:

## Summary
A 2-3 sentence overview of what this paper is about and its main contribution.

## Key Contributions
- List the primary contributions (3-5 bullet points)
- Highlight what makes this work novel or significant

## Methodology
Brief description of the approach, methods, or techniques used.

## Key Findings
Main results, discoveries, or insights from the research.

## Impact and Significance
Why this work matters to the field and its potential influence.

## Limitations
Any acknowledged limitations or potential weaknesses.

## Future Directions
Suggested future work or research directions mentioned by the authors.

Keep each section concise but informative. Focus on accuracy and clarity.""",
            
            max_tokens=2000,
            temperature=0.1,
            model_type=ModelType.ANALYSIS
        )
    
    @staticmethod
    def get_citation_analysis_template() -> PromptTemplate:
        """Template for analyzing citation relationships and context."""
        return PromptTemplate(
            system_prompt="""You are an expert in citation analysis and research impact assessment. 
Your role is to analyze why one paper cites another and explain the intellectual relationship between works.

Focus on:
1. The specific reason for citation (methodology, results, background, criticism, etc.)
2. How the cited work contributes to the citing work
3. The nature of the intellectual relationship
4. The citation's significance in the research narrative

Be precise and analytical in your assessment.""",
            
            user_template="""Analyze the citation relationship between these two papers:

**CITING PAPER:**
Title: {citing_title}
Authors: {citing_authors}
Year: {citing_year}
Abstract: {citing_abstract}

**CITED PAPER:**
Title: {cited_title}
Authors: {cited_authors}
Year: {cited_year}
Abstract: {cited_abstract}

**Citation Context (if available):**
{citation_context}

**Additional Information:**
- Citation Intent: {citation_intent}
- Is Influential Citation: {is_influential}

Please analyze this citation relationship and provide:

## Citation Purpose
Why does the citing paper reference the cited paper? What specific role does it play?

## Intellectual Relationship
How do these works relate intellectually? (builds upon, challenges, uses methodology, compares results, etc.)

## Knowledge Flow
What specific knowledge, method, or insight flows from the cited paper to the citing paper?

## Impact Assessment
How significant is this citation for understanding the citing paper's contribution?

## Citation Type
Classify this citation as one of: Background, Methodology, Comparison, Extension, Criticism, Supporting Evidence

Provide a clear, analytical assessment focusing on the intellectual connections.""",
            
            max_tokens=1500,
            temperature=0.1,
            model_type=ModelType.ANALYSIS
        )
    
    @staticmethod
    def get_research_trajectory_template() -> PromptTemplate:
        """Template for analyzing research trajectory and evolution."""
        return PromptTemplate(
            system_prompt="""You are an expert in research trajectory analysis and intellectual history. 
Your task is to trace how research ideas evolve over time through citation networks.

Focus on:
1. How ideas develop and transform across papers
2. Key milestones and breakthrough moments
3. Methodological evolution and refinement
4. Shifting research questions and focus areas
5. The narrative of intellectual development

Provide insightful analysis that reveals the deeper patterns in research evolution.""",
            
            user_template="""Analyze the research trajectory represented by these papers in chronological order:

{papers_data}

**Research Field:** {field}
**Time Span:** {time_span}
**Number of Papers:** {num_papers}

Please provide a comprehensive trajectory analysis:

## Intellectual Evolution
How have the core ideas, questions, or approaches evolved from the earliest to most recent papers?

## Key Milestones
Identify 3-5 critical papers that represent major advances or shifts in the trajectory.

## Methodological Development
How have methods, techniques, or approaches been refined or changed over time?

## Research Focus Shifts
What changes in research questions, problem focus, or scope can you identify?

## Innovation Patterns
What patterns of innovation do you see? (incremental improvement, breakthrough moments, paradigm shifts)

## Future Trajectory
Based on this evolution, what directions might this research take next?

## Significance Assessment
What makes this research trajectory important for the broader field?

Provide a narrative that captures the intellectual journey and its significance.""",
            
            max_tokens=2500,
            temperature=0.2,
            model_type=ModelType.ANALYSIS
        )
    
    @staticmethod
    def get_theme_identification_template() -> PromptTemplate:
        """Template for identifying research themes and community analysis."""
        return PromptTemplate(
            system_prompt="""You are an expert in research community analysis and thematic identification. 
Your role is to identify major research themes, trends, and community patterns from collections of papers.

Focus on:
1. Identifying coherent research themes and sub-themes
2. Understanding community structure and collaborations
3. Recognizing emerging trends and hot topics
4. Mapping interdisciplinary connections
5. Assessing the health and direction of research areas

Provide structured analysis that reveals community-level insights.""",
            
            user_template="""Analyze the research themes present in this collection of papers:

**Papers Overview:**
- Total Papers: {num_papers}
- Time Range: {time_range}
- Primary Field: {primary_field}
- Key Venues: {key_venues}

**Paper Sample:** {papers_sample}

**Citation Network Statistics:**
- Total Citations: {total_citations}
- Average Citations per Paper: {avg_citations}
- Most Cited Papers: {top_cited}

**Author Information:**
- Total Authors: {total_authors}
- Collaboration Patterns: {collaboration_info}

Please provide a comprehensive thematic analysis:

## Major Research Themes
Identify 4-6 major themes with brief descriptions of each.

## Emerging Trends
What new or growing research directions can you identify?

## Community Structure
What collaboration patterns or research groups are evident?

## Methodological Approaches
What are the dominant methods or approaches in this community?

## Interdisciplinary Connections
What connections to other fields or disciplines are apparent?

## Research Maturity
Is this a mature field or an emerging area? What evidence supports this assessment?

## Knowledge Gaps
What important questions or areas seem underexplored?

## Impact Assessment
Which themes appear to have the most impact or citation activity?

Provide insights that would help researchers understand this community's landscape.""",
            
            max_tokens=2000,
            temperature=0.2,
            model_type=ModelType.ANALYSIS
        )
    
    @staticmethod
    def get_comparative_analysis_template() -> PromptTemplate:
        """Template for comparative analysis between papers or research approaches."""
        return PromptTemplate(
            system_prompt="""You are an expert in comparative research analysis. Your role is to systematically 
compare research papers, identifying similarities, differences, and relationships between approaches, findings, and contributions.

Focus on:
1. Methodological comparisons and trade-offs
2. Complementary vs. competing approaches
3. Incremental vs. fundamental differences
4. Strengths and limitations of each approach
5. Synthesis opportunities and future directions

Provide balanced, analytical comparisons that highlight key insights.""",
            
            user_template="""Compare and analyze the following research papers:

{papers_for_comparison}

**Comparison Context:**
- Research Question Focus: {research_focus}
- Time Period: {time_period}
- Methodological Scope: {method_scope}

Please provide a systematic comparison:

## Research Questions & Objectives
How do the research questions and objectives compare across papers?

## Methodological Approaches
Compare the methods, techniques, and experimental designs used.

## Key Findings Comparison
How do the main results and findings relate to each other?

## Novelty & Contributions
What are the unique contributions of each work?

## Strengths & Limitations
Comparative assessment of strengths and weaknesses.

## Complementary Insights
How do these works complement each other?

## Conflicting Results
Are there any contradictory findings that need explanation?

## Synthesis Opportunities
How might insights from these works be combined or integrated?

## Research Impact
Comparative assessment of each work's impact and influence.

Provide analytical insights that reveal the relationships and relative contributions.""",
            
            max_tokens=2000,
            temperature=0.1,
            model_type=ModelType.ANALYSIS
        )
    
    @staticmethod
    def get_timeline_narrative_template() -> PromptTemplate:
        """Template for generating timeline narratives of research evolution."""
        return PromptTemplate(
            system_prompt="""You are a science historian and expert storyteller specialized in research narratives. 
Your task is to create compelling, accurate narratives that tell the story of how research ideas have evolved over time.

Focus on:
1. Creating a coherent narrative arc
2. Highlighting key moments and breakthrough discoveries
3. Showing cause-and-effect relationships between works
4. Emphasizing human elements and scientific drama
5. Making complex research accessible and engaging

Write in an engaging but scientifically accurate style.""",
            
            user_template="""Create a compelling narrative timeline for this research evolution:

**Timeline Data:**
{timeline_data}

**Research Domain:** {domain}
**Key Players:** {key_researchers}
**Time Span:** {timeline_span}
**Major Breakthroughs:** {breakthroughs}

Please create a narrative timeline with:

## The Beginning: Setting the Stage
What were the initial questions, problems, or needs that started this research direction?

## Early Foundations (Years {early_years})
{early_papers_context}

## Growing Momentum (Years {middle_years})
{middle_papers_context}

## Major Breakthroughs (Years {breakthrough_years})
{breakthrough_papers_context}

## Recent Developments (Years {recent_years})
{recent_papers_context}

## The Current Landscape
Where does this research stand today?

## Looking Forward
Based on this trajectory, what might the next chapter hold?

Create an engaging narrative that shows how ideas built upon each other, 
highlighting the human ingenuity and scientific insights that drove progress. 
Make it accessible to both experts and educated general readers.""",
            
            max_tokens=3000,
            temperature=0.3,
            model_type=ModelType.ANALYSIS
        )
    
    @staticmethod
    def get_influence_assessment_template() -> PromptTemplate:
        """Template for assessing research influence and impact."""
        return PromptTemplate(
            system_prompt="""You are an expert in research impact assessment and bibliometric analysis. 
Your role is to evaluate the influence and significance of research works within their scholarly communities.

Focus on:
1. Multiple dimensions of impact (methodological, theoretical, practical)
2. Direct vs. indirect influence patterns
3. Long-term vs. short-term impact
4. Breadth and depth of influence
5. Quality of impact beyond just citation counts

Provide nuanced assessment that goes beyond simple metrics.""",
            
            user_template="""Assess the research influence and impact of this work:

**Paper Details:**
Title: {title}
Authors: {authors}
Year: {year}
Venue: {venue}
Abstract: {abstract}

**Citation Metrics:**
- Total Citations: {total_citations}
- Citations per Year: {citations_per_year}
- Peak Citation Year: {peak_year}
- Recent Citation Trend: {citation_trend}

**Citation Context Analysis:**
- Influential Citations: {influential_citations}
- Citation Intents Distribution: {citation_intents}
- Citing Paper Venues: {citing_venues}
- Citing Paper Fields: {citing_fields}

**Network Position:**
- Papers Citing This Work: {citing_papers_count}
- Papers This Work Cites: {reference_count}
- Co-citation Patterns: {cocitation_info}

Please provide a comprehensive influence assessment:

## Impact Dimensions
Assess impact across different dimensions (theoretical, methodological, practical, interdisciplinary).

## Influence Patterns
How has this work influenced subsequent research? Direct methodological adoption, theoretical extensions, empirical validation?

## Community Reception
How has the research community received and built upon this work?

## Temporal Impact Analysis
How has the influence evolved over time? Immediate vs. long-term recognition?

## Breadth vs. Depth
Is the influence broad (many fields) or deep (transformative in specific area)?

## Quality Indicators
Beyond citation count, what indicates the quality and significance of this impact?

## Comparison to Field Norms
How does this impact compare to typical papers in the same field and time period?

## Future Impact Potential
Based on current trends, how might the influence of this work evolve?

Provide a nuanced assessment that captures the multifaceted nature of research impact.""",
            
            max_tokens=2000,
            temperature=0.1,
            model_type=ModelType.ANALYSIS
        )


class PromptTemplateManager:
    """Manages and provides access to prompt templates."""
    
    def __init__(self):
        self._templates: Dict[PromptType, PromptTemplate] = {
            PromptType.PAPER_SUMMARY: LLMPrompts.get_paper_summary_template(),
            PromptType.CITATION_ANALYSIS: LLMPrompts.get_citation_analysis_template(),
            PromptType.RESEARCH_TRAJECTORY: LLMPrompts.get_research_trajectory_template(),
            PromptType.THEME_IDENTIFICATION: LLMPrompts.get_theme_identification_template(),
            PromptType.COMPARATIVE_ANALYSIS: LLMPrompts.get_comparative_analysis_template(),
            PromptType.TIMELINE_NARRATIVE: LLMPrompts.get_timeline_narrative_template(),
        }
    
    def get_template(self, prompt_type: PromptType) -> PromptTemplate:
        """Get a prompt template by type."""
        if prompt_type not in self._templates:
            raise ValueError(f"Unknown prompt type: {prompt_type}")
        return self._templates[prompt_type]
    
    def get_all_templates(self) -> Dict[PromptType, PromptTemplate]:
        """Get all available templates."""
        return self._templates.copy()
    
    def create_custom_template(
        self,
        system_prompt: str,
        user_template: str,
        max_tokens: int = 2000,
        temperature: float = 0.1,
        model_type: ModelType = ModelType.ANALYSIS
    ) -> PromptTemplate:
        """Create a custom prompt template."""
        return PromptTemplate(
            system_prompt=system_prompt,
            user_template=user_template,
            max_tokens=max_tokens,
            temperature=temperature,
            model_type=model_type
        )


# Global template manager instance
_template_manager: PromptTemplateManager = None


def get_template_manager() -> PromptTemplateManager:
    """Get the global prompt template manager."""
    global _template_manager
    if _template_manager is None:
        _template_manager = PromptTemplateManager()
    return _template_manager