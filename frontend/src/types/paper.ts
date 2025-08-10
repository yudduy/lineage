export interface Author {
  name: string;
  orcid?: string;
  affiliation?: string;
  email?: string;
}

export interface Journal {
  name: string;
  issn?: string;
  publisher?: string;
  impact_factor?: number;
}

export interface CitationCount {
  total: number;
  crossref?: number;
  semantic_scholar?: number;
  openalex?: number;
  last_updated?: string;
}

export interface Paper {
  // Core identifiers
  id?: string;
  doi?: string;
  pmid?: string;
  arxiv_id?: string;
  openalex_id?: string;
  semantic_scholar_id?: string;
  
  // Basic metadata
  title: string;
  abstract?: string;
  authors: Author[];
  
  // Publication details
  journal?: Journal;
  publication_date?: string;
  publication_year?: number;
  volume?: string;
  issue?: string;
  pages?: string;
  
  // URLs and links
  url?: string;
  pdf_url?: string;
  open_access_url?: string;
  
  // Citation information
  citation_count: CitationCount;
  references: string[];
  cited_by: string[];
  
  // Classification and topics
  subjects: string[];
  keywords: string[];
  concepts: Array<{ [key: string]: any }>;
  
  // Metadata
  language?: string;
  paper_type?: string;
  is_open_access: boolean;
  
  // System metadata
  created_at?: string;
  updated_at?: string;

  // Legacy compatibility and visualization properties
  microsoftID?: string;
  author?: string; // For backward compatibility
  seed?: boolean;
  ID?: number; // Legacy ID for visualization
  hide?: boolean;
  
  // Computed metrics (for visualization)
  localCitedBy?: number;
  localReferences?: number;
  seedsCitedBy?: number;
  seedsCited?: number;
}

export interface PaperEdge {
  source_id: string;
  target_id: string;
  edge_type: 'cites' | 'cited_by';
  weight: number;
  context?: string;
  created_at?: string;
  
  // For visualization compatibility
  source?: Paper | string | number;
  target?: Paper | string | number;
}

// Force graph node with position data
export interface GraphNode extends Paper {
  x?: number;
  y?: number;
  z?: number;
  fx?: number;
  fy?: number;
  fz?: number;
}

// Force graph link with resolved references
export interface GraphLink extends PaperEdge {
  source: string | number | GraphNode;
  target: string | number | GraphNode;
}

export interface PaperSearchRequest {
  query?: string;
  title?: string;
  authors?: string;
  journal?: string;
  doi?: string;
  dois?: string[];
  publication_year_min?: number;
  publication_year_max?: number;
  citation_count_min?: number;
  is_open_access?: boolean;
  sort_by?: string;
  sort_order?: 'asc' | 'desc';
  limit?: number;
  offset?: number;
}

export interface PaperEdgesRequest {
  paper_id: string;
  direction?: 'cited_by' | 'references' | 'both';
  max_depth?: number;
  include_metadata?: boolean;
  limit?: number;
  offset?: number;
}

export interface PaperResponse {
  paper: Paper;
  edges?: PaperEdge[];
  related_papers?: Paper[];
}

export interface PaperEdgesResponse {
  center_paper: Paper;
  nodes: Paper[];
  edges: PaperEdge[];
  total_nodes: number;
  total_edges: number;
  depth: number;
}

export interface PaperBulkRequest {
  paper_ids: string[];
  operation: string;
  parameters?: { [key: string]: any };
}

// Graph visualization types
export interface GraphConfig {
  mode: 'references' | 'citations';
  minConnections: number;
  sizeMetric: 'localCitedBy' | 'localReferences' | 'seedsCitedBy' | 'seedsCited';
  selectedNode?: Paper;
  threshold: number;
  showLabels: boolean;
  colorScheme: 'default' | 'year' | 'citations' | 'community' | 'impact';
  layout: '2d' | '3d';
  clustering: boolean;
  animation: boolean;
  pathHighlight: boolean;
  communityDetection: boolean;
  showCitationFlow: boolean;
}

export interface GraphData {
  nodes: GraphNode[];
  links: GraphLink[];
  allNodes: GraphNode[];
  allEdges: GraphLink[];
  communities?: Community[];
  pathData?: PathData[];
}

export interface ViewMode {
  type: 'network' | 'list' | 'table' | 'timeline' | 'dashboard';
  active: boolean;
}

// Advanced Graph Features
export interface Community {
  id: string;
  name: string;
  color: string;
  nodes: string[];
  size: number;
  cohesion: number;
  centrality: number;
}

export interface PathData {
  id: string;
  source: string;
  target: string;
  path: string[];
  weight: number;
  type: 'citation' | 'influence' | 'collaboration';
}

export interface CitationFlow {
  sourceId: string;
  targetId: string;
  intensity: number;
  timestamp: number;
  context: string;
}

// Research Intelligence Types
export interface ResearchIntelligence {
  id: string;
  paperId: string;
  summary: string;
  keyInsights: string[];
  impactPrediction: number;
  trendAnalysis: TrendAnalysis;
  recommendations: string[];
  generatedAt: string;
}

export interface TrendAnalysis {
  momentum: number;
  growth_rate: number;
  peak_year?: number;
  declining: boolean;
  emerging_topics: string[];
}

// Export Types
export interface ExportOptions {
  format: 'png' | 'svg' | 'json' | 'graphml' | 'csv' | 'bibtex';
  quality?: 'low' | 'medium' | 'high';
  includeMetadata?: boolean;
  selectedOnly?: boolean;
  transparent?: boolean;
}

// Modal Types
export interface ModalState {
  isOpen: boolean;
  type: 'doi' | 'search' | 'bibtex' | 'zotero' | 'export' | 'details' | 'help';
  data?: any;
}

// WebSocket Types
export interface WebSocketMessage {
  type: 'progress' | 'update' | 'notification' | 'collaboration';
  data: any;
  timestamp: number;
}

// Advanced Search Types
export interface SearchFilters {
  yearRange?: [number, number];
  citationRange?: [number, number];
  authors?: string[];
  journals?: string[];
  subjects?: string[];
  openAccess?: boolean;
  hasFullText?: boolean;
}