import { create } from 'zustand';
import { devtools, subscribeWithSelector } from 'zustand/middleware';
import { Paper, PaperEdge, GraphNode, GraphLink, GraphConfig } from '@types/paper';

// Paper management state
interface PaperState {
  // Core data
  papers: Paper[];
  edges: PaperEdge[];
  seedPapers: Paper[];
  
  // Graph visualization data
  graphNodes: GraphNode[];
  graphLinks: GraphLink[];
  allNodes: GraphNode[];
  allEdges: GraphLink[];
  
  // Configuration
  graphConfig: GraphConfig;
  
  // UI state
  selectedPapers: Paper[];
  highlightedPaper: Paper | null;
  
  // Actions
  addPapers: (papers: Paper[]) => void;
  updatePaper: (paperId: string, updates: Partial<Paper>) => void;
  removePaper: (paperId: string) => void;
  
  addEdges: (edges: PaperEdge[]) => void;
  removeEdges: (edgeIds: string[]) => void;
  
  makeSeed: (papers: Paper[]) => void;
  removeSeed: (paperId: string) => void;
  
  setSelectedPapers: (papers: Paper[]) => void;
  addSelectedPaper: (paper: Paper) => void;
  removeSelectedPaper: (paperId: string) => void;
  clearSelection: () => void;
  
  setHighlightedPaper: (paper: Paper | null) => void;
  
  updateGraphConfig: (config: Partial<GraphConfig>) => void;
  filterGraph: () => void;
  updateMetrics: () => void;
  
  // Reset functions
  clearAllPapers: () => void;
  clearNonSeedPapers: () => void;
}

// Utility functions
const calculateMetrics = (paper: Paper, papers: Paper[], edges: PaperEdge[]): Partial<Paper> => {
  const localCitedBy = edges.filter(e => e.target_id === paper.id).length;
  const localReferences = edges.filter(e => e.source_id === paper.id).length;
  const seedsCitedBy = edges.filter(e => {
    const sourcePaper = papers.find(p => p.id === e.source_id);
    return sourcePaper?.seed && e.target_id === paper.id;
  }).length;
  const seedsCited = edges.filter(e => {
    const targetPaper = papers.find(p => p.id === e.target_id);
    return targetPaper?.seed && e.source_id === paper.id;
  }).length;

  return {
    localCitedBy,
    localReferences,
    seedsCitedBy,
    seedsCited,
  };
};

const matchPaper = (newPaper: Paper, existingPapers: Paper[]): Paper | null => {
  // Match by various identifiers
  if (newPaper.id) {
    const match = existingPapers.find(p => p.id === newPaper.id);
    if (match) return match;
  }
  
  if (newPaper.doi) {
    const match = existingPapers.find(p => p.doi?.toLowerCase() === newPaper.doi?.toLowerCase());
    if (match) return match;
  }
  
  if (newPaper.title && newPaper.authors.length > 0) {
    const match = existingPapers.find(p => 
      p.title.toLowerCase() === newPaper.title.toLowerCase() &&
      p.authors.length > 0 &&
      p.authors[0].name.toLowerCase() === newPaper.authors[0].name.toLowerCase()
    );
    if (match) return match;
  }
  
  return null;
};

const mergePapers = (existing: Paper, incoming: Paper): Paper => {
  const merged = { ...existing };
  
  // Merge all fields, preferring existing values where they exist
  Object.keys(incoming).forEach(key => {
    const typedKey = key as keyof Paper;
    if (incoming[typedKey] && !existing[typedKey]) {
      (merged as any)[typedKey] = incoming[typedKey];
    }
  });
  
  // Special handling for seed status - preserve if either is true
  merged.seed = existing.seed || incoming.seed || false;
  
  return merged;
};

export const usePaperStore = create<PaperState>()(
  devtools(
    subscribeWithSelector(
      (set, get) => ({
        // Initial state
        papers: [],
        edges: [],
        seedPapers: [],
        graphNodes: [],
        graphLinks: [],
        allNodes: [],
        allEdges: [],
        selectedPapers: [],
        highlightedPaper: null,
        
        graphConfig: {
          mode: 'references',
          minConnections: 0,
          sizeMetric: 'seedsCitedBy',
          selectedNode: undefined,
          threshold: 0,
          showLabels: true,
          colorScheme: 'default',
          layout: '2d',
          clustering: false,
          animation: true,
          pathHighlight: false,
          communityDetection: false,
          showCitationFlow: false,
        },
        
        // Actions
        addPapers: (newPapers: Paper[]) => {
          set(state => {
            const addedPapers: Paper[] = [];
            const updatedPapers = [...state.papers];
            
            newPapers.forEach(paper => {
              const existingPaper = matchPaper(paper, updatedPapers);
              
              if (existingPaper) {
                // Update existing paper
                const index = updatedPapers.findIndex(p => p.id === existingPaper.id);
                updatedPapers[index] = mergePapers(existingPaper, paper);
              } else {
                // Add new paper
                const newId = paper.id || `paper_${Date.now()}_${Math.random()}`;
                const newPaper = {
                  ...paper,
                  id: newId,
                  ID: updatedPapers.length, // Legacy ID for visualization
                };
                updatedPapers.push(newPaper);
                addedPapers.push(newPaper);
              }
            });
            
            // Update seed papers list
            const seedPapers = updatedPapers.filter(p => p.seed);
            
            return {
              papers: updatedPapers,
              seedPapers,
            };
          });
          
          // Update metrics after adding papers
          get().updateMetrics();
        },
        
        updatePaper: (paperId: string, updates: Partial<Paper>) => {
          set(state => ({
            papers: state.papers.map(paper =>
              paper.id === paperId ? { ...paper, ...updates } : paper
            ),
          }));
          
          get().updateMetrics();
        },
        
        removePaper: (paperId: string) => {
          set(state => {
            const updatedPapers = state.papers.filter(p => p.id !== paperId);
            const updatedEdges = state.edges.filter(e => 
              e.source_id !== paperId && e.target_id !== paperId
            );
            
            return {
              papers: updatedPapers,
              edges: updatedEdges,
              seedPapers: updatedPapers.filter(p => p.seed),
              selectedPapers: state.selectedPapers.filter(p => p.id !== paperId),
            };
          });
          
          get().filterGraph();
        },
        
        addEdges: (newEdges: PaperEdge[]) => {
          set(state => {
            const existingEdgeKeys = new Set(
              state.edges.map(e => `${e.source_id}-${e.target_id}`)
            );
            
            const uniqueNewEdges = newEdges.filter(edge =>
              !existingEdgeKeys.has(`${edge.source_id}-${edge.target_id}`)
            );
            
            return {
              edges: [...state.edges, ...uniqueNewEdges],
            };
          });
          
          get().updateMetrics();
          get().filterGraph();
        },
        
        removeEdges: (edgeIds: string[]) => {
          set(state => ({
            edges: state.edges.filter(edge => 
              !edgeIds.includes(`${edge.source_id}-${edge.target_id}`)
            ),
          }));
          
          get().filterGraph();
        },
        
        makeSeed: (papers: Paper[]) => {
          set(state => ({
            papers: state.papers.map(paper => ({
              ...paper,
              seed: papers.some(p => p.id === paper.id) ? true : paper.seed,
            })),
          }));
          
          // Update seed papers list
          set(state => ({
            seedPapers: state.papers.filter(p => p.seed),
          }));
          
          get().updateMetrics();
          get().filterGraph();
        },
        
        removeSeed: (paperId: string) => {
          set(state => {
            // Remove seed status
            const updatedPapers = state.papers.map(paper => ({
              ...paper,
              seed: paper.id === paperId ? false : paper.seed,
            }));
            
            // Remove edges connecting this paper to non-seeds
            const updatedEdges = state.edges.filter(edge => {
              const sourceIsSeed = updatedPapers.find(p => p.id === edge.source_id)?.seed;
              const targetIsSeed = updatedPapers.find(p => p.id === edge.target_id)?.seed;
              
              // Keep edge if both are seeds, or if neither involves the removed seed
              return !(
                (edge.source_id === paperId && !targetIsSeed) ||
                (edge.target_id === paperId && !sourceIsSeed)
              );
            });
            
            // Remove papers that are no longer connected to anything
            const connectedPaperIds = new Set([
              ...updatedEdges.map(e => e.source_id),
              ...updatedEdges.map(e => e.target_id),
            ]);
            
            const finalPapers = updatedPapers.filter(paper =>
              paper.seed || connectedPaperIds.has(paper.id!)
            );
            
            return {
              papers: finalPapers,
              edges: updatedEdges,
              seedPapers: finalPapers.filter(p => p.seed),
            };
          });
          
          get().updateMetrics();
          get().filterGraph();
        },
        
        setSelectedPapers: (papers: Paper[]) => {
          set({ selectedPapers: papers });
        },
        
        addSelectedPaper: (paper: Paper) => {
          set(state => ({
            selectedPapers: [...state.selectedPapers, paper],
          }));
        },
        
        removeSelectedPaper: (paperId: string) => {
          set(state => ({
            selectedPapers: state.selectedPapers.filter(p => p.id !== paperId),
          }));
        },
        
        clearSelection: () => {
          set({ selectedPapers: [] });
        },
        
        setHighlightedPaper: (paper: Paper | null) => {
          set({ highlightedPaper: paper });
        },
        
        updateGraphConfig: (config: Partial<GraphConfig>) => {
          set(state => ({
            graphConfig: { ...state.graphConfig, ...config },
          }));
          
          get().filterGraph();
        },
        
        filterGraph: () => {
          set(state => {
            const { graphConfig, papers, edges } = state;
            const { mode, threshold } = graphConfig;
            
            let filteredEdges: PaperEdge[] = [];
            
            switch (mode) {
              case 'references':
                filteredEdges = edges.filter(edge => {
                  const sourcePaper = papers.find(p => p.id === edge.source_id);
                  return sourcePaper?.seed;
                });
                break;
              case 'citations':
                filteredEdges = edges.filter(edge => {
                  const targetPaper = papers.find(p => p.id === edge.target_id);
                  return targetPaper?.seed;
                });
                break;
            }
            
            // Get connected paper IDs
            const connectedPaperIds = new Set([
              ...filteredEdges.map(e => e.source_id),
              ...filteredEdges.map(e => e.target_id),
            ]);
            
            // Filter nodes (include seeds and connected papers above threshold)
            const sizeMetric = mode === 'references' ? 'seedsCitedBy' : 'seedsCited';
            const filteredNodes = papers.filter(paper => {
              if (paper.seed) return true;
              if (!connectedPaperIds.has(paper.id!)) return false;
              const metricValue = (paper as any)[sizeMetric] || 0;
              return metricValue >= threshold;
            });
            
            // Convert to graph format
            const graphNodes: GraphNode[] = filteredNodes.map(paper => ({
              ...paper,
              hide: false,
            }));
            
            const graphLinks: GraphLink[] = filteredEdges
              .filter(edge => {
                const sourceExists = filteredNodes.some(n => n.id === edge.source_id);
                const targetExists = filteredNodes.some(n => n.id === edge.target_id);
                return sourceExists && targetExists;
              })
              .map(edge => ({
                ...edge,
                source: edge.source_id,
                target: edge.target_id,
              }));
            
            return {
              graphNodes,
              graphLinks,
              allNodes: papers.map(p => ({ ...p, hide: false })),
              allEdges: edges.map(e => ({ ...e, source: e.source_id, target: e.target_id })),
            };
          });
        },
        
        updateMetrics: () => {
          set(state => {
            const updatedPapers = state.papers.map(paper => ({
              ...paper,
              ...calculateMetrics(paper, state.papers, state.edges),
            }));
            
            return {
              papers: updatedPapers,
              seedPapers: updatedPapers.filter(p => p.seed),
            };
          });
        },
        
        clearAllPapers: () => {
          set({
            papers: [],
            edges: [],
            seedPapers: [],
            graphNodes: [],
            graphLinks: [],
            allNodes: [],
            allEdges: [],
            selectedPapers: [],
            highlightedPaper: null,
          });
        },
        
        clearNonSeedPapers: () => {
          set(state => {
            const seedPapers = state.papers.filter(p => p.seed);
            const seedEdges = state.edges.filter(edge => {
              const sourceIsSeed = seedPapers.some(p => p.id === edge.source_id);
              const targetIsSeed = seedPapers.some(p => p.id === edge.target_id);
              return sourceIsSeed && targetIsSeed;
            });
            
            return {
              papers: seedPapers,
              edges: seedEdges,
              seedPapers,
              selectedPapers: state.selectedPapers.filter(p => p.seed),
            };
          });
          
          get().filterGraph();
        },
      })
    ),
    {
      name: 'paper-store',
    }
  )
);