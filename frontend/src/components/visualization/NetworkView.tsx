import React, { useRef, useEffect, useState, useCallback } from 'react';
import ForceGraph2D from 'react-force-graph-2d';
import { motion } from 'framer-motion';
import { ZoomIn, ZoomOut, Maximize2, RotateCcw } from 'lucide-react';

interface NetworkViewProps {
  nodes: Array<{
    id: string;
    title: string;
    publication_year: number;
    doi: string | null;
    citation_count: number;
  }>;
  edges: Array<{
    source_id: string;
    target_id: string;
  }>;
  centerNodeId?: string;
}

const NetworkView: React.FC<NetworkViewProps> = ({ nodes, edges, centerNodeId }) => {
  const fgRef = useRef<any>();
  const containerRef = useRef<HTMLDivElement>(null);
  const [dimensions, setDimensions] = useState({ width: 800, height: 600 });
  const [hoveredNode, setHoveredNode] = useState<any>(null);

  // Prepare graph data
  const graphData = React.useMemo(() => {
    // Create node map
    const nodeMap = new Map(nodes.map(n => [n.id, { ...n }]));
    
    // Create links with valid references
    const validLinks = edges
      .filter(e => nodeMap.has(e.source_id) && nodeMap.has(e.target_id))
      .map(e => ({
        source: e.source_id,
        target: e.target_id
      }));

    // Add visual properties to nodes
    const enhancedNodes = nodes.map(node => ({
      ...node,
      val: Math.sqrt(node.citation_count + 1) * 3,
      color: node.id === centerNodeId ? '#ef4444' : '#3b82f6',
      isCenter: node.id === centerNodeId
    }));

    return {
      nodes: enhancedNodes,
      links: validLinks
    };
  }, [nodes, edges, centerNodeId]);

  // Handle container resize
  useEffect(() => {
    const updateDimensions = () => {
      if (containerRef.current) {
        const rect = containerRef.current.getBoundingClientRect();
        setDimensions({
          width: rect.width,
          height: rect.height,
        });
      }
    };

    updateDimensions();
    const resizeObserver = new ResizeObserver(updateDimensions);
    if (containerRef.current) {
      resizeObserver.observe(containerRef.current);
    }

    return () => resizeObserver.disconnect();
  }, []);

  // Node paint function
  const nodeCanvasObject = useCallback((node: any, ctx: CanvasRenderingContext2D, globalScale: number) => {
    const label = node.title ? node.title.substring(0, 30) + (node.title.length > 30 ? '...' : '') : '';
    const fontSize = 12 / globalScale;
    ctx.font = `${fontSize}px Sans-Serif`;
    
    // Draw node circle
    ctx.beginPath();
    ctx.arc(node.x, node.y, node.val, 0, 2 * Math.PI, false);
    ctx.fillStyle = node.isCenter ? '#ef4444' : (hoveredNode?.id === node.id ? '#10b981' : '#3b82f6');
    ctx.fill();
    
    // Draw border for center node
    if (node.isCenter) {
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 2 / globalScale;
      ctx.stroke();
    }
    
    // Draw label
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillStyle = '#ffffff';
    ctx.fillText(label, node.x, node.y);
    
    // Draw year below
    if (node.publication_year) {
      ctx.font = `${fontSize * 0.8}px Sans-Serif`;
      ctx.fillStyle = '#94a3b8';
      ctx.fillText(node.publication_year.toString(), node.x, node.y + node.val + fontSize);
    }
  }, [hoveredNode]);

  // Control functions
  const handleZoomIn = () => {
    if (fgRef.current) {
      fgRef.current.zoom(1.2, 400);
    }
  };

  const handleZoomOut = () => {
    if (fgRef.current) {
      fgRef.current.zoom(0.8, 400);
    }
  };

  const handleZoomFit = () => {
    if (fgRef.current) {
      fgRef.current.zoomToFit(400, 50);
    }
  };

  const handleReset = () => {
    if (fgRef.current) {
      fgRef.current.centerAt(0, 0, 400);
      fgRef.current.zoom(1, 400);
    }
  };

  return (
    <div ref={containerRef} className="relative w-full h-full bg-gray-50 dark:bg-gray-900 rounded-lg overflow-hidden">
      {/* Controls */}
      <div className="absolute top-4 right-4 z-10 flex flex-col space-y-2">
        <button
          onClick={handleZoomIn}
          className="p-2 bg-white dark:bg-gray-800 rounded-lg shadow hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors"
          title="Zoom In"
        >
          <ZoomIn className="h-5 w-5 text-gray-600 dark:text-gray-400" />
        </button>
        <button
          onClick={handleZoomOut}
          className="p-2 bg-white dark:bg-gray-800 rounded-lg shadow hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors"
          title="Zoom Out"
        >
          <ZoomOut className="h-5 w-5 text-gray-600 dark:text-gray-400" />
        </button>
        <button
          onClick={handleZoomFit}
          className="p-2 bg-white dark:bg-gray-800 rounded-lg shadow hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors"
          title="Fit to View"
        >
          <Maximize2 className="h-5 w-5 text-gray-600 dark:text-gray-400" />
        </button>
        <button
          onClick={handleReset}
          className="p-2 bg-white dark:bg-gray-800 rounded-lg shadow hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors"
          title="Reset View"
        >
          <RotateCcw className="h-5 w-5 text-gray-600 dark:text-gray-400" />
        </button>
      </div>

      {/* Tooltip */}
      {hoveredNode && (
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          className="absolute z-20 p-3 bg-white dark:bg-gray-800 rounded-lg shadow-xl max-w-sm pointer-events-none"
          style={{
            left: hoveredNode.x + 20,
            top: hoveredNode.y - 20,
          }}
        >
          <div className="text-sm font-semibold text-gray-900 dark:text-white mb-1">
            {hoveredNode.title}
          </div>
          <div className="text-xs text-gray-600 dark:text-gray-400 space-y-1">
            {hoveredNode.publication_year && <div>Year: {hoveredNode.publication_year}</div>}
            {hoveredNode.doi && <div>DOI: {hoveredNode.doi}</div>}
            <div>Citations: {hoveredNode.citation_count}</div>
          </div>
        </motion.div>
      )}

      {/* Graph */}
      <ForceGraph2D
        ref={fgRef}
        graphData={graphData}
        width={dimensions.width}
        height={dimensions.height}
        nodeCanvasObject={nodeCanvasObject}
        nodePointerAreaPaint={(node, color, ctx) => {
          ctx.fillStyle = color;
          ctx.beginPath();
          ctx.arc(node.x, node.y, node.val, 0, 2 * Math.PI, false);
          ctx.fill();
        }}
        onNodeHover={(node) => setHoveredNode(node)}
        onNodeClick={(node) => {
          if (node.doi) {
            window.open(`https://doi.org/${node.doi}`, '_blank');
          }
        }}
        linkColor={() => '#94a3b8'}
        linkWidth={1}
        linkDirectionalArrowLength={3}
        linkDirectionalArrowRelPos={1}
        backgroundColor="transparent"
        cooldownTicks={100}
        onEngineStop={() => handleZoomFit()}
      />
    </div>
  );
};

export default NetworkView;