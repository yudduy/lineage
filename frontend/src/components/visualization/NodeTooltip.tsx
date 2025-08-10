import React from 'react';
import { motion } from 'framer-motion';
import { GraphNode, Community } from '@types/paper';
import { formatDistanceToNow } from 'date-fns';
import { Users, Award, TrendingUp, ExternalLink } from 'lucide-react';

interface NodeTooltipProps {
  node: GraphNode;
  position: { x: number; y: number };
  community?: Community | null;
}

const NodeTooltip: React.FC<NodeTooltipProps> = ({ node, position }) => {
  const formatYear = (year?: number) => {
    if (!year) return 'Unknown';
    return year.toString();
  };

  const formatAuthors = (authors: any[]) => {
    if (!authors || authors.length === 0) return 'Unknown authors';
    
    const firstAuthor = authors[0]?.name || 'Unknown';
    if (authors.length === 1) return firstAuthor;
    if (authors.length === 2) return `${firstAuthor} & ${authors[1].name}`;
    return `${firstAuthor} et al.`;
  };

  const formatCitations = (citationCount: any) => {
    const total = citationCount?.total || 0;
    return total.toLocaleString();
  };

  // Calculate tooltip position to keep it in viewport
  const tooltipStyle: React.CSSProperties = {
    position: 'fixed',
    left: position.x + 10,
    top: position.y - 10,
    zIndex: 9999,
    pointerEvents: 'none',
    transform: 'translate(0, -100%)',
  };

  // Adjust position if tooltip would go off screen
  if (typeof window !== 'undefined') {
    const viewportWidth = window.innerWidth;
    const viewportHeight = window.innerHeight;
    const tooltipWidth = 320; // approximate width
    const tooltipHeight = 200; // approximate height

    if (position.x + tooltipWidth + 20 > viewportWidth) {
      tooltipStyle.left = position.x - tooltipWidth - 10;
    }

    if (position.y - tooltipHeight < 0) {
      tooltipStyle.transform = 'translate(0, 10px)';
    }
  }

  return (
    <motion.div
      style={tooltipStyle}
      initial={{ opacity: 0, scale: 0.9, y: 10 }}
      animate={{ opacity: 1, scale: 1, y: 0 }}
      exit={{ opacity: 0, scale: 0.9, y: 10 }}
      transition={{ duration: 0.15 }}
      className="bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-700 rounded-lg shadow-xl p-4 max-w-sm backdrop-blur-sm"
    >
      {/* Paper Type Badges */}
      <div className="flex items-center gap-2 mb-3">
        {node.seed && (
          <span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-300">
            <Award className="h-3 w-3 mr-1" />
            Seed Paper
          </span>
        )}
        {node.is_open_access && (
          <span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-300">
            Open Access
          </span>
        )}
        {community && (
          <span 
            className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium text-white"
            style={{ backgroundColor: community.color }}
          >
            <Users className="h-3 w-3 mr-1" />
            {community.name}
          </span>
        )}
      </div>

      {/* Title */}
      <div className="font-semibold text-sm text-gray-900 dark:text-white mb-2 leading-tight">
        {node.title || 'Untitled Paper'}
      </div>

      {/* Authors */}
      <div className="text-xs text-gray-600 dark:text-gray-400 mb-2">
        {formatAuthors(node.authors)}
      </div>

      {/* Journal and Year */}
      <div className="flex items-center justify-between text-xs text-gray-600 dark:text-gray-400 mb-3">
        <span className="truncate flex-1 mr-2">
          {node.journal?.name || 'Unknown journal'}
        </span>
        <span className="text-blue-600 dark:text-blue-400 font-medium flex-shrink-0">
          {formatYear(node.publication_year)}
        </span>
      </div>

      {/* Enhanced Metrics */}
      <div className="grid grid-cols-3 gap-2 text-xs mb-3">
        <div className="bg-gray-50 dark:bg-gray-800 rounded-lg p-2">
          <div className="text-gray-500 dark:text-gray-400 text-xs">Citations</div>
          <div className="font-bold text-gray-900 dark:text-white">
            {formatCitations(node.citation_count)}
          </div>
        </div>
        <div className="bg-gray-50 dark:bg-gray-800 rounded-lg p-2">
          <div className="text-gray-500 dark:text-gray-400 text-xs">Local Refs</div>
          <div className="font-bold text-gray-900 dark:text-white">
            {(node.localCitedBy || 0) + (node.localReferences || 0)}
          </div>
        </div>
        <div className="bg-gray-50 dark:bg-gray-800 rounded-lg p-2">
          <div className="text-gray-500 dark:text-gray-400 text-xs">Seeds</div>
          <div className="font-bold text-gray-900 dark:text-white">
            {(node.seedsCitedBy || 0) + (node.seedsCited || 0)}
          </div>
        </div>
      </div>
      
      {/* Impact Indicators */}
      <div className="flex items-center space-x-4 text-xs mb-3">
        <div className="flex items-center space-x-1">
          <TrendingUp className="h-3 w-3 text-blue-500" />
          <span className="text-gray-600 dark:text-gray-400">Impact Score:</span>
          <span className="font-medium text-gray-900 dark:text-white">
            {Math.round(((node.citation_count?.total || 0) * (node.seedsCitedBy || 1)) / 10)}
          </span>
        </div>
      </div>

      {/* DOI */}
      {node.doi && (
        <div className="mt-3 pt-2 border-t border-gray-200 dark:border-gray-700">
          <div className="text-xs text-gray-600 dark:text-gray-400">
            <strong>DOI:</strong> <span className="font-mono">{node.doi}</span>
          </div>
        </div>
      )}

      {/* Enhanced Abstract preview */}
      {node.abstract && (
        <div className="mt-3 pt-2 border-t border-gray-200 dark:border-gray-700">
          <div className="text-xs text-gray-600 dark:text-gray-400">
            <strong className="text-gray-900 dark:text-white">Abstract:</strong>
          </div>
          <div className="text-xs text-gray-600 dark:text-gray-400 mt-1 line-clamp-3 leading-relaxed">
            {node.abstract}
          </div>
        </div>
      )}

      {/* Enhanced URLs */}
      <div className="flex gap-2 mt-3">
        {node.url && (
          <a
            href={node.url}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center px-3 py-1.5 rounded-lg text-xs bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300 hover:bg-blue-200 dark:hover:bg-blue-900/50 transition-colors font-medium"
            onClick={(e) => e.stopPropagation()}
          >
            <ExternalLink className="h-3 w-3 mr-1" />
            View Paper
          </a>
        )}
        {node.pdf_url && (
          <a
            href={node.pdf_url}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center px-3 py-1.5 rounded-lg text-xs bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-300 hover:bg-green-200 dark:hover:bg-green-900/50 transition-colors font-medium"
            onClick={(e) => e.stopPropagation()}
          >
            PDF
          </a>
        )}
      </div>
      
      {/* Interaction Hint */}
      <div className="mt-3 pt-2 border-t border-gray-200 dark:border-gray-700">
        <div className="text-xs text-gray-500 dark:text-gray-400">
          💡 <strong>Click</strong> to select • <strong>Ctrl+Click</strong> to multi-select • <strong>Shift+Click</strong> to find path
        </div>
      </div>
    </motion.div>
  );
};

export default NodeTooltip;