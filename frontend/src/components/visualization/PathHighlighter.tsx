import React from 'react';
import { motion } from 'framer-motion';
import { Route, X, ArrowRight } from 'lucide-react';
import { GraphNode } from '@types/paper';

interface PathHighlighterProps {
  path: string[];
  nodes: GraphNode[];
  onPathClear: () => void;
}

const PathHighlighter: React.FC<PathHighlighterProps> = ({
  path,
  nodes,
  onPathClear,
}) => {
  const pathNodes = path.map(nodeId => 
    nodes.find(node => node.id === nodeId)
  ).filter(Boolean) as GraphNode[];

  if (pathNodes.length < 2) return null;

  const truncateTitle = (title: string, maxLength: number = 30) => {
    if (title.length <= maxLength) return title;
    return title.substring(0, maxLength) + '...';
  };

  const getNodeDisplayInfo = (node: GraphNode) => ({
    title: node.title || 'Untitled Paper',
    year: node.publication_year,
    authors: node.authors?.slice(0, 2).map(a => a.name.split(' ').pop()).join(', '),
    citations: node.citation_count?.total || 0,
    isSeed: node.seed,
  });

  return (
    <div className="absolute top-1/2 right-4 transform -translate-y-1/2 z-10">
      <motion.div
        initial={{ opacity: 0, x: 20 }}
        animate={{ opacity: 1, x: 0 }}
        className="bg-white dark:bg-gray-900 rounded-lg shadow-xl border border-gray-200 dark:border-gray-700 p-4 max-w-md"
      >
        {/* Header */}
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center space-x-2">
            <Route className="h-5 w-5 text-yellow-500" />
            <h3 className="font-semibold text-gray-900 dark:text-white text-sm">
              Citation Path
            </h3>
            <span className="text-xs bg-yellow-100 dark:bg-yellow-900/30 text-yellow-700 dark:text-yellow-300 px-2 py-0.5 rounded-full">
              {pathNodes.length} nodes
            </span>
          </div>
          <button
            onClick={onPathClear}
            className="p-1 rounded-md hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
            title="Clear path"
          >
            <X className="h-4 w-4 text-gray-500" />
          </button>
        </div>

        {/* Path Visualization */}
        <div className="space-y-3 max-h-80 overflow-y-auto">
          {pathNodes.map((node, index) => {
            const info = getNodeDisplayInfo(node);
            const isLast = index === pathNodes.length - 1;

            return (
              <div key={node.id} className="relative">
                {/* Node Card */}
                <motion.div
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: index * 0.1 }}
                  className={`p-3 rounded-lg border-2 transition-all ${
                    info.isSeed 
                      ? 'border-red-300 bg-red-50 dark:border-red-600 dark:bg-red-900/20' 
                      : 'border-yellow-300 bg-yellow-50 dark:border-yellow-600 dark:bg-yellow-900/20'
                  }`}
                >
                  {/* Node Header */}
                  <div className="flex items-start justify-between mb-2">
                    <div className="flex items-center space-x-2">
                      <div 
                        className={`w-3 h-3 rounded-full ${
                          info.isSeed ? 'bg-red-500' : 'bg-yellow-500'
                        }`} 
                      />
                      <span className="text-xs font-medium text-gray-600 dark:text-gray-400">
                        Node {index + 1}
                        {info.isSeed && (
                          <span className="ml-1 text-red-600 dark:text-red-400">• Seed</span>
                        )}
                      </span>
                    </div>
                    <div className="text-xs text-gray-500 dark:text-gray-400">
                      {info.year || 'N/A'}
                    </div>
                  </div>

                  {/* Paper Info */}
                  <div className="space-y-1">
                    <h4 className="font-medium text-gray-900 dark:text-white text-sm leading-tight">
                      {truncateTitle(info.title)}
                    </h4>
                    
                    {info.authors && (
                      <p className="text-xs text-gray-600 dark:text-gray-400">
                        {info.authors}
                      </p>
                    )}
                    
                    <div className="flex items-center justify-between text-xs text-gray-500 dark:text-gray-400">
                      <span>{info.citations} citations</span>
                      <span className="font-mono bg-gray-100 dark:bg-gray-800 px-1 py-0.5 rounded">
                        #{index + 1}
                      </span>
                    </div>
                  </div>
                </motion.div>

                {/* Arrow Connector */}
                {!isLast && (
                  <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ delay: index * 0.1 + 0.05 }}
                    className="flex justify-center my-2"
                  >
                    <div className="flex items-center space-x-1 text-yellow-500 dark:text-yellow-400">
                      <div className="w-px h-4 bg-current opacity-50"></div>
                      <ArrowRight className="h-3 w-3" />
                      <div className="w-px h-4 bg-current opacity-50"></div>
                    </div>
                  </motion.div>
                )}
              </div>
            );
          })}
        </div>

        {/* Path Summary */}
        <div className="mt-4 pt-3 border-t border-gray-200 dark:border-gray-700">
          <div className="grid grid-cols-3 gap-4 text-center text-xs">
            <div className="text-gray-600 dark:text-gray-400">
              <div className="font-semibold text-gray-900 dark:text-white">
                {pathNodes.length}
              </div>
              <div>Nodes</div>
            </div>
            <div className="text-gray-600 dark:text-gray-400">
              <div className="font-semibold text-gray-900 dark:text-white">
                {pathNodes.filter(n => n.seed).length}
              </div>
              <div>Seeds</div>
            </div>
            <div className="text-gray-600 dark:text-gray-400">
              <div className="font-semibold text-gray-900 dark:text-white">
                {pathNodes.length - 1}
              </div>
              <div>Steps</div>
            </div>
          </div>
        </div>

        {/* Instructions */}
        <div className="mt-3 p-2 bg-blue-50 dark:bg-blue-900/20 rounded-md">
          <p className="text-xs text-blue-700 dark:text-blue-300">
            💡 This path shows how papers are connected through citations. 
            Shift+click nodes to find new paths.
          </p>
        </div>
      </motion.div>
    </div>
  );
};

export default PathHighlighter;