import React, { useEffect, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Activity, Zap, TrendingUp } from 'lucide-react';
import { CitationFlow } from '@types/paper';

interface CitationFlowVisualizerProps {
  flows: CitationFlow[];
  onFlowSelect?: (flow: CitationFlow) => void;
}

const CitationFlowVisualizer: React.FC<CitationFlowVisualizerProps> = ({
  flows,
  onFlowSelect,
}) => {
  const [activeFlows, setActiveFlows] = useState<CitationFlow[]>([]);
  const [isExpanded, setIsExpanded] = useState(false);

  useEffect(() => {
    // Filter and update active flows
    const now = Date.now();
    const active = flows.filter(flow => {
      const age = now - flow.timestamp;
      return age < 5000; // Show flows for 5 seconds
    });
    setActiveFlows(active);
  }, [flows]);

  const getFlowIntensityColor = (intensity: number) => {
    if (intensity >= 5) return 'text-red-500';
    if (intensity >= 3) return 'text-orange-500';
    if (intensity >= 2) return 'text-yellow-500';
    return 'text-blue-500';
  };

  const getFlowDescription = (flow: CitationFlow) => {
    switch (flow.context) {
      case 'citation':
        return 'Citation link';
      case 'influence':
        return 'Influence flow';
      case 'collaboration':
        return 'Collaboration';
      default:
        return 'Connection';
    }
  };

  if (activeFlows.length === 0) return null;

  return (
    <div className="absolute top-20 right-4 z-10">
      <motion.div
        initial={{ opacity: 0, scale: 0.8 }}
        animate={{ opacity: 1, scale: 1 }}
        className="bg-white dark:bg-gray-900 rounded-lg shadow-lg border border-gray-200 dark:border-gray-700 p-4 min-w-72"
      >
        {/* Header */}
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center space-x-2">
            <Activity className="h-5 w-5 text-blue-500" />
            <h3 className="font-semibold text-gray-900 dark:text-white text-sm">
              Citation Flows
            </h3>
            <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
          </div>
          <button
            onClick={() => setIsExpanded(!isExpanded)}
            className="text-xs text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-300"
          >
            {isExpanded ? 'Collapse' : 'Expand'}
          </button>
        </div>

        {/* Flow Summary */}
        <div className="mb-3 p-2 bg-gray-50 dark:bg-gray-800 rounded-md">
          <div className="flex items-center justify-between text-xs text-gray-600 dark:text-gray-400">
            <span>Active Flows: {activeFlows.length}</span>
            <div className="flex items-center space-x-1">
              <TrendingUp className="h-3 w-3" />
              <span>
                Avg. Intensity: {(activeFlows.reduce((sum, f) => sum + f.intensity, 0) / activeFlows.length).toFixed(1)}
              </span>
            </div>
          </div>
        </div>

        {/* Flow List */}
        {isExpanded && (
          <AnimatePresence>
            <motion.div
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: 'auto', opacity: 1 }}
              exit={{ height: 0, opacity: 0 }}
              className="space-y-2 max-h-60 overflow-y-auto"
            >
              {activeFlows.slice(0, 10).map((flow, index) => {
                const age = (Date.now() - flow.timestamp) / 1000;
                const opacity = Math.max(0.3, 1 - age / 5);

                return (
                  <motion.div
                    key={`${flow.sourceId}-${flow.targetId}-${flow.timestamp}`}
                    initial={{ opacity: 0, x: 20 }}
                    animate={{ opacity: opacity, x: 0 }}
                    exit={{ opacity: 0, x: -20 }}
                    transition={{ duration: 0.2, delay: index * 0.05 }}
                    onClick={() => onFlowSelect?.(flow)}
                    className="p-2 bg-gray-50 dark:bg-gray-800 rounded border border-gray-200 dark:border-gray-700 cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
                  >
                    <div className="flex items-center space-x-2">
                      <Zap className={`h-3 w-3 ${getFlowIntensityColor(flow.intensity)}`} />
                      <div className="flex-1 min-w-0">
                        <div className="text-xs font-medium text-gray-900 dark:text-white truncate">
                          {getFlowDescription(flow)}
                        </div>
                        <div className="text-xs text-gray-500 dark:text-gray-400">
                          Intensity: {flow.intensity} • {age.toFixed(1)}s ago
                        </div>
                      </div>
                    </div>
                    
                    {/* Flow Progress Bar */}
                    <div className="mt-2 w-full bg-gray-200 dark:bg-gray-600 rounded-full h-1">
                      <div
                        className={`h-1 rounded-full transition-all duration-100 ${
                          flow.intensity >= 5 ? 'bg-red-500' :
                          flow.intensity >= 3 ? 'bg-orange-500' :
                          flow.intensity >= 2 ? 'bg-yellow-500' : 'bg-blue-500'
                        }`}
                        style={{ 
                          width: `${Math.min(100, (flow.intensity / 10) * 100)}%`,
                          opacity: opacity
                        }}
                      />
                    </div>
                  </motion.div>
                );
              })}
            </motion.div>
          </AnimatePresence>
        )}

        {/* Flow Legend */}
        {!isExpanded && (
          <div className="flex items-center justify-between text-xs text-gray-500 dark:text-gray-400">
            <div className="flex items-center space-x-3">
              <div className="flex items-center space-x-1">
                <div className="w-2 h-2 bg-red-500 rounded-full"></div>
                <span>High</span>
              </div>
              <div className="flex items-center space-x-1">
                <div className="w-2 h-2 bg-orange-500 rounded-full"></div>
                <span>Med</span>
              </div>
              <div className="flex items-center space-x-1">
                <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
                <span>Low</span>
              </div>
            </div>
            <span className="text-xs opacity-75">Click flows to select</span>
          </div>
        )}
      </motion.div>
    </div>
  );
};

export default CitationFlowVisualizer;