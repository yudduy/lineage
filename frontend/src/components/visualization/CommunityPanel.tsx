import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Users, ChevronDown, ChevronRight, Eye, EyeOff } from 'lucide-react';
import { Community } from '@types/paper';

interface CommunityPanelProps {
  communities: Community[];
  onCommunitySelect?: (communityId: string) => void;
  onCommunityToggle?: (communityId: string, visible: boolean) => void;
}

const CommunityPanel: React.FC<CommunityPanelProps> = ({
  communities,
  onCommunitySelect,
  onCommunityToggle,
}) => {
  const [isExpanded, setIsExpanded] = useState(true);
  const [visibleCommunities, setVisibleCommunities] = useState<Set<string>>(
    new Set(communities.map(c => c.id))
  );

  const handleCommunityToggle = (communityId: string) => {
    const newVisible = new Set(visibleCommunities);
    if (visibleCommunities.has(communityId)) {
      newVisible.delete(communityId);
    } else {
      newVisible.add(communityId);
    }
    setVisibleCommunities(newVisible);
    onCommunityToggle?.(communityId, newVisible.has(communityId));
  };

  const sortedCommunities = [...communities].sort((a, b) => b.size - a.size);

  return (
    <div className="absolute top-4 left-4 z-10 max-w-xs">
      <motion.div
        initial={{ opacity: 0, x: -20 }}
        animate={{ opacity: 1, x: 0 }}
        className="bg-white dark:bg-gray-900 rounded-lg shadow-lg border border-gray-200 dark:border-gray-700"
      >
        {/* Header */}
        <div className="p-4 border-b border-gray-200 dark:border-gray-700">
          <button
            onClick={() => setIsExpanded(!isExpanded)}
            className="flex items-center space-x-2 w-full text-left hover:bg-gray-50 dark:hover:bg-gray-800 rounded-md p-1 -m-1"
          >
            {isExpanded ? (
              <ChevronDown className="h-4 w-4 text-gray-500" />
            ) : (
              <ChevronRight className="h-4 w-4 text-gray-500" />
            )}
            <Users className="h-5 w-5 text-purple-500" />
            <div className="flex-1">
              <h3 className="font-semibold text-gray-900 dark:text-white text-sm">
                Research Communities
              </h3>
              <p className="text-xs text-gray-500 dark:text-gray-400">
                {communities.length} detected
              </p>
            </div>
          </button>
        </div>

        {/* Community List */}
        <AnimatePresence>
          {isExpanded && (
            <motion.div
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: 'auto', opacity: 1 }}
              exit={{ height: 0, opacity: 0 }}
              className="max-h-80 overflow-y-auto"
            >
              {sortedCommunities.map((community, index) => (
                <motion.div
                  key={community.id}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: index * 0.05 }}
                  className="p-3 border-b border-gray-100 dark:border-gray-800 last:border-b-0 hover:bg-gray-50 dark:hover:bg-gray-800/50 transition-colors"
                >
                  <div className="flex items-start space-x-3">
                    {/* Community Color Indicator */}
                    <div
                      className="w-4 h-4 rounded-full flex-shrink-0 mt-0.5"
                      style={{ backgroundColor: community.color }}
                    />

                    {/* Community Info */}
                    <div className="flex-1 min-w-0">
                      <button
                        onClick={() => onCommunitySelect?.(community.id)}
                        className="text-left w-full group"
                      >
                        <h4 className="font-medium text-gray-900 dark:text-white text-sm group-hover:text-purple-600 dark:group-hover:text-purple-400 transition-colors">
                          {community.name}
                        </h4>
                        <div className="flex items-center space-x-3 mt-1 text-xs text-gray-500 dark:text-gray-400">
                          <span>{community.size} papers</span>
                          <span>•</span>
                          <span>Cohesion: {community.cohesion.toFixed(2)}</span>
                        </div>
                      </button>

                      {/* Community Metrics */}
                      <div className="mt-2 space-y-1">
                        <div className="flex items-center justify-between text-xs">
                          <span className="text-gray-600 dark:text-gray-400">Centrality</span>
                          <span className="font-medium text-gray-900 dark:text-white">
                            {community.centrality.toFixed(2)}
                          </span>
                        </div>
                        
                        {/* Centrality Bar */}
                        <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-1">
                          <div
                            className="h-1 rounded-full transition-all duration-300"
                            style={{ 
                              backgroundColor: community.color,
                              width: `${Math.min(100, community.centrality * 100)}%`
                            }}
                          />
                        </div>
                      </div>
                    </div>

                    {/* Visibility Toggle */}
                    <button
                      onClick={() => handleCommunityToggle(community.id)}
                      className="p-1 rounded-md hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
                      title={visibleCommunities.has(community.id) ? 'Hide community' : 'Show community'}
                    >
                      {visibleCommunities.has(community.id) ? (
                        <Eye className="h-4 w-4 text-gray-500" />
                      ) : (
                        <EyeOff className="h-4 w-4 text-gray-400" />
                      )}
                    </button>
                  </div>
                </motion.div>
              ))}
            </motion.div>
          )}
        </AnimatePresence>

        {/* Footer with Actions */}
        {isExpanded && (
          <div className="p-3 bg-gray-50 dark:bg-gray-800 rounded-b-lg border-t border-gray-200 dark:border-gray-700">
            <div className="flex items-center justify-between text-xs text-gray-500 dark:text-gray-400">
              <span>
                {visibleCommunities.size} of {communities.length} visible
              </span>
              <div className="flex items-center space-x-2">
                <button
                  onClick={() => {
                    setVisibleCommunities(new Set(communities.map(c => c.id)));
                    communities.forEach(c => onCommunityToggle?.(c.id, true));
                  }}
                  className="hover:text-gray-700 dark:hover:text-gray-300 transition-colors"
                >
                  Show All
                </button>
                <span>•</span>
                <button
                  onClick={() => {
                    setVisibleCommunities(new Set());
                    communities.forEach(c => onCommunityToggle?.(c.id, false));
                  }}
                  className="hover:text-gray-700 dark:hover:text-gray-300 transition-colors"
                >
                  Hide All
                </button>
              </div>
            </div>
          </div>
        )}
      </motion.div>
    </div>
  );
};

export default CommunityPanel;