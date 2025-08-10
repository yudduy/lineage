import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { Search, Network, Loader2, AlertCircle } from 'lucide-react';
import NetworkView from '@components/visualization/NetworkView';
import { api } from '@services/api';

interface NetworkData {
  center_paper_id: string;
  total_nodes: number;
  total_edges: number;
  max_depth_reached: number;
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
}

const DashboardPage: React.FC = () => {
  const [identifier, setIdentifier] = useState('');
  const [direction, setDirection] = useState('both');
  const [depth, setDepth] = useState(2);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [networkData, setNetworkData] = useState<NetworkData | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!identifier.trim()) {
      setError('Please enter a DOI, OpenAlex URL/ID, or paper title');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const response = await api.post('/api/v1/openalex/network/build-sync', {
        identifier: identifier.trim(),
        direction,
        max_depth: depth,
        max_per_level: 20
      });

      setNetworkData(response.data);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to build network. Please try again.');
      console.error('Network build error:', err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="h-full flex flex-col bg-gradient-to-b from-gray-50 to-white dark:from-gray-900 dark:to-gray-950">
      {/* Main Container */}
      <div className="flex-1 flex flex-col max-w-7xl mx-auto w-full px-4 py-8">
        
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center mb-8"
        >
          <h1 className="text-4xl font-bold text-gray-900 dark:text-white mb-2">
            Citation Network Explorer
          </h1>
          <p className="text-gray-600 dark:text-gray-400">
            Build and explore citation networks from academic papers
          </p>
        </motion.div>

        {/* Input Section */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-8"
        >
          <form onSubmit={handleSubmit} className="bg-white dark:bg-gray-800 rounded-2xl shadow-xl p-6">
            {/* Main Input */}
            <div className="mb-6">
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Enter Paper Identifier
              </label>
              <div className="relative">
                <input
                  type="text"
                  value={identifier}
                  onChange={(e) => setIdentifier(e.target.value)}
                  placeholder="Paste a DOI (10.1234/...), OpenAlex URL/ID (W...), or paper title..."
                  className="w-full px-4 py-3 pl-12 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent dark:bg-gray-700 dark:text-white"
                  disabled={loading}
                />
                <Search className="absolute left-4 top-3.5 h-5 w-5 text-gray-400" />
              </div>
            </div>

            {/* Options Row */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
              {/* Direction Selector */}
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Direction
                </label>
                <select
                  value={direction}
                  onChange={(e) => setDirection(e.target.value)}
                  className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:text-white"
                  disabled={loading}
                >
                  <option value="both">Both (Citations & References)</option>
                  <option value="backward">Backward (References Only)</option>
                  <option value="forward">Forward (Citations Only)</option>
                </select>
              </div>

              {/* Depth Slider */}
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Depth: {depth}
                </label>
                <input
                  type="range"
                  min="1"
                  max="3"
                  value={depth}
                  onChange={(e) => setDepth(Number(e.target.value))}
                  className="w-full"
                  disabled={loading}
                />
                <div className="flex justify-between text-xs text-gray-500 dark:text-gray-400">
                  <span>1</span>
                  <span>2</span>
                  <span>3</span>
                </div>
              </div>
            </div>

            {/* Error Message */}
            {error && (
              <div className="mb-4 p-3 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg flex items-start">
                <AlertCircle className="h-5 w-5 text-red-500 mr-2 flex-shrink-0 mt-0.5" />
                <span className="text-sm text-red-700 dark:text-red-400">{error}</span>
              </div>
            )}

            {/* Submit Button */}
            <button
              type="submit"
              disabled={loading}
              className="w-full py-3 px-6 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-400 text-white font-medium rounded-lg transition-colors flex items-center justify-center"
            >
              {loading ? (
                <>
                  <Loader2 className="h-5 w-5 mr-2 animate-spin" />
                  Building Network...
                </>
              ) : (
                <>
                  <Network className="h-5 w-5 mr-2" />
                  Build Citation Network
                </>
              )}
            </button>
          </form>
        </motion.div>

        {/* Network Visualization */}
        {networkData && (
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            className="flex-1 bg-white dark:bg-gray-800 rounded-2xl shadow-xl p-6 min-h-[500px]"
          >
            {/* Stats Bar */}
            <div className="mb-4 flex items-center justify-between">
              <h2 className="text-lg font-semibold text-gray-900 dark:text-white">
                Citation Network
              </h2>
              <div className="flex items-center space-x-4 text-sm text-gray-600 dark:text-gray-400">
                <span>{networkData.total_nodes} Papers</span>
                <span>•</span>
                <span>{networkData.total_edges} Citations</span>
                <span>•</span>
                <span>Depth: {networkData.max_depth_reached}</span>
              </div>
            </div>

            {/* Network Graph */}
            <div className="h-[500px] relative">
              <NetworkView 
                nodes={networkData.nodes}
                edges={networkData.edges}
                centerNodeId={networkData.center_paper_id}
              />
            </div>
          </motion.div>
        )}

        {/* Empty State */}
        {!networkData && !loading && (
          <div className="flex-1 flex items-center justify-center">
            <div className="text-center">
              <Network className="h-16 w-16 text-gray-300 dark:text-gray-600 mx-auto mb-4" />
              <p className="text-gray-500 dark:text-gray-400">
                Enter a paper identifier above to explore its citation network
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default DashboardPage;