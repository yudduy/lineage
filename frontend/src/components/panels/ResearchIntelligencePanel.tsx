import React, { useState, useEffect, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Brain, 
  TrendingUp, 
  TrendingDown, 
  Target, 
  Lightbulb, 
  Clock,
  Award,
  Users,
  BookOpen,
  RefreshCw,
  ChevronDown,
  ChevronRight,
  Sparkles
} from 'lucide-react';
import { LineChart, Line, AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { usePaperStore, useUiStore } from '@store/index';
import { ResearchIntelligence, TrendAnalysis } from '@types/paper';
import Button from '@components/ui/Button';

interface ResearchIntelligencePanelProps {
  className?: string;
}

const ResearchIntelligencePanel: React.FC<ResearchIntelligencePanelProps> = ({ className }) => {
  const [isExpanded, setIsExpanded] = useState(true);
  const [isLoading, setIsLoading] = useState(false);
  const [intelligenceData, setIntelligenceData] = useState<ResearchIntelligence[]>([]);
  const [selectedTimeframe, setSelectedTimeframe] = useState<'1y' | '3y' | '5y' | 'all'>('3y');

  const { papers, selectedPapers, seedPapers } = usePaperStore();
  const { openModal } = useUiStore();

  // Generate mock research intelligence data
  const generateIntelligenceData = useMemo(() => {
    return papers.slice(0, 10).map((paper, index) => ({
      id: `intel_${paper.id}`,
      paperId: paper.id!,
      summary: `AI-generated insights for "${paper.title?.substring(0, 50)}...": This research shows significant impact in ${paper.subjects?.[0] || 'interdisciplinary'} studies with growing citation patterns.`,
      keyInsights: [
        'High citation velocity suggests emerging importance',
        'Strong interdisciplinary connections identified',
        'Potential for breakthrough applications',
        'Growing research community interest'
      ],
      impactPrediction: Math.random() * 10,
      trendAnalysis: {
        momentum: Math.random() * 2 - 1,
        growth_rate: Math.random() * 0.5,
        peak_year: paper.publication_year ? paper.publication_year + Math.floor(Math.random() * 5) : undefined,
        declining: Math.random() > 0.7,
        emerging_topics: ['AI/ML Applications', 'Interdisciplinary Research', 'Novel Methodologies']
      },
      recommendations: [
        'Consider exploring related methodologies',
        'Investigate collaboration opportunities',
        'Monitor for emerging applications',
        'Track citation patterns for trend validation'
      ],
      generatedAt: new Date().toISOString()
    }));
  }, [papers]);

  useEffect(() => {
    setIntelligenceData(generateIntelligenceData);
  }, [generateIntelligenceData]);

  // Generate trend data for visualization
  const trendData = useMemo(() => {
    const currentYear = new Date().getFullYear();
    const startYear = selectedTimeframe === '1y' ? currentYear - 1 : 
                     selectedTimeframe === '3y' ? currentYear - 3 :
                     selectedTimeframe === '5y' ? currentYear - 5 : currentYear - 10;
    
    return Array.from({ length: currentYear - startYear + 1 }, (_, i) => ({
      year: startYear + i,
      citations: Math.floor(Math.random() * 100) + 20,
      impact: Math.random() * 10,
      papers: Math.floor(Math.random() * 20) + 5,
      collaboration: Math.random() * 50
    }));
  }, [selectedTimeframe]);

  const handleRefreshIntelligence = async () => {
    setIsLoading(true);
    // Simulate API call
    await new Promise(resolve => setTimeout(resolve, 2000));
    setIntelligenceData(generateIntelligenceData);
    setIsLoading(false);
  };

  const getImpactColor = (score: number) => {
    if (score >= 8) return 'text-green-600 dark:text-green-400';
    if (score >= 6) return 'text-yellow-600 dark:text-yellow-400';
    if (score >= 4) return 'text-orange-600 dark:text-orange-400';
    return 'text-red-600 dark:text-red-400';
  };

  const getImpactLabel = (score: number) => {
    if (score >= 8) return 'High Impact';
    if (score >= 6) return 'Medium Impact';
    if (score >= 4) return 'Moderate Impact';
    return 'Emerging';
  };

  const getTrendIcon = (trend: TrendAnalysis) => {
    if (trend.declining) return <TrendingDown className="h-4 w-4 text-red-500" />;
    if (trend.growth_rate > 0.3) return <TrendingUp className="h-4 w-4 text-green-500" />;
    return <Target className="h-4 w-4 text-blue-500" />;
  };

  return (
    <div className={`bg-white dark:bg-gray-900 rounded-lg shadow-lg border border-gray-200 dark:border-gray-700 ${className}`}>
      {/* Header */}
      <div className="p-4 border-b border-gray-200 dark:border-gray-700">
        <button
          onClick={() => setIsExpanded(!isExpanded)}
          className="flex items-center space-x-3 w-full text-left hover:bg-gray-50 dark:hover:bg-gray-800 rounded-lg p-2 -m-2"
        >
          {isExpanded ? (
            <ChevronDown className="h-5 w-5 text-gray-500" />
          ) : (
            <ChevronRight className="h-5 w-5 text-gray-500" />
          )}
          <Brain className="h-6 w-6 text-purple-500" />
          <div className="flex-1">
            <h2 className="font-bold text-gray-900 dark:text-white">
              Research Intelligence
            </h2>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              AI-powered insights and trend analysis
            </p>
          </div>
          <div className="flex items-center space-x-2">
            <Button
              size="sm"
              variant="outline"
              onClick={(e) => {
                e.stopPropagation();
                handleRefreshIntelligence();
              }}
              disabled={isLoading}
              className="text-xs"
            >
              {isLoading ? (
                <RefreshCw className="h-3 w-3 animate-spin" />
              ) : (
                <RefreshCw className="h-3 w-3" />
              )}
            </Button>
          </div>
        </button>
      </div>

      {/* Content */}
      <AnimatePresence>
        {isExpanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            className="overflow-hidden"
          >
            <div className="p-4 space-y-6">
              {/* Overview Stats */}
              <div className="grid grid-cols-4 gap-4">
                <div className="text-center p-3 bg-purple-50 dark:bg-purple-900/20 rounded-lg">
                  <div className="text-2xl font-bold text-purple-600 dark:text-purple-400">
                    {papers.length}
                  </div>
                  <div className="text-xs text-gray-600 dark:text-gray-400">Total Papers</div>
                </div>
                <div className="text-center p-3 bg-green-50 dark:bg-green-900/20 rounded-lg">
                  <div className="text-2xl font-bold text-green-600 dark:text-green-400">
                    {intelligenceData.filter(i => i.impactPrediction >= 7).length}
                  </div>
                  <div className="text-xs text-gray-600 dark:text-gray-400">High Impact</div>
                </div>
                <div className="text-center p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
                  <div className="text-2xl font-bold text-blue-600 dark:text-blue-400">
                    {intelligenceData.filter(i => !i.trendAnalysis.declining).length}
                  </div>
                  <div className="text-xs text-gray-600 dark:text-gray-400">Trending Up</div>
                </div>
                <div className="text-center p-3 bg-orange-50 dark:bg-orange-900/20 rounded-lg">
                  <div className="text-2xl font-bold text-orange-600 dark:text-orange-400">
                    {seedPapers.length}
                  </div>
                  <div className="text-xs text-gray-600 dark:text-gray-400">Seed Papers</div>
                </div>
              </div>

              {/* Trend Visualization */}
              <div>
                <div className="flex items-center justify-between mb-4">
                  <h3 className="font-semibold text-gray-900 dark:text-white">
                    Research Trends
                  </h3>
                  <div className="flex space-x-1">
                    {(['1y', '3y', '5y', 'all'] as const).map((timeframe) => (
                      <button
                        key={timeframe}
                        onClick={() => setSelectedTimeframe(timeframe)}
                        className={`px-3 py-1 text-xs rounded-full transition-colors ${
                          selectedTimeframe === timeframe
                            ? 'bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300'
                            : 'text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-800'
                        }`}
                      >
                        {timeframe === 'all' ? 'All' : timeframe}
                      </button>
                    ))}
                  </div>
                </div>
                
                <div className="h-48">
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={trendData}>
                      <defs>
                        <linearGradient id="impactGradient" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor="#8b5cf6" stopOpacity={0.8}/>
                          <stop offset="95%" stopColor="#8b5cf6" stopOpacity={0}/>
                        </linearGradient>
                      </defs>
                      <CartesianGrid strokeDasharray="3 3" className="opacity-30" />
                      <XAxis 
                        dataKey="year" 
                        tick={{ fontSize: 12 }}
                        className="text-gray-600 dark:text-gray-400"
                      />
                      <YAxis 
                        tick={{ fontSize: 12 }}
                        className="text-gray-600 dark:text-gray-400"
                      />
                      <Tooltip 
                        contentStyle={{
                          backgroundColor: 'rgba(255, 255, 255, 0.95)',
                          border: '1px solid #e5e7eb',
                          borderRadius: '8px',
                          fontSize: '12px'
                        }}
                      />
                      <Area
                        type="monotone"
                        dataKey="impact"
                        stroke="#8b5cf6"
                        fillOpacity={1}
                        fill="url(#impactGradient)"
                      />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
              </div>

              {/* Intelligence Insights */}
              <div>
                <h3 className="font-semibold text-gray-900 dark:text-white mb-4 flex items-center">
                  <Sparkles className="h-5 w-5 text-purple-500 mr-2" />
                  AI Insights
                </h3>
                
                <div className="space-y-4 max-h-80 overflow-y-auto">
                  {intelligenceData.slice(0, 5).map((intel, index) => {
                    const paper = papers.find(p => p.id === intel.paperId);
                    if (!paper) return null;

                    return (
                      <motion.div
                        key={intel.id}
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: index * 0.1 }}
                        className="p-4 border border-gray-200 dark:border-gray-700 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-800/50 transition-colors"
                      >
                        <div className="flex items-start justify-between mb-2">
                          <div className="flex-1 min-w-0">
                            <h4 className="font-medium text-gray-900 dark:text-white text-sm leading-tight mb-1">
                              {paper.title?.substring(0, 60)}...
                            </h4>
                            <div className="flex items-center space-x-3 text-xs text-gray-500 dark:text-gray-400">
                              <span>Impact: {intel.impactPrediction.toFixed(1)}</span>
                              <span>•</span>
                              <span className={getImpactColor(intel.impactPrediction)}>
                                {getImpactLabel(intel.impactPrediction)}
                              </span>
                            </div>
                          </div>
                          <div className="flex items-center space-x-1">
                            {getTrendIcon(intel.trendAnalysis)}
                          </div>
                        </div>

                        <p className="text-xs text-gray-600 dark:text-gray-400 mb-3 leading-relaxed">
                          {intel.summary}
                        </p>

                        <div className="space-y-2">
                          <div>
                            <span className="text-xs font-medium text-gray-700 dark:text-gray-300">Key Insights:</span>
                            <ul className="text-xs text-gray-600 dark:text-gray-400 mt-1 space-y-1">
                              {intel.keyInsights.slice(0, 2).map((insight, i) => (
                                <li key={i} className="flex items-start">
                                  <span className="text-purple-500 mr-1">•</span>
                                  {insight}
                                </li>
                              ))}
                            </ul>
                          </div>

                          {intel.trendAnalysis.emerging_topics.length > 0 && (
                            <div className="flex items-center space-x-2">
                              <span className="text-xs font-medium text-gray-700 dark:text-gray-300">Topics:</span>
                              <div className="flex flex-wrap gap-1">
                                {intel.trendAnalysis.emerging_topics.slice(0, 2).map((topic, i) => (
                                  <span
                                    key={i}
                                    className="text-xs bg-purple-100 dark:bg-purple-900/30 text-purple-700 dark:text-purple-300 px-2 py-1 rounded-full"
                                  >
                                    {topic}
                                  </span>
                                ))}
                              </div>
                            </div>
                          )}
                        </div>
                      </motion.div>
                    );
                  })}
                </div>
              </div>

              {/* Action Buttons */}
              <div className="flex space-x-3 pt-4 border-t border-gray-200 dark:border-gray-700">
                <Button
                  size="sm"
                  variant="outline"
                  onClick={() => openModal('researchIntelligence')}
                  className="flex-1 text-xs"
                >
                  <Brain className="h-4 w-4 mr-1" />
                  Detailed Analysis
                </Button>
                <Button
                  size="sm"
                  variant="outline"
                  onClick={() => openModal('communityAnalysis')}
                  className="flex-1 text-xs"
                >
                  <Users className="h-4 w-4 mr-1" />
                  Community Insights
                </Button>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default ResearchIntelligencePanel;