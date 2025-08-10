import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import {
  FileText,
  Users,
  Calendar,
  BookOpen,
  ExternalLink,
  Quote,
  TrendingUp,
  Globe,
  Award,
  ChevronDown,
  ChevronRight,
  Sparkles,
  Copy,
  Check,
  Eye,
  Download
} from 'lucide-react';
import { usePaperStore } from '@store/index';
import { Paper } from '@types/paper';
import Button from '@components/ui/Button';
import toast from 'react-hot-toast';

interface PaperDetailsPanelProps {
  className?: string;
}

const PaperDetailsPanel: React.FC<PaperDetailsPanelProps> = ({ className }) => {
  const [isExpanded, setIsExpanded] = useState(true);
  const [selectedPaper, setSelectedPaper] = useState<Paper | null>(null);
  const [activeTab, setActiveTab] = useState<'overview' | 'abstract' | 'citations' | 'ai-summary'>('overview');
  const [aiSummary, setAiSummary] = useState<string>('');
  const [isLoadingAI, setIsLoadingAI] = useState(false);
  const [copiedField, setCopiedField] = useState<string | null>(null);

  const { selectedPapers, papers, graphLinks } = usePaperStore();

  useEffect(() => {
    if (selectedPapers.length > 0) {
      setSelectedPaper(selectedPapers[0]);
    } else {
      setSelectedPaper(null);
    }
  }, [selectedPapers]);

  // Generate AI summary (mock)
  const generateAISummary = async (paper: Paper) => {
    setIsLoadingAI(true);
    // Simulate AI processing
    await new Promise(resolve => setTimeout(resolve, 2000));
    
    const summary = `## AI-Generated Summary

**Key Contributions:**
This research makes significant contributions to ${paper.subjects?.[0] || 'the field'} by introducing novel methodologies and providing comprehensive analysis. The work demonstrates:

- Innovative approach to problem-solving in the domain
- Strong empirical validation with substantial dataset
- Theoretical foundations that advance current understanding
- Practical implications for future research directions

**Impact Analysis:**
With ${paper.citation_count?.total || 0} citations, this work has shown ${paper.citation_count?.total > 50 ? 'significant' : 'emerging'} influence in the research community. The citation pattern suggests ${paper.publication_year && (new Date().getFullYear() - paper.publication_year) < 3 ? 'growing momentum' : 'sustained relevance'} in the field.

**Research Context:**
The paper builds upon established foundations while introducing novel perspectives. It connects to ${paper.localReferences || 0} related works in the current network, suggesting strong integration with existing research landscapes.

**Future Directions:**
The work opens several promising avenues for future investigation, particularly in interdisciplinary applications and methodological refinements.`;

    setAiSummary(summary);
    setIsLoadingAI(false);
  };

  const handleCopyToClipboard = async (text: string, field: string) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopiedField(field);
      toast.success('Copied to clipboard');
      setTimeout(() => setCopiedField(null), 2000);
    } catch (error) {
      toast.error('Failed to copy to clipboard');
    }
  };

  const formatAuthors = (authors: any[]) => {
    if (!authors || authors.length === 0) return 'Unknown authors';
    return authors.map(author => author.name).join(', ');
  };

  const getConnectedPapers = () => {
    if (!selectedPaper) return { citing: [], cited: [] };
    
    const citing = graphLinks
      .filter(link => link.target === selectedPaper.id)
      .map(link => papers.find(p => p.id === link.source))
      .filter(Boolean) as Paper[];
    
    const cited = graphLinks
      .filter(link => link.source === selectedPaper.id)
      .map(link => papers.find(p => p.id === link.target))
      .filter(Boolean) as Paper[];
    
    return { citing, cited };
  };

  if (!selectedPaper) {
    return (
      <div className={`bg-white dark:bg-gray-900 rounded-lg shadow-lg border border-gray-200 dark:border-gray-700 ${className}`}>
        <div className="p-8 text-center text-gray-500 dark:text-gray-400">
          <FileText className="h-12 w-12 mx-auto mb-4 opacity-50" />
          <p>Select a paper to view detailed information</p>
        </div>
      </div>
    );
  }

  const { citing, cited } = getConnectedPapers();

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
          <FileText className="h-6 w-6 text-blue-500" />
          <div className="flex-1">
            <h2 className="font-bold text-gray-900 dark:text-white">
              Paper Details
            </h2>
            <p className="text-sm text-gray-600 dark:text-gray-400 truncate">
              {selectedPaper.title}
            </p>
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
            <div className="p-4">
              {/* Paper Header */}
              <div className="mb-6">
                <div className="flex items-start justify-between mb-3">
                  <div className="flex-1 min-w-0 mr-4">
                    <h3 className="font-semibold text-gray-900 dark:text-white leading-tight mb-2">
                      {selectedPaper.title}
                    </h3>
                    <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                      {formatAuthors(selectedPaper.authors)}
                    </p>
                  </div>
                  {selectedPaper.seed && (
                    <span className="flex-shrink-0 bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-300 px-2 py-1 rounded-full text-xs font-medium">
                      <Award className="h-3 w-3 inline mr-1" />
                      Seed Paper
                    </span>
                  )}
                </div>

                <div className="flex items-center space-x-4 text-sm text-gray-500 dark:text-gray-400 mb-4">
                  {selectedPaper.journal?.name && (
                    <span className="flex items-center">
                      <BookOpen className="h-4 w-4 mr-1" />
                      {selectedPaper.journal.name}
                    </span>
                  )}
                  {selectedPaper.publication_year && (
                    <span className="flex items-center">
                      <Calendar className="h-4 w-4 mr-1" />
                      {selectedPaper.publication_year}
                    </span>
                  )}
                  <span className="flex items-center">
                    <TrendingUp className="h-4 w-4 mr-1" />
                    {selectedPaper.citation_count?.total || 0} citations
                  </span>
                </div>

                {/* Quick Actions */}
                <div className="flex space-x-2 mb-4">
                  {selectedPaper.url && (
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={() => window.open(selectedPaper.url, '_blank')}
                      className="text-xs"
                    >
                      <ExternalLink className="h-3 w-3 mr-1" />
                      View Paper
                    </Button>
                  )}
                  {selectedPaper.doi && (
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={() => handleCopyToClipboard(selectedPaper.doi!, 'doi')}
                      className="text-xs"
                    >
                      {copiedField === 'doi' ? (
                        <Check className="h-3 w-3 mr-1" />
                      ) : (
                        <Copy className="h-3 w-3 mr-1" />
                      )}
                      Copy DOI
                    </Button>
                  )}
                  <Button
                    size="sm"
                    variant="outline"
                    onClick={() => generateAISummary(selectedPaper)}
                    disabled={isLoadingAI}
                    className="text-xs"
                  >
                    <Sparkles className="h-3 w-3 mr-1" />
                    AI Summary
                  </Button>
                </div>
              </div>

              {/* Tabs */}
              <div className="border-b border-gray-200 dark:border-gray-700 mb-4">
                <nav className="flex space-x-6">
                  {[
                    { id: 'overview', label: 'Overview', icon: Eye },
                    { id: 'abstract', label: 'Abstract', icon: FileText },
                    { id: 'citations', label: 'Citations', icon: Quote },
                    { id: 'ai-summary', label: 'AI Summary', icon: Sparkles }
                  ].map(({ id, label, icon: Icon }) => (
                    <button
                      key={id}
                      onClick={() => setActiveTab(id as typeof activeTab)}
                      className={`flex items-center space-x-2 py-2 px-1 border-b-2 text-sm font-medium transition-colors ${
                        activeTab === id
                          ? 'border-blue-500 text-blue-600 dark:text-blue-400'
                          : 'border-transparent text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300'
                      }`}
                    >
                      <Icon className="h-4 w-4" />
                      <span>{label}</span>
                    </button>
                  ))}
                </nav>
              </div>

              {/* Tab Content */}
              <div className="space-y-4 max-h-96 overflow-y-auto">
                {/* Overview Tab */}
                {activeTab === 'overview' && (
                  <motion.div
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="space-y-4"
                  >
                    {/* Metrics Grid */}
                    <div className="grid grid-cols-2 gap-4">
                      <div className="bg-gray-50 dark:bg-gray-800 rounded-lg p-3">
                        <div className="text-sm text-gray-600 dark:text-gray-400">Total Citations</div>
                        <div className="text-xl font-bold text-gray-900 dark:text-white">
                          {selectedPaper.citation_count?.total || 0}
                        </div>
                      </div>
                      <div className="bg-gray-50 dark:bg-gray-800 rounded-lg p-3">
                        <div className="text-sm text-gray-600 dark:text-gray-400">Local Network</div>
                        <div className="text-xl font-bold text-gray-900 dark:text-white">
                          {(selectedPaper.localCitedBy || 0) + (selectedPaper.localReferences || 0)}
                        </div>
                      </div>
                    </div>

                    {/* Publication Details */}
                    <div className="space-y-3">
                      {selectedPaper.doi && (
                        <div>
                          <span className="text-sm font-medium text-gray-700 dark:text-gray-300">DOI:</span>
                          <div className="text-sm text-gray-600 dark:text-gray-400 font-mono mt-1">
                            {selectedPaper.doi}
                          </div>
                        </div>
                      )}

                      {selectedPaper.subjects && selectedPaper.subjects.length > 0 && (
                        <div>
                          <span className="text-sm font-medium text-gray-700 dark:text-gray-300">Subjects:</span>
                          <div className="flex flex-wrap gap-1 mt-1">
                            {selectedPaper.subjects.map((subject, index) => (
                              <span
                                key={index}
                                className="text-xs bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300 px-2 py-1 rounded-full"
                              >
                                {subject}
                              </span>
                            ))}
                          </div>
                        </div>
                      )}

                      {selectedPaper.keywords && selectedPaper.keywords.length > 0 && (
                        <div>
                          <span className="text-sm font-medium text-gray-700 dark:text-gray-300">Keywords:</span>
                          <div className="flex flex-wrap gap-1 mt-1">
                            {selectedPaper.keywords.map((keyword, index) => (
                              <span
                                key={index}
                                className="text-xs bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 px-2 py-1 rounded"
                              >
                                {keyword}
                              </span>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  </motion.div>
                )}

                {/* Abstract Tab */}
                {activeTab === 'abstract' && (
                  <motion.div
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                  >
                    {selectedPaper.abstract ? (
                      <div className="prose prose-sm dark:prose-invert max-w-none">
                        <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
                          {selectedPaper.abstract}
                        </p>
                      </div>
                    ) : (
                      <div className="text-center py-8 text-gray-500 dark:text-gray-400">
                        <FileText className="h-8 w-8 mx-auto mb-2 opacity-50" />
                        <p>No abstract available</p>
                      </div>
                    )}
                  </motion.div>
                )}

                {/* Citations Tab */}
                {activeTab === 'citations' && (
                  <motion.div
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="space-y-4"
                  >
                    {/* Citing Papers */}
                    {citing.length > 0 && (
                      <div>
                        <h4 className="font-medium text-gray-900 dark:text-white mb-2">
                          Citing Papers ({citing.length})
                        </h4>
                        <div className="space-y-2">
                          {citing.slice(0, 5).map((paper, index) => (
                            <div key={paper.id} className="text-sm p-2 bg-gray-50 dark:bg-gray-800 rounded">
                              <div className="font-medium text-gray-900 dark:text-white">{paper.title}</div>
                              <div className="text-gray-600 dark:text-gray-400">
                                {formatAuthors(paper.authors)} • {paper.publication_year}
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Cited Papers */}
                    {cited.length > 0 && (
                      <div>
                        <h4 className="font-medium text-gray-900 dark:text-white mb-2">
                          Cited Papers ({cited.length})
                        </h4>
                        <div className="space-y-2">
                          {cited.slice(0, 5).map((paper, index) => (
                            <div key={paper.id} className="text-sm p-2 bg-gray-50 dark:bg-gray-800 rounded">
                              <div className="font-medium text-gray-900 dark:text-white">{paper.title}</div>
                              <div className="text-gray-600 dark:text-gray-400">
                                {formatAuthors(paper.authors)} • {paper.publication_year}
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {citing.length === 0 && cited.length === 0 && (
                      <div className="text-center py-8 text-gray-500 dark:text-gray-400">
                        <Quote className="h-8 w-8 mx-auto mb-2 opacity-50" />
                        <p>No citation connections found in current network</p>
                      </div>
                    )}
                  </motion.div>
                )}

                {/* AI Summary Tab */}
                {activeTab === 'ai-summary' && (
                  <motion.div
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                  >
                    {isLoadingAI ? (
                      <div className="text-center py-8">
                        <div className="animate-spin w-8 h-8 border-2 border-purple-500 border-t-transparent rounded-full mx-auto mb-4"></div>
                        <p className="text-gray-600 dark:text-gray-400">Generating AI summary...</p>
                      </div>
                    ) : aiSummary ? (
                      <div className="prose prose-sm dark:prose-invert max-w-none">
                        <ReactMarkdown remarkPlugins={[remarkGfm]}>
                          {aiSummary}
                        </ReactMarkdown>
                      </div>
                    ) : (
                      <div className="text-center py-8 text-gray-500 dark:text-gray-400">
                        <Sparkles className="h-8 w-8 mx-auto mb-2 opacity-50" />
                        <p>Click "AI Summary" button to generate insights</p>
                      </div>
                    )}
                  </motion.div>
                )}
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default PaperDetailsPanel;