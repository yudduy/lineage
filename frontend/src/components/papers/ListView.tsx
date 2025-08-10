import React, { useState, useMemo } from 'react';
import { usePaperStore, useUiStore } from '@store/index';
import { Paper } from '@types/paper';
import Button from '@components/ui/Button';
import Select from '@components/ui/Select';
import { clsx } from 'clsx';

const ListView: React.FC = () => {
  const {
    papers,
    seedPapers,
    selectedPapers,
    setSelectedPapers,
    setHighlightedPaper,
    makeSeed,
    removeSeed,
  } = usePaperStore();
  
  const { searchQuery, sortConfig, setSortConfig } = useUiStore();
  
  const [viewFilter, setViewFilter] = useState<'all' | 'seeds' | 'connected'>('all');
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());

  // Filter and sort papers
  const filteredAndSortedPapers = useMemo(() => {
    let filtered = papers;

    // Apply view filter
    switch (viewFilter) {
      case 'seeds':
        filtered = seedPapers;
        break;
      case 'connected':
        filtered = papers.filter(p => !p.seed);
        break;
      default:
        break;
    }

    // Apply search filter
    if (searchQuery.trim()) {
      const query = searchQuery.toLowerCase();
      filtered = filtered.filter(paper =>
        paper.title.toLowerCase().includes(query) ||
        paper.authors.some(author => author.name.toLowerCase().includes(query)) ||
        paper.abstract?.toLowerCase().includes(query) ||
        paper.journal?.name?.toLowerCase().includes(query)
      );
    }

    // Sort papers
    const sorted = [...filtered].sort((a, b) => {
      const { field, order } = sortConfig;
      let aVal: any = '';
      let bVal: any = '';

      switch (field) {
        case 'title':
          aVal = a.title || '';
          bVal = b.title || '';
          break;
        case 'year':
          aVal = a.publication_year || 0;
          bVal = b.publication_year || 0;
          break;
        case 'citations':
          aVal = a.citation_count?.total || 0;
          bVal = b.citation_count?.total || 0;
          break;
        case 'created_at':
          aVal = new Date(a.created_at || 0).getTime();
          bVal = new Date(b.created_at || 0).getTime();
          break;
        default:
          aVal = a.title || '';
          bVal = b.title || '';
      }

      if (typeof aVal === 'string') {
        return order === 'asc' 
          ? aVal.localeCompare(bVal)
          : bVal.localeCompare(aVal);
      } else {
        return order === 'asc' ? aVal - bVal : bVal - aVal;
      }
    });

    return sorted;
  }, [papers, seedPapers, viewFilter, searchQuery, sortConfig]);

  const filterOptions = [
    { value: 'all', label: `All Papers (${papers.length})` },
    { value: 'seeds', label: `Seed Papers (${seedPapers.length})` },
    { value: 'connected', label: `Connected (${papers.length - seedPapers.length})` },
  ];

  const sortOptions = [
    { value: 'title', label: 'Title' },
    { value: 'year', label: 'Publication Year' },
    { value: 'citations', label: 'Citation Count' },
    { value: 'created_at', label: 'Date Added' },
  ];

  const handleSelectPaper = (paper: Paper) => {
    setSelectedPapers([paper]);
    setHighlightedPaper(paper);
  };

  const handleSelectMultiple = (paperId: string) => {
    const newSelected = new Set(selectedIds);
    if (newSelected.has(paperId)) {
      newSelected.delete(paperId);
    } else {
      newSelected.add(paperId);
    }
    setSelectedIds(newSelected);

    // Update global selection
    const selectedPapers = papers.filter(p => newSelected.has(p.id!));
    setSelectedPapers(selectedPapers);
  };

  const handleSelectAll = () => {
    if (selectedIds.size === filteredAndSortedPapers.length) {
      setSelectedIds(new Set());
      setSelectedPapers([]);
    } else {
      const allIds = new Set(filteredAndSortedPapers.map(p => p.id!));
      setSelectedIds(allIds);
      setSelectedPapers(filteredAndSortedPapers);
    }
  };

  const selectedNonSeeds = Array.from(selectedIds)
    .map(id => papers.find(p => p.id === id))
    .filter(p => p && !p.seed) as Paper[];

  const selectedSeeds = Array.from(selectedIds)
    .map(id => papers.find(p => p.id === id))
    .filter(p => p && p.seed) as Paper[];

  const handleMakeSeeds = () => {
    if (selectedNonSeeds.length > 0) {
      makeSeed(selectedNonSeeds);
      setSelectedIds(new Set());
    }
  };

  const handleRemoveSeeds = () => {
    if (selectedSeeds.length > 0) {
      selectedSeeds.forEach(paper => removeSeed(paper.id!));
      setSelectedIds(new Set());
    }
  };

  return (
    <div className="h-full flex flex-col bg-background">
      {/* Header */}
      <div className="border-b border-border p-4">
        <div className="flex items-center justify-between mb-4">
          <h1 className="text-xl font-semibold text-text">Papers List</h1>
          
          <div className="flex items-center space-x-2">
            <Select
              value={sortConfig.field}
              onChange={(field) => setSortConfig({ ...sortConfig, field })}
              options={sortOptions}
              size="sm"
            />
            
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setSortConfig({ 
                ...sortConfig, 
                order: sortConfig.order === 'asc' ? 'desc' : 'asc' 
              })}
            >
              {sortConfig.order === 'asc' ? '↑' : '↓'}
            </Button>
          </div>
        </div>
        
        <div className="flex items-center justify-between">
          <Select
            value={viewFilter}
            onChange={(value) => setViewFilter(value as any)}
            options={filterOptions}
            size="sm"
          />
          
          {selectedIds.size > 0 && (
            <div className="flex items-center space-x-2">
              <span className="text-sm text-text-secondary">
                {selectedIds.size} selected
              </span>
              
              {selectedNonSeeds.length > 0 && (
                <Button
                  variant="primary"
                  size="xs"
                  onClick={handleMakeSeeds}
                >
                  Make Seeds ({selectedNonSeeds.length})
                </Button>
              )}
              
              {selectedSeeds.length > 0 && (
                <Button
                  variant="outline"
                  size="xs"
                  onClick={handleRemoveSeeds}
                >
                  Remove Seeds ({selectedSeeds.length})
                </Button>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Paper List */}
      <div className="flex-1 overflow-y-auto">
        {filteredAndSortedPapers.length === 0 ? (
          <EmptyState />
        ) : (
          <div className="p-4">
            {/* Select All */}
            <div className="mb-4 pb-3 border-b border-border">
              <label className="flex items-center space-x-2 text-sm">
                <input
                  type="checkbox"
                  checked={selectedIds.size === filteredAndSortedPapers.length && filteredAndSortedPapers.length > 0}
                  onChange={handleSelectAll}
                  className="rounded border-border"
                />
                <span>
                  Select all {filteredAndSortedPapers.length} papers
                </span>
              </label>
            </div>
            
            {/* Papers */}
            <div className="space-y-3">
              {filteredAndSortedPapers.map((paper) => (
                <PaperCard
                  key={paper.id}
                  paper={paper}
                  isSelected={selectedIds.has(paper.id!)}
                  isHighlighted={selectedPapers.some(p => p.id === paper.id)}
                  onSelect={() => handleSelectPaper(paper)}
                  onToggleSelect={() => handleSelectMultiple(paper.id!)}
                />
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

const EmptyState: React.FC = () => (
  <div className="flex flex-col items-center justify-center h-full text-center px-6">
    <div className="w-16 h-16 bg-surface rounded-lg flex items-center justify-center mb-4">
      <svg className="w-8 h-8 text-text-secondary" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
      </svg>
    </div>
    <h3 className="font-medium text-text mb-2">No papers found</h3>
    <p className="text-text-secondary">
      Try adjusting your filters or search terms
    </p>
  </div>
);

interface PaperCardProps {
  paper: Paper;
  isSelected: boolean;
  isHighlighted: boolean;
  onSelect: () => void;
  onToggleSelect: () => void;
}

const PaperCard: React.FC<PaperCardProps> = ({
  paper,
  isSelected,
  isHighlighted,
  onSelect,
  onToggleSelect,
}) => {
  const formatAuthors = (authors: any[]) => {
    if (!authors || authors.length === 0) return 'Unknown authors';
    const names = authors.map(a => a.name).filter(Boolean);
    if (names.length === 0) return 'Unknown authors';
    if (names.length === 1) return names[0];
    if (names.length <= 3) return names.join(', ');
    return `${names.slice(0, 2).join(', ')} et al.`;
  };

  return (
    <div
      className={clsx(
        'border rounded-lg p-4 transition-all hover:shadow-sm cursor-pointer',
        {
          'border-primary-300 bg-primary-50': isHighlighted,
          'border-border bg-background hover:bg-surface': !isHighlighted,
        }
      )}
      onClick={onSelect}
    >
      <div className="flex items-start space-x-3">
        {/* Checkbox */}
        <input
          type="checkbox"
          checked={isSelected}
          onChange={onToggleSelect}
          onClick={(e) => e.stopPropagation()}
          className="mt-1 rounded border-border"
        />
        
        <div className="flex-1 min-w-0">
          {/* Header */}
          <div className="flex items-start justify-between mb-2">
            <div className="flex items-center space-x-2">
              {paper.seed && (
                <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-accent/10 text-accent">
                  Seed
                </span>
              )}
              {paper.is_open_access && (
                <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-green-100 text-green-800">
                  Open Access
                </span>
              )}
            </div>
            
            <div className="text-xs text-text-secondary">
              {paper.publication_year || 'Unknown year'}
            </div>
          </div>
          
          {/* Title */}
          <h3 className="font-semibold text-text mb-2 line-clamp-2">
            {paper.title || 'Untitled Paper'}
          </h3>
          
          {/* Authors */}
          <p className="text-sm text-text-secondary mb-2">
            {formatAuthors(paper.authors)}
          </p>
          
          {/* Journal */}
          {paper.journal?.name && (
            <p className="text-sm text-text-secondary mb-2 font-medium">
              {paper.journal.name}
            </p>
          )}
          
          {/* Abstract */}
          {paper.abstract && (
            <p className="text-sm text-text-secondary line-clamp-3 mb-3">
              {paper.abstract}
            </p>
          )}
          
          {/* Metrics */}
          <div className="flex items-center space-x-6 text-sm text-text-secondary mb-3">
            <div className="flex items-center space-x-1">
              <span>📊</span>
              <span>{paper.citation_count?.total || 0} citations</span>
            </div>
            <div className="flex items-center space-x-1">
              <span>🔗</span>
              <span>{(paper.localReferences || 0) + (paper.localCitedBy || 0)} local</span>
            </div>
            {paper.subjects.length > 0 && (
              <div className="flex items-center space-x-1">
                <span>🏷️</span>
                <span>{paper.subjects.length} topics</span>
              </div>
            )}
          </div>
          
          {/* URLs and Actions */}
          <div className="flex items-center space-x-2">
            {paper.url && (
              <a
                href={paper.url}
                target="_blank"
                rel="noopener noreferrer"
                onClick={(e) => e.stopPropagation()}
                className="inline-flex items-center px-3 py-1 rounded text-sm bg-primary-100 text-primary-700 hover:bg-primary-200 transition-colors"
              >
                View Paper
              </a>
            )}
            {paper.pdf_url && (
              <a
                href={paper.pdf_url}
                target="_blank"
                rel="noopener noreferrer"
                onClick={(e) => e.stopPropagation()}
                className="inline-flex items-center px-3 py-1 rounded text-sm bg-green-100 text-green-700 hover:bg-green-200 transition-colors"
              >
                PDF
              </a>
            )}
            {paper.doi && (
              <a
                href={`https://doi.org/${paper.doi}`}
                target="_blank"
                rel="noopener noreferrer"
                onClick={(e) => e.stopPropagation()}
                className="inline-flex items-center px-3 py-1 rounded text-sm bg-gray-100 text-gray-700 hover:bg-gray-200 transition-colors"
              >
                DOI
              </a>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default ListView;