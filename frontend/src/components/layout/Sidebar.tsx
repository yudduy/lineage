import React from 'react';
import { usePaperStore, useUiStore } from '@store/index';
import Button from '@components/ui/Button';
import { Paper } from '@types/paper';
import { clsx } from 'clsx';

const Sidebar: React.FC = () => {
  const {
    seedPapers,
    papers,
    selectedPapers,
    setSelectedPapers,
    setHighlightedPaper,
    removeSeed,
    makeSeed,
  } = usePaperStore();
  
  const { openModal } = useUiStore();

  return (
    <div className="h-full flex flex-col bg-surface">
      {/* Header */}
      <div className="p-4 border-b border-border">
        <div className="flex items-center justify-between">
          <h2 className="font-semibold text-text">Seed Papers</h2>
          <Button
            variant="primary"
            size="xs"
            onClick={() => openModal('addPaper')}
            leftIcon={
              <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
              </svg>
            }
          >
            Add
          </Button>
        </div>
        <p className="text-xs text-text-secondary mt-1">
          {seedPapers.length} seed papers • {papers.length - seedPapers.length} connected papers
        </p>
      </div>

      {/* Seed Papers List */}
      <div className="flex-1 overflow-y-auto">
        {seedPapers.length === 0 ? (
          <EmptyState />
        ) : (
          <div className="p-2 space-y-1">
            {seedPapers.map((paper) => (
              <PaperCard
                key={paper.id}
                paper={paper}
                isSelected={selectedPapers.some(p => p.id === paper.id)}
                onSelect={() => {
                  setSelectedPapers([paper]);
                  setHighlightedPaper(paper);
                }}
                onRemoveSeed={() => removeSeed(paper.id!)}
              />
            ))}
          </div>
        )}
      </div>

      {/* Connected Papers Section */}
      {papers.length > seedPapers.length && (
        <ConnectedPapersSection />
      )}
    </div>
  );
};

const EmptyState: React.FC = () => {
  const { openModal } = useUiStore();

  return (
    <div className="flex flex-col items-center justify-center h-full px-6 py-8 text-center">
      <div className="w-12 h-12 bg-primary-100 rounded-lg flex items-center justify-center mb-4">
        <svg className="w-6 h-6 text-primary-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
        </svg>
      </div>
      
      <h3 className="font-medium text-text mb-2">No papers yet</h3>
      <p className="text-sm text-text-secondary mb-4">
        Add papers to start exploring citation networks
      </p>
      
      <div className="space-y-2 w-full">
        <Button
          variant="primary"
          size="sm"
          onClick={() => openModal('addPaper')}
          fullWidth
        >
          Add Papers
        </Button>
        <Button
          variant="outline"
          size="sm"
          onClick={() => openModal('help')}
          fullWidth
        >
          Learn More
        </Button>
      </div>
    </div>
  );
};

interface PaperCardProps {
  paper: Paper;
  isSelected: boolean;
  onSelect: () => void;
  onRemoveSeed: () => void;
}

const PaperCard: React.FC<PaperCardProps> = ({
  paper,
  isSelected,
  onSelect,
  onRemoveSeed,
}) => {
  const formatAuthors = (authors: any[]) => {
    if (!authors || authors.length === 0) return 'Unknown authors';
    const firstAuthor = authors[0]?.name || 'Unknown';
    if (authors.length === 1) return firstAuthor;
    return `${firstAuthor} et al.`;
  };

  const handleRemove = (e: React.MouseEvent) => {
    e.stopPropagation();
    onRemoveSeed();
  };

  return (
    <div
      className={clsx(
        'p-3 rounded-lg border cursor-pointer transition-all',
        'hover:shadow-sm group',
        {
          'border-primary-300 bg-primary-50': isSelected,
          'border-border bg-background hover:bg-surface': !isSelected,
        }
      )}
      onClick={onSelect}
    >
      <div className="flex items-start justify-between">
        <div className="flex-1 min-w-0">
          {/* Title */}
          <h4 className="font-medium text-sm text-text line-clamp-2 mb-1">
            {paper.title || 'Untitled Paper'}
          </h4>
          
          {/* Authors */}
          <p className="text-xs text-text-secondary line-clamp-1 mb-1">
            {formatAuthors(paper.authors)}
          </p>
          
          {/* Year and Journal */}
          <div className="flex items-center space-x-2 text-xs text-text-secondary">
            {paper.publication_year && (
              <span className="font-medium text-primary-600">
                {paper.publication_year}
              </span>
            )}
            {paper.journal?.name && (
              <span className="line-clamp-1">
                {paper.journal.name}
              </span>
            )}
          </div>
          
          {/* Metrics */}
          <div className="flex items-center space-x-3 text-xs text-text-secondary mt-2">
            <div className="flex items-center space-x-1">
              <span>📊</span>
              <span>{paper.citation_count?.total || 0} citations</span>
            </div>
            <div className="flex items-center space-x-1">
              <span>🔗</span>
              <span>{(paper.localReferences || 0) + (paper.localCitedBy || 0)} local</span>
            </div>
          </div>
        </div>
        
        {/* Remove button */}
        <button
          onClick={handleRemove}
          className="ml-2 p-1 text-text-secondary hover:text-red-600 opacity-0 group-hover:opacity-100 transition-opacity"
          title="Remove from seeds"
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
      </div>
      
      {/* URLs */}
      {(paper.url || paper.pdf_url) && (
        <div className="flex space-x-1 mt-2">
          {paper.url && (
            <a
              href={paper.url}
              target="_blank"
              rel="noopener noreferrer"
              onClick={(e) => e.stopPropagation()}
              className="inline-flex items-center px-2 py-1 rounded text-xs bg-primary-100 text-primary-700 hover:bg-primary-200 transition-colors"
            >
              View
            </a>
          )}
          {paper.pdf_url && (
            <a
              href={paper.pdf_url}
              target="_blank"
              rel="noopener noreferrer"
              onClick={(e) => e.stopPropagation()}
              className="inline-flex items-center px-2 py-1 rounded text-xs bg-green-100 text-green-700 hover:bg-green-200 transition-colors"
            >
              PDF
            </a>
          )}
        </div>
      )}
    </div>
  );
};

const ConnectedPapersSection: React.FC = () => {
  const { papers, seedPapers, selectedPapers, makeSeed } = usePaperStore();
  
  const connectedPapers = papers.filter(p => !p.seed);
  const selectedNonSeeds = selectedPapers.filter(p => !p.seed);

  const handleMakeSeeds = () => {
    if (selectedNonSeeds.length > 0) {
      makeSeed(selectedNonSeeds);
    }
  };

  return (
    <div className="border-t border-border">
      <div className="p-3">
        <div className="flex items-center justify-between mb-2">
          <h3 className="font-medium text-sm text-text">Connected Papers</h3>
          <span className="text-xs text-text-secondary">
            {connectedPapers.length} papers
          </span>
        </div>
        
        {selectedNonSeeds.length > 0 && (
          <Button
            variant="outline"
            size="xs"
            onClick={handleMakeSeeds}
            fullWidth
            className="mb-2"
          >
            Make {selectedNonSeeds.length} Selected into Seeds
          </Button>
        )}
        
        <p className="text-xs text-text-secondary">
          Papers connected to your seeds. Select papers in the network to see more details.
        </p>
      </div>
    </div>
  );
};

export default Sidebar;