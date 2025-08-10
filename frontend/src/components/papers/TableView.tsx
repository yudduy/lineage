import React, { useState, useMemo } from 'react';
import { usePaperStore, useUiStore } from '@store/index';
import { Paper } from '@types/paper';
import { TableColumn } from '@types/index';
import Button from '@components/ui/Button';
import Select from '@components/ui/Select';
import { clsx } from 'clsx';

const TableView: React.FC = () => {
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
  const [currentPage, setCurrentPage] = useState(1);
  const [itemsPerPage, setItemsPerPage] = useState(50);

  // Table columns configuration
  const columns: TableColumn<Paper>[] = [
    {
      key: 'select',
      label: '',
      width: '40px',
      render: (_, paper) => (
        <input
          type="checkbox"
          checked={selectedIds.has(paper.id!)}
          onChange={() => handleSelectMultiple(paper.id!)}
          className="rounded border-border"
          onClick={(e) => e.stopPropagation()}
        />
      ),
    },
    {
      key: 'title',
      label: 'Title',
      sortable: true,
      render: (title: string, paper) => (
        <div className="min-w-0">
          <div className="flex items-center space-x-2 mb-1">
            {paper.seed && (
              <span className="inline-flex items-center px-1.5 py-0.5 rounded text-xs font-medium bg-accent/10 text-accent">
                Seed
              </span>
            )}
            {paper.is_open_access && (
              <span className="inline-flex items-center px-1.5 py-0.5 rounded text-xs font-medium bg-green-100 text-green-800">
                OA
              </span>
            )}
          </div>
          <div className="font-medium text-text line-clamp-2" title={title}>
            {title || 'Untitled Paper'}
          </div>
          <div className="text-xs text-text-secondary mt-1">
            {formatAuthors(paper.authors)}
          </div>
        </div>
      ),
    },
    {
      key: 'publication_year',
      label: 'Year',
      sortable: true,
      width: '80px',
      render: (year: number) => (
        <span className="font-medium text-primary-600">
          {year || '—'}
        </span>
      ),
    },
    {
      key: 'journal',
      label: 'Journal',
      width: '200px',
      render: (journal: any) => (
        <div className="line-clamp-2 text-sm" title={journal?.name}>
          {journal?.name || '—'}
        </div>
      ),
    },
    {
      key: 'citation_count',
      label: 'Citations',
      sortable: true,
      width: '90px',
      render: (citationCount: any) => (
        <span className="font-medium">
          {citationCount?.total?.toLocaleString() || '0'}
        </span>
      ),
    },
    {
      key: 'local_metrics',
      label: 'Local',
      width: '80px',
      render: (_, paper) => {
        const total = (paper.localReferences || 0) + (paper.localCitedBy || 0);
        return (
          <div className="text-sm">
            <div className="font-medium">{total}</div>
            <div className="text-xs text-text-secondary">
              {paper.localReferences || 0}↗ {paper.localCitedBy || 0}↘
            </div>
          </div>
        );
      },
    },
    {
      key: 'actions',
      label: 'Actions',
      width: '120px',
      render: (_, paper) => (
        <div className="flex space-x-1">
          {paper.url && (
            <a
              href={paper.url}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center px-2 py-1 rounded text-xs bg-primary-100 text-primary-700 hover:bg-primary-200 transition-colors"
              onClick={(e) => e.stopPropagation()}
            >
              View
            </a>
          )}
          {paper.pdf_url && (
            <a
              href={paper.pdf_url}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center px-2 py-1 rounded text-xs bg-green-100 text-green-700 hover:bg-green-200 transition-colors"
              onClick={(e) => e.stopPropagation()}
            >
              PDF
            </a>
          )}
        </div>
      ),
    },
  ];

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
        case 'publication_year':
          aVal = a.publication_year || 0;
          bVal = b.publication_year || 0;
          break;
        case 'citation_count':
          aVal = a.citation_count?.total || 0;
          bVal = b.citation_count?.total || 0;
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

  // Pagination
  const totalPages = Math.ceil(filteredAndSortedPapers.length / itemsPerPage);
  const paginatedPapers = filteredAndSortedPapers.slice(
    (currentPage - 1) * itemsPerPage,
    currentPage * itemsPerPage
  );

  // Options
  const filterOptions = [
    { value: 'all', label: `All Papers (${papers.length})` },
    { value: 'seeds', label: `Seed Papers (${seedPapers.length})` },
    { value: 'connected', label: `Connected (${papers.length - seedPapers.length})` },
  ];

  const itemsPerPageOptions = [
    { value: '25', label: '25 per page' },
    { value: '50', label: '50 per page' },
    { value: '100', label: '100 per page' },
  ];

  // Handlers
  const formatAuthors = (authors: any[]) => {
    if (!authors || authors.length === 0) return 'Unknown authors';
    const names = authors.map(a => a.name).filter(Boolean);
    if (names.length === 0) return 'Unknown authors';
    if (names.length === 1) return names[0];
    return `${names[0]} et al.`;
  };

  const handleRowClick = (paper: Paper) => {
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
    if (selectedIds.size === paginatedPapers.length) {
      setSelectedIds(new Set());
      setSelectedPapers([]);
    } else {
      const allIds = new Set(paginatedPapers.map(p => p.id!));
      setSelectedIds(allIds);
      setSelectedPapers(paginatedPapers);
    }
  };

  const handleSort = (columnKey: string) => {
    const column = columns.find(col => col.key === columnKey);
    if (!column?.sortable) return;

    setSortConfig({
      field: columnKey,
      order: sortConfig.field === columnKey && sortConfig.order === 'asc' ? 'desc' : 'asc',
    });
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
          <h1 className="text-xl font-semibold text-text">Papers Table</h1>
          
          <div className="flex items-center space-x-2">
            <Select
              value={itemsPerPage.toString()}
              onChange={(value) => {
                setItemsPerPage(parseInt(value));
                setCurrentPage(1);
              }}
              options={itemsPerPageOptions}
              size="sm"
            />
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

      {/* Table */}
      <div className="flex-1 overflow-auto">
        {paginatedPapers.length === 0 ? (
          <EmptyState />
        ) : (
          <table className="w-full">
            {/* Header */}
            <thead className="bg-surface border-b border-border sticky top-0">
              <tr>
                <th className="w-10 px-3 py-3">
                  <input
                    type="checkbox"
                    checked={selectedIds.size === paginatedPapers.length && paginatedPapers.length > 0}
                    onChange={handleSelectAll}
                    className="rounded border-border"
                  />
                </th>
                {columns.slice(1).map((column) => (
                  <th
                    key={column.key}
                    className={clsx(
                      'px-3 py-3 text-left text-sm font-medium text-text-secondary',
                      column.width && `w-[${column.width}]`,
                      { 'cursor-pointer hover:text-text': column.sortable }
                    )}
                    onClick={() => column.sortable && handleSort(column.key)}
                  >
                    <div className="flex items-center space-x-1">
                      <span>{column.label}</span>
                      {column.sortable && sortConfig.field === column.key && (
                        <span className="text-primary-600">
                          {sortConfig.order === 'asc' ? '↑' : '↓'}
                        </span>
                      )}
                    </div>
                  </th>
                ))}
              </tr>
            </thead>
            
            {/* Body */}
            <tbody>
              {paginatedPapers.map((paper, index) => (
                <tr
                  key={paper.id}
                  className={clsx(
                    'border-b border-border hover:bg-surface cursor-pointer transition-colors',
                    {
                      'bg-primary-50': selectedPapers.some(p => p.id === paper.id),
                      'bg-background': !selectedPapers.some(p => p.id === paper.id),
                    }
                  )}
                  onClick={() => handleRowClick(paper)}
                >
                  <td className="px-3 py-3">
                    {columns[0].render?.('', paper)}
                  </td>
                  {columns.slice(1).map((column) => (
                    <td
                      key={column.key}
                      className={clsx(
                        'px-3 py-3 text-sm',
                        column.width && `w-[${column.width}]`
                      )}
                    >
                      {column.render
                        ? column.render((paper as any)[column.key], paper)
                        : (paper as any)[column.key] || '—'
                      }
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      {/* Pagination */}
      {totalPages > 1 && (
        <div className="border-t border-border px-4 py-3">
          <div className="flex items-center justify-between">
            <div className="text-sm text-text-secondary">
              Showing {(currentPage - 1) * itemsPerPage + 1} to{' '}
              {Math.min(currentPage * itemsPerPage, filteredAndSortedPapers.length)} of{' '}
              {filteredAndSortedPapers.length} papers
            </div>
            
            <div className="flex items-center space-x-2">
              <Button
                variant="outline"
                size="sm"
                onClick={() => setCurrentPage(Math.max(1, currentPage - 1))}
                disabled={currentPage === 1}
              >
                Previous
              </Button>
              
              <span className="text-sm text-text">
                Page {currentPage} of {totalPages}
              </span>
              
              <Button
                variant="outline"
                size="sm"
                onClick={() => setCurrentPage(Math.min(totalPages, currentPage + 1))}
                disabled={currentPage === totalPages}
              >
                Next
              </Button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

const EmptyState: React.FC = () => (
  <div className="flex flex-col items-center justify-center h-full text-center px-6">
    <div className="w-16 h-16 bg-surface rounded-lg flex items-center justify-center mb-4">
      <svg className="w-8 h-8 text-text-secondary" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 17V7m0 10a2 2 0 01-2 2H5a2 2 0 01-2-2V7a2 2 0 012-2h2a2 2 0 012 2m0 10a2 2 0 002 2h2a2 2 0 002-2M9 7a2 2 0 012-2h2a2 2 0 012 2m0 10V7m0 10a2 2 0 002 2h2a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
      </svg>
    </div>
    <h3 className="font-medium text-text mb-2">No papers found</h3>
    <p className="text-text-secondary">
      Try adjusting your filters or search terms
    </p>
  </div>
);

export default TableView;