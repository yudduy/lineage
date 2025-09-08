import React, { useState, useCallback, useRef, useEffect } from 'react';
import { useForm, useFieldArray } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Search, 
  Filter, 
  Plus, 
  X, 
  Calendar,
  User,
  BookOpen,
  Tag,
  TrendingUp,
  Loader2,
  ChevronDown
} from 'lucide-react';
import toast from 'react-hot-toast';
import Modal from './Modal';
import { useUiStore, usePaperStore } from '@store/index';
import { SearchFilters, PaperSearchRequest } from '@types/paper';
import { PaperService } from '@services/paperService';
import Button from '@components/ui/Button';
import Fuse from 'fuse.js';

// Advanced search schema
const searchSchema = z.object({
  query: z.string().optional(),
  title: z.string().optional(),
  authors: z.array(z.string()).optional(),
  journal: z.string().optional(),
  subjects: z.array(z.string()).optional(),
  yearFrom: z.number().min(1900).max(new Date().getFullYear()).optional(),
  yearTo: z.number().min(1900).max(new Date().getFullYear()).optional(),
  citationMin: z.number().min(0).optional(),
  citationMax: z.number().min(0).optional(),
  openAccess: z.boolean().optional(),
  hasFullText: z.boolean().optional(),
  sortBy: z.enum(['relevance', 'year', 'citations', 'title']).default('relevance'),
  sortOrder: z.enum(['asc', 'desc']).default('desc'),
  limit: z.number().min(10).max(1000).default(50),
});

type SearchFormData = z.infer<typeof searchSchema>;

interface SearchModalProps {
  isOpen: boolean;
  onClose: () => void;
}

// Mock data for autocomplete
const mockJournals = [
  'Nature', 'Science', 'Cell', 'The Lancet', 'New England Journal of Medicine',
  'Journal of the American Chemical Society', 'Physical Review Letters',
  'Nature Genetics', 'Nature Medicine', 'PLOS ONE'
];

const mockSubjects = [
  'Computer Science', 'Biology', 'Physics', 'Chemistry', 'Medicine',
  'Mathematics', 'Engineering', 'Psychology', 'Economics', 'Neuroscience'
];

const SearchModal: React.FC<SearchModalProps> = ({ isOpen, onClose }) => {
  const [isSearching, setIsSearching] = useState(false);
  const [searchResults, setSearchResults] = useState<any[]>([]);
  const [showFilters, setShowFilters] = useState(false);
  const [authorSuggestions, setAuthorSuggestions] = useState<string[]>([]);
  const [journalSuggestions, setJournalSuggestions] = useState<string[]>([]);
  
  const { addPapers } = usePaperStore();
  const { setLoading } = useUiStore();

  const {
    register,
    control,
    handleSubmit,
    formState: { errors, isSubmitting },
    reset,
    watch,
    setValue,
    getValues,
  } = useForm<SearchFormData>({
    resolver: zodResolver(searchSchema),
    defaultValues: {
      sortBy: 'relevance',
      sortOrder: 'desc',
      limit: 50,
    },
  });

  const { fields: authorFields, append: appendAuthor, remove: removeAuthor } = useFieldArray({
    control,
    name: 'authors',
  });

  const { fields: subjectFields, append: appendSubject, remove: removeSubject } = useFieldArray({
    control,
    name: 'subjects',
  });

  const watchQuery = watch('query');
  const watchJournal = watch('journal');

  // Initialize fuzzy search for suggestions
  const journalFuse = useRef(new Fuse(mockJournals, { threshold: 0.3 }));
  const subjectFuse = useRef(new Fuse(mockSubjects, { threshold: 0.3 }));

  // Journal autocomplete
  useEffect(() => {
    if (watchJournal && watchJournal.length > 1) {
      const results = journalFuse.current.search(watchJournal).map(r => r.item);
      setJournalSuggestions(results.slice(0, 5));
    } else {
      setJournalSuggestions([]);
    }
  }, [watchJournal]);

  const onSubmit = async (data: SearchFormData) => {
    setIsSearching(true);
    setLoading('search', true);
    
    try {
      // Convert form data to API request
      const searchRequest: PaperSearchRequest = {
        query: data.query,
        title: data.title,
        authors: data.authors?.join(', '),
        journal: data.journal,
        publication_year_min: data.yearFrom,
        publication_year_max: data.yearTo,
        citation_count_min: data.citationMin,
        is_open_access: data.openAccess,
        sort_by: data.sortBy === 'relevance' ? 'score' : data.sortBy,
        sort_order: data.sortOrder,
        limit: data.limit,
      };

      // Perform search
      const results = await PaperService.searchPapers(searchRequest);
      setSearchResults(results);

      if (results.length === 0) {
        toast.error('No papers found matching your criteria. Try adjusting your search.');
      } else {
        toast.success(`Found ${results.length} papers matching your criteria`);
      }

    } catch (error) {
      console.error('Search error:', error);
      toast.error('Search failed. Please try again.');
    } finally {
      setIsSearching(false);
      setLoading('search', false);
    }
  };

  const handleAddSelectedPapers = useCallback((papers: any[]) => {
    addPapers(papers);
    toast.success(`Added ${papers.length} papers to your network`);
    onClose();
  }, [addPapers, onClose]);

  const handleReset = useCallback(() => {
    reset();
    setSearchResults([]);
    setShowFilters(false);
  }, [reset]);

  return (
    <Modal
      isOpen={isOpen}
      onClose={onClose}
      title="Advanced Paper Search"
      description="Search for papers using advanced filters and criteria"
      size="xl"
    >
      <div className="flex h-[80vh]">
        {/* Search Form */}
        <div className="w-1/2 p-6 border-r border-gray-200 dark:border-gray-700 overflow-y-auto">
          <form onSubmit={handleSubmit(onSubmit)} className="space-y-6">
            {/* Main Query */}
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Search Query
              </label>
              <div className="relative">
                <input
                  {...register('query')}
                  placeholder="Enter keywords, titles, or concepts..."
                  className="w-full px-4 py-3 pr-10 border border-gray-300 dark:border-gray-600 rounded-lg 
                           bg-white dark:bg-gray-800 text-gray-900 dark:text-white
                           focus:ring-2 focus:ring-blue-500 focus:border-transparent
                           placeholder-gray-500 dark:placeholder-gray-400"
                />
                <Search className="absolute right-3 top-3.5 h-5 w-5 text-gray-400" />
              </div>
            </div>

            {/* Title Search */}
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Title Contains
              </label>
              <input
                {...register('title')}
                placeholder="Search in paper titles..."
                className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg 
                         bg-white dark:bg-gray-800 text-gray-900 dark:text-white
                         focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
            </div>

            {/* Authors */}
            <div>
              <div className="flex items-center justify-between mb-2">
                <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
                  Authors
                </label>
                <button
                  type="button"
                  onClick={() => appendAuthor('')}
                  className="text-sm text-blue-600 dark:text-blue-400 hover:text-blue-800 dark:hover:text-blue-300"
                >
                  <Plus className="h-4 w-4 inline mr-1" />
                  Add Author
                </button>
              </div>
              <div className="space-y-2">
                {authorFields.map((field, index) => (
                  <div key={field.id} className="flex space-x-2">
                    <input
                      {...register(`authors.${index}` as const)}
                      placeholder="Author name..."
                      className="flex-1 px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg 
                               bg-white dark:bg-gray-800 text-gray-900 dark:text-white
                               focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    />
                    <button
                      type="button"
                      onClick={() => removeAuthor(index)}
                      className="p-2 text-red-500 hover:text-red-700 dark:hover:text-red-300"
                    >
                      <X className="h-4 w-4" />
                    </button>
                  </div>
                ))}
              </div>
            </div>

            {/* Advanced Filters Toggle */}
            <button
              type="button"
              onClick={() => setShowFilters(!showFilters)}
              className="flex items-center space-x-2 text-sm text-gray-600 dark:text-gray-400 hover:text-gray-800 dark:hover:text-gray-200"
            >
              <Filter className="h-4 w-4" />
              <span>Advanced Filters</span>
              <ChevronDown className={`h-4 w-4 transition-transform ${showFilters ? 'rotate-180' : ''}`} />
            </button>

            {/* Advanced Filters */}
            <AnimatePresence>
              {showFilters && (
                <motion.div
                  initial={{ height: 0, opacity: 0 }}
                  animate={{ height: 'auto', opacity: 1 }}
                  exit={{ height: 0, opacity: 0 }}
                  className="space-y-4"
                >
                  {/* Journal */}
                  <div className="relative">
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                      Journal
                    </label>
                    <input
                      {...register('journal')}
                      placeholder="Journal name..."
                      className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg 
                               bg-white dark:bg-gray-800 text-gray-900 dark:text-white
                               focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    />
                    {journalSuggestions.length > 0 && (
                      <div className="absolute z-10 w-full mt-1 bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-600 rounded-lg shadow-lg">
                        {journalSuggestions.map((journal, index) => (
                          <button
                            key={index}
                            type="button"
                            onClick={() => setValue('journal', journal)}
                            className="w-full px-4 py-2 text-left hover:bg-gray-100 dark:hover:bg-gray-700 first:rounded-t-lg last:rounded-b-lg"
                          >
                            {journal}
                          </button>
                        ))}
                      </div>
                    )}
                  </div>

                  {/* Publication Year Range */}
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                        Year From
                      </label>
                      <input
                        {...register('yearFrom', { valueAsNumber: true })}
                        type="number"
                        min="1900"
                        max={new Date().getFullYear()}
                        placeholder="1990"
                        className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg 
                                 bg-white dark:bg-gray-800 text-gray-900 dark:text-white
                                 focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                      />
                    </div>
                    <div>
                      <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                        Year To
                      </label>
                      <input
                        {...register('yearTo', { valueAsNumber: true })}
                        type="number"
                        min="1900"
                        max={new Date().getFullYear()}
                        placeholder="2024"
                        className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg 
                                 bg-white dark:bg-gray-800 text-gray-900 dark:text-white
                                 focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                      />
                    </div>
                  </div>

                  {/* Citation Count Range */}
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                        Min Citations
                      </label>
                      <input
                        {...register('citationMin', { valueAsNumber: true })}
                        type="number"
                        min="0"
                        placeholder="0"
                        className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg 
                                 bg-white dark:bg-gray-800 text-gray-900 dark:text-white
                                 focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                      />
                    </div>
                    <div>
                      <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                        Max Citations
                      </label>
                      <input
                        {...register('citationMax', { valueAsNumber: true })}
                        type="number"
                        min="0"
                        placeholder="10000"
                        className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg 
                                 bg-white dark:bg-gray-800 text-gray-900 dark:text-white
                                 focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                      />
                    </div>
                  </div>

                  {/* Checkboxes */}
                  <div className="space-y-2">
                    <label className="flex items-center">
                      <input
                        {...register('openAccess')}
                        type="checkbox"
                        className="mr-2"
                      />
                      <span className="text-sm text-gray-700 dark:text-gray-300">
                        Open Access only
                      </span>
                    </label>
                    <label className="flex items-center">
                      <input
                        {...register('hasFullText')}
                        type="checkbox"
                        className="mr-2"
                      />
                      <span className="text-sm text-gray-700 dark:text-gray-300">
                        Has full text available
                      </span>
                    </label>
                  </div>

                  {/* Sort Options */}
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                        Sort By
                      </label>
                      <select
                        {...register('sortBy')}
                        className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg 
                                 bg-white dark:bg-gray-800 text-gray-900 dark:text-white
                                 focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                      >
                        <option value="relevance">Relevance</option>
                        <option value="year">Publication Year</option>
                        <option value="citations">Citation Count</option>
                        <option value="title">Title</option>
                      </select>
                    </div>
                    <div>
                      <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                        Order
                      </label>
                      <select
                        {...register('sortOrder')}
                        className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg 
                                 bg-white dark:bg-gray-800 text-gray-900 dark:text-white
                                 focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                      >
                        <option value="desc">Descending</option>
                        <option value="asc">Ascending</option>
                      </select>
                    </div>
                  </div>

                  {/* Limit */}
                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                      Max Results
                    </label>
                    <select
                      {...register('limit', { valueAsNumber: true })}
                      className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg 
                               bg-white dark:bg-gray-800 text-gray-900 dark:text-white
                               focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    >
                      <option value={10}>10 results</option>
                      <option value={25}>25 results</option>
                      <option value={50}>50 results</option>
                      <option value={100}>100 results</option>
                      <option value={250}>250 results</option>
                      <option value={500}>500 results</option>
                    </select>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>

            {/* Search Actions */}
            <div className="flex space-x-3 pt-4 border-t border-gray-200 dark:border-gray-700">
              <Button
                type="submit"
                disabled={isSearching}
                className="flex-1"
              >
                {isSearching ? (
                  <>
                    <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                    Searching...
                  </>
                ) : (
                  <>
                    <Search className="h-4 w-4 mr-2" />
                    Search Papers
                  </>
                )}
              </Button>
              <Button
                type="button"
                variant="outline"
                onClick={handleReset}
                disabled={isSearching}
              >
                Reset
              </Button>
            </div>
          </form>
        </div>

        {/* Search Results */}
        <div className="w-1/2 p-6 overflow-y-auto">
          <div className="flex items-center justify-between mb-4">
            <h3 className="font-semibold text-gray-900 dark:text-white">
              Search Results
            </h3>
            {searchResults.length > 0 && (
              <Button
                onClick={() => handleAddSelectedPapers(searchResults)}
                size="sm"
                variant="outline"
              >
                <Plus className="h-4 w-4 mr-1" />
                Add All ({searchResults.length})
              </Button>
            )}
          </div>

          {searchResults.length === 0 ? (
            <div className="text-center py-12 text-gray-500 dark:text-gray-400">
              <Search className="h-12 w-12 mx-auto mb-4 opacity-50" />
              <p>No search results yet.</p>
              <p className="text-sm">Use the form to search for papers.</p>
            </div>
          ) : (
            <div className="space-y-4">
              {searchResults.map((paper, index) => (
                <motion.div
                  key={`${paper.id}-${index}`}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: index * 0.05 }}
                  className="p-4 border border-gray-200 dark:border-gray-700 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-800/50 transition-colors"
                >
                  <div className="space-y-2">
                    <h4 className="font-medium text-gray-900 dark:text-white leading-tight">
                      {paper.title}
                    </h4>
                    
                    {paper.authors && paper.authors.length > 0 && (
                      <p className="text-sm text-gray-600 dark:text-gray-400">
                        {paper.authors.slice(0, 3).map((a: any) => a.name).join(', ')}
                        {paper.authors.length > 3 && ' et al.'}
                      </p>
                    )}
                    
                    <div className="flex items-center space-x-4 text-sm text-gray-500 dark:text-gray-400">
                      {paper.journal?.name && (
                        <span>{paper.journal.name}</span>
                      )}
                      {paper.publication_year && (
                        <span>{paper.publication_year}</span>
                      )}
                      {paper.citation_count?.total !== undefined && (
                        <span>{paper.citation_count.total} citations</span>
                      )}
                    </div>

                    <div className="flex items-center justify-between">
                      <div className="flex items-center space-x-2">
                        {paper.is_open_access && (
                          <span className="text-xs bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-300 px-2 py-1 rounded">
                            Open Access
                          </span>
                        )}
                      </div>
                      <Button
                        size="sm"
                        variant="outline"
                        onClick={() => handleAddSelectedPapers([paper])}
                      >
                        <Plus className="h-4 w-4 mr-1" />
                        Add
                      </Button>
                    </div>
                  </div>
                </motion.div>
              ))}
            </div>
          )}
        </div>
      </div>
    </Modal>
  );
};

export default SearchModal;