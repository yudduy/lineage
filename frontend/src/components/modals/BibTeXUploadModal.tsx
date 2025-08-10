import React, { useState, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useDropzone } from 'react-dropzone';
import { 
  Upload, 
  FileText, 
  Check, 
  X, 
  AlertCircle, 
  Download,
  Plus,
  Eye,
  Loader2
} from 'lucide-react';
import toast from 'react-hot-toast';
// @ts-ignore
import { parseBibTeX } from 'bibtex-parser-js';
import Modal from './Modal';
import { useUiStore, usePaperStore } from '@store/index';
import { Paper } from '@types/paper';
import Button from '@components/ui/Button';

interface BibTeXUploadModalProps {
  isOpen: boolean;
  onClose: () => void;
}

interface ParsedEntry {
  key: string;
  type: string;
  title?: string;
  author?: string;
  journal?: string;
  year?: string;
  doi?: string;
  url?: string;
  abstract?: string;
  volume?: string;
  number?: string;
  pages?: string;
  publisher?: string;
  [key: string]: any;
}

const BibTeXUploadModal: React.FC<BibTeXUploadModalProps> = ({ isOpen, onClose }) => {
  const [bibTexContent, setBibTexContent] = useState('');
  const [parsedEntries, setParsedEntries] = useState<ParsedEntry[]>([]);
  const [selectedEntries, setSelectedEntries] = useState<Set<string>>(new Set());
  const [isProcessing, setIsProcessing] = useState(false);
  const [step, setStep] = useState<'upload' | 'preview' | 'success'>('upload');
  const [parseErrors, setParseErrors] = useState<string[]>([]);

  const { addPapers, makeSeed } = usePaperStore();
  const { setLoading } = useUiStore();

  const onDrop = useCallback((acceptedFiles: File[]) => {
    const file = acceptedFiles[0];
    if (file && file.type === 'text/plain' || file.name.endsWith('.bib')) {
      const reader = new FileReader();
      reader.onload = (e) => {
        const content = e.target?.result as string;
        setBibTexContent(content);
        handleParseBibTeX(content);
      };
      reader.readAsText(file);
    } else {
      toast.error('Please upload a valid BibTeX file (.bib)');
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'text/plain': ['.bib'],
      'application/x-bibtex': ['.bib'],
    },
    maxFiles: 1,
  });

  const handleParseBibTeX = useCallback(async (content: string) => {
    if (!content.trim()) return;

    setIsProcessing(true);
    setParseErrors([]);

    try {
      // Parse BibTeX content
      const parsed = parseBibTeX(content);
      
      if (parsed && parsed.length > 0) {
        setParsedEntries(parsed);
        setSelectedEntries(new Set(parsed.map((entry: ParsedEntry) => entry.key)));
        setStep('preview');
        toast.success(`Parsed ${parsed.length} BibTeX entries`);
      } else {
        throw new Error('No valid entries found in BibTeX content');
      }
    } catch (error) {
      console.error('BibTeX parsing error:', error);
      setParseErrors([`Failed to parse BibTeX: ${error}`]);
      toast.error('Failed to parse BibTeX file. Please check the format.');
    } finally {
      setIsProcessing(false);
    }
  }, []);

  const handleTextInput = useCallback((e: React.ChangeEvent<HTMLTextAreaElement>) => {
    const content = e.target.value;
    setBibTexContent(content);
  }, []);

  const handlePasteAndParse = useCallback(() => {
    if (bibTexContent.trim()) {
      handleParseBibTeX(bibTexContent);
    }
  }, [bibTexContent, handleParseBibTeX]);

  const convertBibTeXToPaper = useCallback((entry: ParsedEntry): Paper => {
    // Parse authors
    const authors = entry.author 
      ? entry.author.split(' and ').map(author => ({
          name: author.trim().replace(/[{}]/g, ''),
        }))
      : [];

    // Clean title
    const title = entry.title?.replace(/[{}]/g, '').trim() || 'Untitled';

    // Parse year
    const year = entry.year ? parseInt(entry.year) : undefined;

    return {
      id: `bibtex_${entry.key}_${Date.now()}`,
      title,
      authors,
      journal: entry.journal ? { name: entry.journal.replace(/[{}]/g, '') } : undefined,
      publication_year: year,
      doi: entry.doi,
      url: entry.url,
      abstract: entry.abstract?.replace(/[{}]/g, ''),
      volume: entry.volume,
      issue: entry.number,
      pages: entry.pages,
      citation_count: { total: 0 },
      references: [],
      cited_by: [],
      subjects: [],
      keywords: entry.keywords ? entry.keywords.split(/[,;]/).map(k => k.trim()) : [],
      concepts: [],
      is_open_access: false,
      seed: true, // Mark BibTeX imports as seed papers by default
    };
  }, []);

  const handleImportSelected = useCallback(async () => {
    if (selectedEntries.size === 0) {
      toast.error('Please select at least one entry to import');
      return;
    }

    setLoading('import', true);
    setIsProcessing(true);

    try {
      const entriesToImport = parsedEntries.filter(entry => selectedEntries.has(entry.key));
      const papers = entriesToImport.map(convertBibTeXToPaper);
      
      addPapers(papers);
      makeSeed(papers);
      
      setStep('success');
      toast.success(`Successfully imported ${papers.length} papers as seed papers`);
      
      // Auto close after success
      setTimeout(() => {
        handleClose();
      }, 2000);

    } catch (error) {
      console.error('Import error:', error);
      toast.error('Failed to import papers. Please try again.');
    } finally {
      setLoading('import', false);
      setIsProcessing(false);
    }
  }, [selectedEntries, parsedEntries, convertBibTeXToPaper, addPapers, makeSeed, setLoading]);

  const handleClose = useCallback(() => {
    setBibTexContent('');
    setParsedEntries([]);
    setSelectedEntries(new Set());
    setStep('upload');
    setParseErrors([]);
    onClose();
  }, [onClose]);

  const toggleEntrySelection = useCallback((key: string) => {
    setSelectedEntries(prev => {
      const newSet = new Set(prev);
      if (newSet.has(key)) {
        newSet.delete(key);
      } else {
        newSet.add(key);
      }
      return newSet;
    });
  }, []);

  const selectAll = useCallback(() => {
    setSelectedEntries(new Set(parsedEntries.map(entry => entry.key)));
  }, [parsedEntries]);

  const selectNone = useCallback(() => {
    setSelectedEntries(new Set());
  }, []);

  const formatAuthors = (author?: string) => {
    if (!author) return 'Unknown authors';
    const authors = author.split(' and ');
    if (authors.length === 1) return authors[0].replace(/[{}]/g, '');
    if (authors.length === 2) return `${authors[0]} and ${authors[1]}`.replace(/[{}]/g, '');
    return `${authors[0]} et al.`.replace(/[{}]/g, '');
  };

  return (
    <Modal
      isOpen={isOpen}
      onClose={handleClose}
      title="Import from BibTeX"
      description="Upload or paste BibTeX entries to add papers to your network"
      size="xl"
    >
      <div className="p-6">
        {/* Upload Step */}
        {step === 'upload' && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="space-y-6"
          >
            {/* File Upload */}
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-3">
                Upload BibTeX File
              </label>
              <div
                {...getRootProps()}
                className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors ${
                  isDragActive
                    ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20'
                    : 'border-gray-300 dark:border-gray-600 hover:border-blue-400 dark:hover:border-blue-500'
                }`}
              >
                <input {...getInputProps()} />
                <div className="space-y-3">
                  <Upload className="h-12 w-12 mx-auto text-gray-400" />
                  {isDragActive ? (
                    <p className="text-blue-600 dark:text-blue-400">
                      Drop the BibTeX file here...
                    </p>
                  ) : (
                    <>
                      <p className="text-gray-900 dark:text-white font-medium">
                        Drop a BibTeX file here, or click to select
                      </p>
                      <p className="text-sm text-gray-500 dark:text-gray-400">
                        Supports .bib files up to 10MB
                      </p>
                    </>
                  )}
                </div>
              </div>
            </div>

            {/* Divider */}
            <div className="relative">
              <div className="absolute inset-0 flex items-center">
                <div className="w-full border-t border-gray-300 dark:border-gray-600" />
              </div>
              <div className="relative flex justify-center text-sm">
                <span className="px-2 bg-white dark:bg-gray-900 text-gray-500 dark:text-gray-400">
                  or paste BibTeX content
                </span>
              </div>
            </div>

            {/* Text Input */}
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                BibTeX Content
              </label>
              <textarea
                value={bibTexContent}
                onChange={handleTextInput}
                placeholder="Paste your BibTeX entries here..."
                rows={12}
                className="w-full px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg 
                         bg-white dark:bg-gray-800 text-gray-900 dark:text-white font-mono text-sm
                         focus:ring-2 focus:ring-blue-500 focus:border-transparent
                         placeholder-gray-500 dark:placeholder-gray-400"
              />
            </div>

            {/* Parse Errors */}
            {parseErrors.length > 0 && (
              <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-4">
                <div className="flex items-start">
                  <AlertCircle className="h-5 w-5 text-red-500 mr-2 flex-shrink-0 mt-0.5" />
                  <div>
                    <h4 className="text-sm font-medium text-red-900 dark:text-red-100">
                      Parsing Errors
                    </h4>
                    <ul className="mt-2 text-sm text-red-700 dark:text-red-200 space-y-1">
                      {parseErrors.map((error, index) => (
                        <li key={index}>• {error}</li>
                      ))}
                    </ul>
                  </div>
                </div>
              </div>
            )}

            {/* Actions */}
            <div className="flex space-x-3">
              <Button
                onClick={handlePasteAndParse}
                disabled={!bibTexContent.trim() || isProcessing}
                className="flex-1"
              >
                {isProcessing ? (
                  <>
                    <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                    Processing...
                  </>
                ) : (
                  <>
                    <Eye className="h-4 w-4 mr-2" />
                    Parse & Preview
                  </>
                )}
              </Button>
            </div>

            {/* Sample BibTeX */}
            <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-4">
              <h4 className="text-sm font-medium text-blue-900 dark:text-blue-100 mb-2">
                Sample BibTeX Format:
              </h4>
              <pre className="text-xs text-blue-700 dark:text-blue-200 font-mono overflow-x-auto">
{`@article{sample2024,
  title={Sample Research Paper},
  author={Smith, John and Doe, Jane},
  journal={Nature},
  year={2024},
  volume={123},
  pages={1-10},
  doi={10.1000/sample}
}`}
              </pre>
            </div>
          </motion.div>
        )}

        {/* Preview Step */}
        {step === 'preview' && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="space-y-4"
          >
            {/* Header */}
            <div className="flex items-center justify-between">
              <h3 className="font-semibold text-gray-900 dark:text-white">
                BibTeX Entries ({parsedEntries.length})
              </h3>
              <div className="flex items-center space-x-2">
                <button
                  onClick={selectAll}
                  className="text-sm text-blue-600 dark:text-blue-400 hover:underline"
                >
                  Select All
                </button>
                <span className="text-gray-400">|</span>
                <button
                  onClick={selectNone}
                  className="text-sm text-blue-600 dark:text-blue-400 hover:underline"
                >
                  Select None
                </button>
              </div>
            </div>

            {/* Selection Summary */}
            <div className="bg-gray-50 dark:bg-gray-800 rounded-lg p-3">
              <div className="text-sm text-gray-600 dark:text-gray-400">
                <strong>{selectedEntries.size}</strong> of <strong>{parsedEntries.length}</strong> entries selected
              </div>
            </div>

            {/* Entries List */}
            <div className="max-h-96 overflow-y-auto space-y-3">
              {parsedEntries.map((entry, index) => (
                <motion.div
                  key={entry.key}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: index * 0.05 }}
                  className={`p-4 border rounded-lg cursor-pointer transition-colors ${
                    selectedEntries.has(entry.key)
                      ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20'
                      : 'border-gray-200 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-800/50'
                  }`}
                  onClick={() => toggleEntrySelection(entry.key)}
                >
                  <div className="flex items-start space-x-3">
                    <div className="flex-shrink-0 pt-1">
                      {selectedEntries.has(entry.key) ? (
                        <Check className="h-5 w-5 text-blue-500" />
                      ) : (
                        <div className="h-5 w-5 border border-gray-300 dark:border-gray-600 rounded" />
                      )}
                    </div>
                    
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center space-x-2 mb-2">
                        <span className="text-xs bg-gray-200 dark:bg-gray-700 px-2 py-1 rounded font-mono">
                          {entry.type}
                        </span>
                        <span className="text-xs text-gray-500 dark:text-gray-400">
                          Key: {entry.key}
                        </span>
                      </div>
                      
                      <h4 className="font-medium text-gray-900 dark:text-white mb-1 leading-tight">
                        {entry.title?.replace(/[{}]/g, '') || 'Untitled'}
                      </h4>
                      
                      {entry.author && (
                        <p className="text-sm text-gray-600 dark:text-gray-400 mb-1">
                          {formatAuthors(entry.author)}
                        </p>
                      )}
                      
                      <div className="flex items-center space-x-4 text-sm text-gray-500 dark:text-gray-400">
                        {entry.journal && (
                          <span>{entry.journal.replace(/[{}]/g, '')}</span>
                        )}
                        {entry.year && <span>{entry.year}</span>}
                        {entry.doi && (
                          <span className="font-mono text-xs">{entry.doi}</span>
                        )}
                      </div>
                    </div>
                  </div>
                </motion.div>
              ))}
            </div>

            {/* Actions */}
            <div className="flex space-x-3 pt-4 border-t border-gray-200 dark:border-gray-700">
              <Button
                variant="outline"
                onClick={() => setStep('upload')}
                disabled={isProcessing}
              >
                Back
              </Button>
              <Button
                onClick={handleImportSelected}
                disabled={selectedEntries.size === 0 || isProcessing}
                className="flex-1"
              >
                {isProcessing ? (
                  <>
                    <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                    Importing...
                  </>
                ) : (
                  <>
                    <Plus className="h-4 w-4 mr-2" />
                    Import {selectedEntries.size} Papers as Seeds
                  </>
                )}
              </Button>
            </div>
          </motion.div>
        )}

        {/* Success Step */}
        {step === 'success' && (
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            className="text-center py-8"
          >
            <div className="w-16 h-16 bg-green-100 dark:bg-green-900/20 rounded-full flex items-center justify-center mx-auto mb-4">
              <Check className="h-8 w-8 text-green-600 dark:text-green-400" />
            </div>
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">
              Import Successful!
            </h3>
            <p className="text-gray-600 dark:text-gray-400 mb-4">
              {selectedEntries.size} papers have been imported as seed papers and added to your network.
            </p>
            <Button onClick={handleClose}>
              Continue
            </Button>
          </motion.div>
        )}
      </div>
    </Modal>
  );
};

export default BibTeXUploadModal;