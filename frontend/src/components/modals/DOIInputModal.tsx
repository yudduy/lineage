import React, { useState, useCallback } from 'react';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';
import { motion } from 'framer-motion';
import { Search, Plus, AlertCircle, CheckCircle, ExternalLink } from 'lucide-react';
import toast from 'react-hot-toast';
import Modal from './Modal';
import { useUiStore, usePaperStore } from '@store/index';
import { PaperService } from '@services/paperService';
import Button from '@components/ui/Button';
import LoadingOverlay from '@components/ui/LoadingOverlay';

// DOI validation schema
const doiSchema = z.object({
  doi: z.string().min(1, 'DOI is required').refine(
    (val) => {
      // Basic DOI format validation
      const doiRegex = /^10\.\d{4,}\/[-._;()\/:a-zA-Z0-9]+$/;
      return doiRegex.test(val) || val.startsWith('http');
    },
    'Please enter a valid DOI (e.g., 10.1000/182 or URL)'
  ),
});

type DOIFormData = z.infer<typeof doiSchema>;

interface DOIInputModalProps {
  isOpen: boolean;
  onClose: () => void;
}

const DOIInputModal: React.FC<DOIInputModalProps> = ({ isOpen, onClose }) => {
  const [isLoading, setIsLoading] = useState(false);
  const [previewPaper, setPreviewPaper] = useState<any>(null);
  const [step, setStep] = useState<'input' | 'preview' | 'success'>('input');
  
  const { addPapers, makeSeed } = usePaperStore();
  const { setLoading } = useUiStore();

  const {
    register,
    handleSubmit,
    formState: { errors, isSubmitting },
    reset,
    watch,
    setValue,
  } = useForm<DOIFormData>({
    resolver: zodResolver(doiSchema),
  });

  const doiValue = watch('doi');

  const extractDOI = useCallback((input: string): string => {
    // Clean and extract DOI from various formats
    const cleanInput = input.trim();
    
    // If it's a URL, try to extract DOI
    if (cleanInput.startsWith('http')) {
      const doiMatch = cleanInput.match(/10\.\d{4,}\/[-._;()\/:a-zA-Z0-9]+/);
      return doiMatch ? doiMatch[0] : cleanInput;
    }
    
    // Remove common prefixes
    return cleanInput.replace(/^(doi:|DOI:)\s*/i, '');
  }, []);

  const handlePreviewDOI = useCallback(async () => {
    if (!doiValue) return;
    
    setIsLoading(true);
    try {
      const cleanDOI = extractDOI(doiValue);
      const paper = await PaperService.getByDOI(cleanDOI);
      
      if (paper) {
        setPreviewPaper(paper);
        setStep('preview');
      } else {
        toast.error('Paper not found. Please check the DOI and try again.');
      }
    } catch (error) {
      console.error('Error previewing DOI:', error);
      toast.error('Failed to fetch paper. Please check the DOI and try again.');
    } finally {
      setIsLoading(false);
    }
  }, [doiValue, extractDOI]);

  const onSubmit = async (data: DOIFormData) => {
    setLoading('papers', true);
    try {
      const cleanDOI = extractDOI(data.doi);
      
      let paper;
      if (previewPaper) {
        paper = previewPaper;
      } else {
        paper = await PaperService.getByDOI(cleanDOI);
      }

      if (paper) {
        // Add as seed paper
        const paperWithSeed = { ...paper, seed: true };
        addPapers([paperWithSeed]);
        makeSeed([paperWithSeed]);
        
        setStep('success');
        
        toast.success(`Added "${paper.title}" as seed paper`);
        
        // Auto close after success
        setTimeout(() => {
          handleClose();
        }, 2000);
      } else {
        toast.error('Paper not found. Please check the DOI and try again.');
      }
    } catch (error) {
      console.error('Error adding paper by DOI:', error);
      toast.error('Failed to add paper. Please try again.');
    } finally {
      setLoading('papers', false);
    }
  };

  const handleClose = useCallback(() => {
    reset();
    setPreviewPaper(null);
    setStep('input');
    setIsLoading(false);
    onClose();
  }, [reset, onClose]);

  const handleBack = useCallback(() => {
    setPreviewPaper(null);
    setStep('input');
  }, []);

  const formatAuthors = (authors: any[]) => {
    if (!authors || authors.length === 0) return 'Unknown authors';
    if (authors.length === 1) return authors[0].name;
    if (authors.length === 2) return `${authors[0].name} and ${authors[1].name}`;
    return `${authors[0].name} et al.`;
  };

  return (
    <Modal
      isOpen={isOpen}
      onClose={handleClose}
      title="Add Paper by DOI"
      description="Enter a DOI to add a paper to your network"
      size="lg"
    >
      <div className="p-6">
        {isLoading && <LoadingOverlay />}
        
        {step === 'input' && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="space-y-4"
          >
            <form onSubmit={handleSubmit(onSubmit)} className="space-y-4">
              <div>
                <label htmlFor="doi" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  DOI or Paper URL
                </label>
                <div className="relative">
                  <input
                    {...register('doi')}
                    type="text"
                    placeholder="e.g., 10.1000/182 or https://doi.org/10.1000/182"
                    className="w-full px-4 py-3 pr-12 border border-gray-300 dark:border-gray-600 rounded-lg 
                             bg-white dark:bg-gray-800 text-gray-900 dark:text-white
                             focus:ring-2 focus:ring-blue-500 focus:border-transparent
                             placeholder-gray-500 dark:placeholder-gray-400"
                  />
                  <Search className="absolute right-3 top-3.5 h-5 w-5 text-gray-400" />
                </div>
                {errors.doi && (
                  <p className="mt-1 text-sm text-red-600 dark:text-red-400 flex items-center">
                    <AlertCircle className="h-4 w-4 mr-1" />
                    {errors.doi.message}
                  </p>
                )}
              </div>

              <div className="flex space-x-3">
                <Button
                  type="button"
                  variant="outline"
                  onClick={handlePreviewDOI}
                  disabled={!doiValue || !!errors.doi || isLoading}
                  className="flex-1"
                >
                  <Search className="h-4 w-4 mr-2" />
                  Preview
                </Button>
                <Button
                  type="submit"
                  disabled={isSubmitting || !doiValue || !!errors.doi}
                  className="flex-1"
                >
                  <Plus className="h-4 w-4 mr-2" />
                  Add Paper
                </Button>
              </div>
            </form>

            <div className="mt-6 p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
              <h4 className="text-sm font-medium text-blue-900 dark:text-blue-100 mb-2">
                Supported formats:
              </h4>
              <ul className="text-sm text-blue-700 dark:text-blue-200 space-y-1">
                <li>• DOI: 10.1000/182</li>
                <li>• DOI URL: https://doi.org/10.1000/182</li>
                <li>• Publisher URL with DOI</li>
              </ul>
            </div>
          </motion.div>
        )}

        {step === 'preview' && previewPaper && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="space-y-4"
          >
            <div className="bg-gray-50 dark:bg-gray-800 rounded-lg p-4">
              <h3 className="font-semibold text-gray-900 dark:text-white mb-2">
                Paper Preview
              </h3>
              
              <div className="space-y-3">
                <div>
                  <h4 className="font-medium text-gray-900 dark:text-white leading-snug">
                    {previewPaper.title}
                  </h4>
                </div>
                
                <div className="text-sm text-gray-600 dark:text-gray-400">
                  <p><strong>Authors:</strong> {formatAuthors(previewPaper.authors)}</p>
                  {previewPaper.journal?.name && (
                    <p><strong>Journal:</strong> {previewPaper.journal.name}</p>
                  )}
                  {previewPaper.publication_year && (
                    <p><strong>Year:</strong> {previewPaper.publication_year}</p>
                  )}
                  {previewPaper.citation_count?.total !== undefined && (
                    <p><strong>Citations:</strong> {previewPaper.citation_count.total}</p>
                  )}
                </div>

                {previewPaper.abstract && (
                  <div className="text-sm text-gray-700 dark:text-gray-300">
                    <p><strong>Abstract:</strong></p>
                    <p className="mt-1 line-clamp-3">{previewPaper.abstract}</p>
                  </div>
                )}

                {previewPaper.url && (
                  <a
                    href={previewPaper.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="inline-flex items-center text-sm text-blue-600 dark:text-blue-400 hover:underline"
                  >
                    <ExternalLink className="h-4 w-4 mr-1" />
                    View original paper
                  </a>
                )}
              </div>
            </div>

            <div className="flex space-x-3">
              <Button
                variant="outline"
                onClick={handleBack}
                className="flex-1"
              >
                Back
              </Button>
              <Button
                onClick={() => onSubmit({ doi: doiValue })}
                disabled={isSubmitting}
                className="flex-1"
              >
                <Plus className="h-4 w-4 mr-2" />
                Add as Seed Paper
              </Button>
            </div>
          </motion.div>
        )}

        {step === 'success' && (
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            className="text-center py-8"
          >
            <div className="w-16 h-16 bg-green-100 dark:bg-green-900/20 rounded-full flex items-center justify-center mx-auto mb-4">
              <CheckCircle className="h-8 w-8 text-green-600 dark:text-green-400" />
            </div>
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">
              Paper Added Successfully!
            </h3>
            <p className="text-gray-600 dark:text-gray-400">
              The paper has been added to your network as a seed paper.
            </p>
          </motion.div>
        )}
      </div>
    </Modal>
  );
};

export default DOIInputModal;