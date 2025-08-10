import React, { useState, useCallback } from 'react';
import { motion } from 'framer-motion';
import { Download, Image, Code, FileText, Database, CheckCircle, AlertCircle } from 'lucide-react';
import toast from 'react-hot-toast';
import Modal from './Modal';
import { useUiStore, usePaperStore } from '@store/index';
import { ExportOptions } from '@types/paper';
import Button from '@components/ui/Button';
import { saveAs } from 'file-saver';

interface ExportModalProps {
  isOpen: boolean;
  onClose: () => void;
}

const exportFormats = [
  {
    id: 'png',
    name: 'PNG Image',
    description: 'High-quality image of the network visualization',
    icon: Image,
    color: 'text-blue-600 dark:text-blue-400',
    bgColor: 'bg-blue-50 dark:bg-blue-900/20',
  },
  {
    id: 'svg',
    name: 'SVG Vector',
    description: 'Scalable vector graphics format',
    icon: Image,
    color: 'text-green-600 dark:text-green-400',
    bgColor: 'bg-green-50 dark:bg-green-900/20',
  },
  {
    id: 'json',
    name: 'JSON Data',
    description: 'Machine-readable data format with all metadata',
    icon: Code,
    color: 'text-purple-600 dark:text-purple-400',
    bgColor: 'bg-purple-50 dark:bg-purple-900/20',
  },
  {
    id: 'graphml',
    name: 'GraphML',
    description: 'Standard graph format for network analysis tools',
    icon: Database,
    color: 'text-orange-600 dark:text-orange-400',
    bgColor: 'bg-orange-50 dark:bg-orange-900/20',
  },
  {
    id: 'csv',
    name: 'CSV Spreadsheet',
    description: 'Tabular data format for Excel and other tools',
    icon: FileText,
    color: 'text-teal-600 dark:text-teal-400',
    bgColor: 'bg-teal-50 dark:bg-teal-900/20',
  },
  {
    id: 'bibtex',
    name: 'BibTeX',
    description: 'Bibliography format for LaTeX and reference managers',
    icon: FileText,
    color: 'text-red-600 dark:text-red-400',
    bgColor: 'bg-red-50 dark:bg-red-900/20',
  },
];

const ExportModal: React.FC<ExportModalProps> = ({ isOpen, onClose }) => {
  const [selectedFormat, setSelectedFormat] = useState<ExportOptions['format']>('png');
  const [options, setOptions] = useState<Omit<ExportOptions, 'format'>>({
    quality: 'high',
    includeMetadata: true,
    selectedOnly: false,
    transparent: false,
  });
  const [isExporting, setIsExporting] = useState(false);
  const [exportProgress, setExportProgress] = useState(0);

  const { graphNodes, graphLinks, selectedPapers, papers } = usePaperStore();
  const { setLoading } = useUiStore();

  const getDataToExport = useCallback(() => {
    if (options.selectedOnly && selectedPapers.length > 0) {
      const selectedIds = new Set(selectedPapers.map(p => p.id));
      return {
        nodes: graphNodes.filter(n => selectedIds.has(n.id)),
        links: graphLinks.filter(l => 
          selectedIds.has(typeof l.source === 'string' ? l.source : l.source.id!) &&
          selectedIds.has(typeof l.target === 'string' ? l.target : l.target.id!)
        ),
        papers: selectedPapers,
      };
    }
    
    return {
      nodes: graphNodes,
      links: graphLinks,
      papers: papers,
    };
  }, [options.selectedOnly, selectedPapers, graphNodes, graphLinks, papers]);

  const generateFileName = useCallback(() => {
    const timestamp = new Date().toISOString().slice(0, 19).replace(/:/g, '-');
    const scope = options.selectedOnly ? 'selected' : 'all';
    return `citation-network-${scope}-${timestamp}.${selectedFormat}`;
  }, [selectedFormat, options.selectedOnly]);

  const exportAsPNG = useCallback(async (data: any): Promise<Blob> => {
    // Implementation would capture the network visualization as PNG
    // This is a placeholder - actual implementation would use canvas capture
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    
    const width = options.quality === 'high' ? 2048 : options.quality === 'medium' ? 1024 : 512;
    const height = Math.round(width * 0.75);
    
    canvas.width = width;
    canvas.height = height;
    
    if (ctx) {
      // Set background
      if (!options.transparent) {
        ctx.fillStyle = '#ffffff';
        ctx.fillRect(0, 0, width, height);
      }
      
      // Placeholder visualization
      ctx.fillStyle = '#3b82f6';
      ctx.font = '24px Arial';
      ctx.textAlign = 'center';
      ctx.fillText('Network Visualization Export', width / 2, height / 2);
      ctx.fillText(`${data.nodes.length} nodes, ${data.links.length} links`, width / 2, height / 2 + 40);
    }
    
    return new Promise((resolve) => {
      canvas.toBlob((blob) => resolve(blob!), 'image/png');
    });
  }, [options]);

  const exportAsSVG = useCallback(async (data: any): Promise<Blob> => {
    const width = 800;
    const height = 600;
    
    const svg = `
      <svg width="${width}" height="${height}" xmlns="http://www.w3.org/2000/svg">
        <rect width="100%" height="100%" fill="${options.transparent ? 'transparent' : 'white'}"/>
        <text x="50%" y="50%" text-anchor="middle" font-family="Arial" font-size="24" fill="#333">
          Network Visualization Export
        </text>
        <text x="50%" y="60%" text-anchor="middle" font-family="Arial" font-size="16" fill="#666">
          ${data.nodes.length} nodes, ${data.links.length} links
        </text>
      </svg>
    `;
    
    return new Blob([svg], { type: 'image/svg+xml' });
  }, [options]);

  const exportAsJSON = useCallback(async (data: any): Promise<Blob> => {
    const exportData = {
      metadata: {
        exportedAt: new Date().toISOString(),
        version: '1.0',
        format: 'citation-network-explorer',
        scope: options.selectedOnly ? 'selected' : 'all',
        includeMetadata: options.includeMetadata,
      },
      nodes: data.nodes,
      links: data.links,
      ...(options.includeMetadata && {
        papers: data.papers,
        statistics: {
          nodeCount: data.nodes.length,
          linkCount: data.links.length,
          seedCount: data.nodes.filter((n: any) => n.seed).length,
        },
      }),
    };
    
    return new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
  }, [options]);

  const exportAsGraphML = useCallback(async (data: any): Promise<Blob> => {
    const graphml = `<?xml version="1.0" encoding="UTF-8"?>
<graphml xmlns="http://graphml.graphdrawing.org/xmlns"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns
         http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd">
  
  <key id="title" for="node" attr.name="title" attr.type="string"/>
  <key id="authors" for="node" attr.name="authors" attr.type="string"/>
  <key id="year" for="node" attr.name="publication_year" attr.type="int"/>
  <key id="citations" for="node" attr.name="citation_count" attr.type="int"/>
  <key id="seed" for="node" attr.name="seed" attr.type="boolean"/>
  <key id="weight" for="edge" attr.name="weight" attr.type="double"/>
  
  <graph id="citation_network" edgedefault="directed">
    ${data.nodes.map((node: any) => `
    <node id="${node.id}">
      <data key="title">${node.title || 'Untitled'}</data>
      <data key="authors">${node.authors?.map((a: any) => a.name).join('; ') || ''}</data>
      <data key="year">${node.publication_year || ''}</data>
      <data key="citations">${node.citation_count?.total || 0}</data>
      <data key="seed">${node.seed || false}</data>
    </node>`).join('')}
    
    ${data.links.map((link: any, index: number) => `
    <edge id="e${index}" source="${typeof link.source === 'string' ? link.source : link.source.id}" 
          target="${typeof link.target === 'string' ? link.target : link.target.id}">
      <data key="weight">${link.weight || 1}</data>
    </edge>`).join('')}
  </graph>
</graphml>`;
    
    return new Blob([graphml], { type: 'application/xml' });
  }, []);

  const exportAsCSV = useCallback(async (data: any): Promise<Blob> => {
    const headers = [
      'id', 'title', 'authors', 'journal', 'publication_year', 
      'citation_count', 'doi', 'is_seed', 'subjects'
    ];
    
    const rows = data.papers.map((paper: any) => [
      paper.id || '',
      `"${(paper.title || '').replace(/"/g, '""')}"`,
      `"${paper.authors?.map((a: any) => a.name).join('; ') || ''}"`,
      `"${paper.journal?.name || ''}"`,
      paper.publication_year || '',
      paper.citation_count?.total || 0,
      paper.doi || '',
      paper.seed || false,
      `"${paper.subjects?.join('; ') || ''}"`,
    ]);
    
    const csv = [headers.join(','), ...rows.map(row => row.join(','))].join('\n');
    return new Blob([csv], { type: 'text/csv' });
  }, []);

  const exportAsBibTeX = useCallback(async (data: any): Promise<Blob> => {
    const bibtex = data.papers.map((paper: any, index: number) => {
      const key = paper.doi?.replace(/[^\w]/g, '') || `paper${index + 1}`;
      const authors = paper.authors?.map((a: any) => a.name).join(' and ') || '';
      
      return `@article{${key},
  title={${paper.title || 'Untitled'}},
  author={${authors}},
  journal={${paper.journal?.name || ''}},
  year={${paper.publication_year || ''}},
  doi={${paper.doi || ''}},
  url={${paper.url || ''}},
  note={Citation count: ${paper.citation_count?.total || 0}}
}`;
    }).join('\n\n');
    
    return new Blob([bibtex], { type: 'text/plain' });
  }, []);

  const handleExport = useCallback(async () => {
    setIsExporting(true);
    setExportProgress(0);
    setLoading('export', true);
    
    try {
      const data = getDataToExport();
      let blob: Blob;
      
      setExportProgress(25);
      
      switch (selectedFormat) {
        case 'png':
          blob = await exportAsPNG(data);
          break;
        case 'svg':
          blob = await exportAsSVG(data);
          break;
        case 'json':
          blob = await exportAsJSON(data);
          break;
        case 'graphml':
          blob = await exportAsGraphML(data);
          break;
        case 'csv':
          blob = await exportAsCSV(data);
          break;
        case 'bibtex':
          blob = await exportAsBibTeX(data);
          break;
        default:
          throw new Error(`Unsupported export format: ${selectedFormat}`);
      }
      
      setExportProgress(75);
      
      const fileName = generateFileName();
      saveAs(blob, fileName);
      
      setExportProgress(100);
      
      toast.success(`Successfully exported as ${selectedFormat.toUpperCase()}`);
      
      // Auto close after success
      setTimeout(() => {
        onClose();
      }, 1500);
      
    } catch (error) {
      console.error('Export error:', error);
      toast.error('Failed to export. Please try again.');
    } finally {
      setIsExporting(false);
      setExportProgress(0);
      setLoading('export', false);
    }
  }, [selectedFormat, options, getDataToExport, generateFileName, exportAsPNG, exportAsSVG, 
      exportAsJSON, exportAsGraphML, exportAsCSV, exportAsBibTeX, setLoading, onClose]);

  const selectedFormatInfo = exportFormats.find(f => f.id === selectedFormat);

  return (
    <Modal
      isOpen={isOpen}
      onClose={onClose}
      title="Export Network"
      description="Choose a format to export your citation network"
      size="lg"
    >
      <div className="p-6 space-y-6">
        {/* Format Selection */}
        <div>
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
            Export Format
          </h3>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            {exportFormats.map((format) => (
              <button
                key={format.id}
                onClick={() => setSelectedFormat(format.id as ExportOptions['format'])}
                className={`p-4 rounded-lg border-2 transition-all text-left ${
                  selectedFormat === format.id
                    ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20'
                    : 'border-gray-200 dark:border-gray-700 hover:border-gray-300 dark:hover:border-gray-600'
                }`}
              >
                <div className="flex items-start space-x-3">
                  <div className={`p-2 rounded-lg ${format.bgColor}`}>
                    <format.icon className={`h-5 w-5 ${format.color}`} />
                  </div>
                  <div className="flex-1 min-w-0">
                    <h4 className="font-medium text-gray-900 dark:text-white">
                      {format.name}
                    </h4>
                    <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                      {format.description}
                    </p>
                  </div>
                  {selectedFormat === format.id && (
                    <CheckCircle className="h-5 w-5 text-blue-500 flex-shrink-0" />
                  )}
                </div>
              </button>
            ))}
          </div>
        </div>

        {/* Export Options */}
        <div>
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
            Export Options
          </h3>
          <div className="space-y-4">
            {/* Quality (for image formats) */}
            {(selectedFormat === 'png' || selectedFormat === 'svg') && (
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Quality
                </label>
                <div className="flex space-x-4">
                  {(['low', 'medium', 'high'] as const).map((quality) => (
                    <label key={quality} className="flex items-center">
                      <input
                        type="radio"
                        name="quality"
                        value={quality}
                        checked={options.quality === quality}
                        onChange={(e) => setOptions(prev => ({ 
                          ...prev, 
                          quality: e.target.value as ExportOptions['quality']
                        }))}
                        className="mr-2"
                      />
                      <span className="text-sm text-gray-700 dark:text-gray-300 capitalize">
                        {quality}
                      </span>
                    </label>
                  ))}
                </div>
              </div>
            )}

            {/* Include Metadata */}
            <label className="flex items-center">
              <input
                type="checkbox"
                checked={options.includeMetadata}
                onChange={(e) => setOptions(prev => ({ 
                  ...prev, 
                  includeMetadata: e.target.checked 
                }))}
                className="mr-2"
              />
              <span className="text-sm text-gray-700 dark:text-gray-300">
                Include detailed metadata
              </span>
            </label>

            {/* Selected Only */}
            <label className="flex items-center">
              <input
                type="checkbox"
                checked={options.selectedOnly}
                onChange={(e) => setOptions(prev => ({ 
                  ...prev, 
                  selectedOnly: e.target.checked 
                }))}
                className="mr-2"
              />
              <span className="text-sm text-gray-700 dark:text-gray-300">
                Export selected papers only ({selectedPapers.length} selected)
              </span>
            </label>

            {/* Transparent Background (for image formats) */}
            {(selectedFormat === 'png' || selectedFormat === 'svg') && (
              <label className="flex items-center">
                <input
                  type="checkbox"
                  checked={options.transparent}
                  onChange={(e) => setOptions(prev => ({ 
                    ...prev, 
                    transparent: e.target.checked 
                  }))}
                  className="mr-2"
                />
                <span className="text-sm text-gray-700 dark:text-gray-300">
                  Transparent background
                </span>
              </label>
            )}
          </div>
        </div>

        {/* Export Progress */}
        {isExporting && (
          <div className="bg-gray-50 dark:bg-gray-800 rounded-lg p-4">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                Exporting...
              </span>
              <span className="text-sm text-gray-500 dark:text-gray-400">
                {exportProgress}%
              </span>
            </div>
            <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
              <div
                className="bg-blue-500 h-2 rounded-full transition-all duration-300"
                style={{ width: `${exportProgress}%` }}
              />
            </div>
          </div>
        )}

        {/* Export Summary */}
        <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-4">
          <h4 className="font-medium text-blue-900 dark:text-blue-100 mb-2">
            Export Summary
          </h4>
          <div className="text-sm text-blue-700 dark:text-blue-200 space-y-1">
            <p>• Format: {selectedFormatInfo?.name}</p>
            <p>• Scope: {options.selectedOnly ? `${selectedPapers.length} selected papers` : `${papers.length} total papers`}</p>
            <p>• Nodes: {options.selectedOnly ? selectedPapers.length : graphNodes.length}</p>
            <p>• Links: {graphLinks.length}</p>
          </div>
        </div>

        {/* Actions */}
        <div className="flex space-x-3 pt-4 border-t border-gray-200 dark:border-gray-700">
          <Button
            variant="outline"
            onClick={onClose}
            disabled={isExporting}
            className="flex-1"
          >
            Cancel
          </Button>
          <Button
            onClick={handleExport}
            disabled={isExporting}
            className="flex-1"
          >
            <Download className="h-4 w-4 mr-2" />
            {isExporting ? 'Exporting...' : 'Export'}
          </Button>
        </div>
      </div>
    </Modal>
  );
};

export default ExportModal;