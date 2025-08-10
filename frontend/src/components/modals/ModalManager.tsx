import React from 'react';
import { useUiStore } from '@store/index';
import DOIInputModal from './DOIInputModal';
import AdvancedSearchModal from './AdvancedSearchModal';
import BibTeXUploadModal from './BibTeXUploadModal';
import ExportModal from './ExportModal';
// Import future modals as they are created
// import ZoteroImportModal from './ZoteroImportModal';
// import PaperDetailsModal from './PaperDetailsModal';
// import ResearchIntelligenceModal from './ResearchIntelligenceModal';
// import CommunityAnalysisModal from './CommunityAnalysisModal';
// import CitationFlowModal from './CitationFlowModal';
// import HelpModal from './HelpModal';
// import SettingsModal from './SettingsModal';

/**
 * ModalManager component that handles all modal states and rendering
 * Centralized modal management to avoid prop drilling and ensure consistent behavior
 */
const ModalManager: React.FC = () => {
  const { modals, closeModal } = useUiStore();

  return (
    <>
      {/* DOI Input Modal */}
      <DOIInputModal
        isOpen={modals.doiInput}
        onClose={() => closeModal('doiInput')}
      />

      {/* Advanced Search Modal */}
      <AdvancedSearchModal
        isOpen={modals.titleSearch}
        onClose={() => closeModal('titleSearch')}
      />

      {/* BibTeX Upload Modal */}
      <BibTeXUploadModal
        isOpen={modals.bibtexUpload}
        onClose={() => closeModal('bibtexUpload')}
      />

      {/* Export Modal */}
      <ExportModal
        isOpen={modals.export}
        onClose={() => closeModal('export')}
      />

      {/* Future Modals - Uncomment as they are implemented */}
      {/*
      <ZoteroImportModal
        isOpen={modals.zoteroImport}
        onClose={() => closeModal('zoteroImport')}
      />

      <PaperDetailsModal
        isOpen={modals.paperDetails}
        onClose={() => closeModal('paperDetails')}
      />

      <ResearchIntelligenceModal
        isOpen={modals.researchIntelligence}
        onClose={() => closeModal('researchIntelligence')}
      />

      <CommunityAnalysisModal
        isOpen={modals.communityAnalysis}
        onClose={() => closeModal('communityAnalysis')}
      />

      <CitationFlowModal
        isOpen={modals.citationFlow}
        onClose={() => closeModal('citationFlow')}
      />

      <HelpModal
        isOpen={modals.help}
        onClose={() => closeModal('help')}
      />

      <SettingsModal
        isOpen={modals.settings}
        onClose={() => closeModal('settings')}
      />
      */}
    </>
  );
};

export default ModalManager;