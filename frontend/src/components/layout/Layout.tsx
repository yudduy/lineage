import React from 'react';
import { useUiStore, usePaperStore } from '@store/index';
import Header from './Header';
import Sidebar from './Sidebar';
import { clsx } from 'clsx';

interface LayoutProps {
  children: React.ReactNode;
}

const Layout: React.FC<LayoutProps> = ({ children }) => {
  const { isSidebarOpen } = useUiStore();
  const { seedPapers } = usePaperStore();

  return (
    <div className="flex h-screen bg-background">
      {/* Sidebar */}
      <div
        className={clsx(
          'transition-all duration-300 ease-in-out bg-surface border-r border-border',
          'flex-shrink-0 overflow-hidden',
          {
            'w-80': isSidebarOpen,
            'w-0': !isSidebarOpen,
          }
        )}
      >
        {isSidebarOpen && <Sidebar />}
      </div>

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Header */}
        <Header />

        {/* Main Content */}
        <main className="flex-1 overflow-hidden relative">
          {seedPapers.length === 0 ? (
            <WelcomeScreen />
          ) : (
            children
          )}
        </main>
      </div>
    </div>
  );
};

// Welcome screen for when no papers are loaded
const WelcomeScreen: React.FC = () => {
  const { openModal } = useUiStore();

  return (
    <div className="flex items-center justify-center h-full bg-background">
      <div className="max-w-md text-center px-6">
        {/* Icon */}
        <div className="w-20 h-20 bg-primary-100 rounded-full flex items-center justify-center mx-auto mb-6">
          <svg
            className="w-10 h-10 text-primary-600"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={1.5}
              d="M13.828 10.172a4 4 0 00-5.656 0l-4 4a4 4 0 105.656 5.656l1.102-1.101m-.758-4.899a4 4 0 005.656 0l4-4a4 4 0 00-5.656-5.656l-1.1 1.1"
            />
          </svg>
        </div>

        {/* Welcome Message */}
        <h1 className="text-2xl font-bold text-text mb-4">
          Welcome to Citation Network Explorer
        </h1>
        
        <p className="text-text-secondary mb-8 leading-relaxed">
          Start by adding some papers to explore their citation networks. 
          You can search by DOI, title, upload BibTeX files, or import from Zotero.
        </p>

        {/* Action Buttons */}
        <div className="space-y-3">
          <button
            onClick={() => openModal('addPaper')}
            className="w-full bg-primary-600 text-white px-6 py-3 rounded-lg font-medium hover:bg-primary-700 transition-colors"
          >
            Add Your First Paper
          </button>
          
          <button
            onClick={() => openModal('help')}
            className="w-full border border-border text-text px-6 py-3 rounded-lg font-medium hover:bg-surface transition-colors"
          >
            Learn How It Works
          </button>
        </div>

        {/* Features */}
        <div className="mt-12 grid grid-cols-1 gap-4 text-sm">
          <div className="flex items-start space-x-3">
            <div className="w-5 h-5 bg-green-100 rounded-full flex items-center justify-center mt-0.5">
              <svg className="w-3 h-3 text-green-600" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
              </svg>
            </div>
            <div>
              <p className="font-medium text-text">Interactive Visualization</p>
              <p className="text-text-secondary">Explore citation networks in 2D or 3D</p>
            </div>
          </div>
          
          <div className="flex items-start space-x-3">
            <div className="w-5 h-5 bg-blue-100 rounded-full flex items-center justify-center mt-0.5">
              <svg className="w-3 h-3 text-blue-600" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
              </svg>
            </div>
            <div>
              <p className="font-medium text-text">Multiple Data Sources</p>
              <p className="text-text-secondary">CrossRef, Semantic Scholar, OpenAlex</p>
            </div>
          </div>
          
          <div className="flex items-start space-x-3">
            <div className="w-5 h-5 bg-purple-100 rounded-full flex items-center justify-center mt-0.5">
              <svg className="w-3 h-3 text-purple-600" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
              </svg>
            </div>
            <div>
              <p className="font-medium text-text">Export & Integration</p>
              <p className="text-text-secondary">BibTeX, Zotero, and various formats</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Layout;