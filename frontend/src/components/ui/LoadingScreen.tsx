import React from 'react';

const LoadingScreen: React.FC = () => {
  return (
    <div className="min-h-screen bg-background flex items-center justify-center">
      <div className="flex flex-col items-center space-y-6">
        {/* Logo or Icon */}
        <div className="w-16 h-16 bg-primary-600 rounded-lg flex items-center justify-center">
          <svg
            className="w-8 h-8 text-white"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M13 10V3L4 14h7v7l9-11h-7z"
            />
          </svg>
        </div>
        
        {/* App Name */}
        <div className="text-center">
          <h1 className="text-2xl font-bold text-text mb-2">
            Citation Network Explorer
          </h1>
          <p className="text-text-secondary">
            Exploring intellectual lineage through citation networks
          </p>
        </div>
        
        {/* Loading Spinner */}
        <div className="relative">
          <div className="w-8 h-8 border-4 border-surface rounded-full"></div>
          <div className="absolute top-0 left-0 w-8 h-8 border-4 border-primary-600 border-t-transparent rounded-full animate-spin"></div>
        </div>
      </div>
    </div>
  );
};

export default LoadingScreen;