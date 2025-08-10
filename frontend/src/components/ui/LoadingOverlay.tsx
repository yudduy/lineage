import React from 'react';
import { clsx } from 'clsx';

interface LoadingOverlayProps {
  message?: string;
  className?: string;
}

const LoadingOverlay: React.FC<LoadingOverlayProps> = ({
  message = 'Loading...',
  className,
}) => {
  return (
    <div
      className={clsx(
        'absolute inset-0 bg-background/80 backdrop-blur-sm',
        'flex items-center justify-center z-50',
        className
      )}
    >
      <div className="flex flex-col items-center space-y-4">
        {/* Spinner */}
        <div className="relative">
          <div className="w-12 h-12 border-4 border-surface rounded-full"></div>
          <div className="absolute top-0 left-0 w-12 h-12 border-4 border-primary-600 border-t-transparent rounded-full animate-spin"></div>
        </div>
        
        {/* Message */}
        <div className="text-sm text-text-secondary font-medium">
          {message}
        </div>
      </div>
    </div>
  );
};

export default LoadingOverlay;