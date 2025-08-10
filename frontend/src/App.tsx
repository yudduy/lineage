import React from 'react';
import { Toaster } from 'react-hot-toast';
import DashboardPage from '@pages/DashboardPage';
import ErrorBoundary from '@components/ui/ErrorBoundary';

const App: React.FC = () => {
  // Set dark mode by default
  React.useEffect(() => {
    document.documentElement.classList.add('dark');
  }, []);

  return (
    <ErrorBoundary>
      <div className="min-h-screen bg-background text-text">
        <DashboardPage />
        
        {/* Toast notifications */}
        <Toaster
          position="top-right"
          toastOptions={{
            duration: 5000,
            style: {
              background: '#1f2937',
              color: '#f9fafb',
              border: '1px solid #374151',
              borderRadius: '0.5rem',
            },
            success: {
              iconTheme: {
                primary: '#10b981',
                secondary: '#1f2937',
              },
            },
            error: {
              iconTheme: {
                primary: '#ef4444',
                secondary: '#1f2937',
              },
            },
          }}
        />
      </div>
    </ErrorBoundary>
  );
};

export default App;