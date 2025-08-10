import React from 'react';
import { useLocation, Link } from 'react-router-dom';
import { useUiStore, useAuthStore } from '@store/index';
import Button from '@components/ui/Button';
import RealtimeStatusIndicator from '@components/ui/RealtimeStatusIndicator';
import { clsx } from 'clsx';

const Header: React.FC = () => {
  const location = useLocation();
  const { 
    currentView, 
    setCurrentView, 
    toggleSidebar, 
    isSidebarOpen,
    openModal 
  } = useUiStore();
  const { isAuthenticated, user } = useAuthStore();

  const navigationItems = [
    { id: 'dashboard', label: 'Dashboard', path: '/dashboard', icon: '📊' },
    { id: 'network', label: 'Network', path: '/network', icon: '🕸️' },
    { id: 'list', label: 'List', path: '/list', icon: '📋' },
    { id: 'table', label: 'Table', path: '/table', icon: '📋' },
  ];

  return (
    <header className="bg-background border-b border-border px-4 py-3">
      <div className="flex items-center justify-between">
        {/* Left Section */}
        <div className="flex items-center space-x-4">
          {/* Sidebar Toggle */}
          <Button
            variant="ghost"
            size="sm"
            onClick={toggleSidebar}
            className="p-2"
          >
            <svg
              className={clsx('w-5 h-5 transition-transform', {
                'rotate-180': !isSidebarOpen,
              })}
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M4 6h16M4 12h16M4 18h16"
              />
            </svg>
          </Button>

          {/* Logo and Title */}
          <div className="flex items-center space-x-3">
            <div className="w-8 h-8 bg-primary-600 rounded-lg flex items-center justify-center">
              <svg
                className="w-4 h-4 text-white"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M13.828 10.172a4 4 0 00-5.656 0l-4 4a4 4 0 105.656 5.656l1.102-1.101m-.758-4.899a4 4 0 005.656 0l4-4a4 4 0 00-5.656-5.656l-1.1 1.1"
                />
              </svg>
            </div>
            <h1 className="font-semibold text-text text-lg hidden md:block">
              Citation Network Explorer
            </h1>
          </div>
        </div>

        {/* Center Section - Navigation */}
        <div className="flex items-center bg-surface rounded-lg p-1">
          {navigationItems.map((item) => {
            const isActive = location.pathname === item.path || 
              (location.pathname === '/' && item.id === 'dashboard');
            
            return (
              <Link
                key={item.id}
                to={item.path}
                onClick={() => setCurrentView(item.id as any)}
                className={clsx(
                  'flex items-center space-x-2 px-3 py-2 rounded-md text-sm font-medium transition-colors',
                  {
                    'bg-background text-text shadow-sm': isActive,
                    'text-text-secondary hover:text-text hover:bg-background/50': !isActive,
                  }
                )}
              >
                <span className="text-base">{item.icon}</span>
                <span className="hidden sm:inline">{item.label}</span>
              </Link>
            );
          })}
        </div>

        {/* Right Section */}
        <div className="flex items-center space-x-3">
          {/* Real-time Status */}
          <RealtimeStatusIndicator />
          
          {/* Add Paper Button */}
          <Button
            variant="primary"
            size="sm"
            onClick={() => openModal('addPaper')}
            leftIcon={
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
              </svg>
            }
          >
            <span className="hidden sm:inline">Add Papers</span>
          </Button>

          {/* Search */}
          <Button
            variant="ghost"
            size="sm"
            onClick={() => openModal('titleSearch')}
            className="p-2"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
              />
            </svg>
          </Button>

          {/* Settings */}
          <Button
            variant="ghost"
            size="sm"
            onClick={() => openModal('settings')}
            className="p-2"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z"
              />
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"
              />
            </svg>
          </Button>

          {/* User Menu */}
          {isAuthenticated && user ? (
            <div className="flex items-center space-x-2">
              <div className="w-8 h-8 bg-primary-600 rounded-full flex items-center justify-center">
                <span className="text-white text-sm font-medium">
                  {user.name?.charAt(0) || user.email?.charAt(0) || 'U'}
                </span>
              </div>
              <span className="text-sm text-text hidden lg:inline">
                {user.name || user.email}
              </span>
            </div>
          ) : (
            <Button
              variant="outline"
              size="sm"
              onClick={() => {/* Handle login */}}
            >
              Sign In
            </Button>
          )}

          {/* Help */}
          <Button
            variant="ghost"
            size="sm"
            onClick={() => openModal('help')}
            className="p-2"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M8.228 9c.549-1.165 2.03-2 3.772-2 2.21 0 4 1.343 4 3 0 1.4-1.278 2.575-3.006 2.907-.542.104-.994.54-.994 1.093m0 3h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
              />
            </svg>
          </Button>
        </div>
      </div>
    </header>
  );
};

export default Header;