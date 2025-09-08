/**
 * Comprehensive Frontend Testing Suite for Intellectual Lineage Tracer
 * 
 * This test suite covers:
 * - React component interaction testing
 * - State management with Zustand
 * - Graph visualization rendering
 * - Real-time updates via WebSocket
 * - User workflow testing (search, analyze, export)
 * - Cross-browser compatibility scenarios
 * - Performance and accessibility testing
 */

import React from 'react';
import { render, screen, fireEvent, waitFor, act } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { vi, describe, it, expect, beforeEach, afterEach } from 'vitest';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { BrowserRouter } from 'react-router-dom';

// Import components
import App from '../App';
import DashboardPage from '../pages/DashboardPage';
import NetworkView from '../components/visualization/NetworkView';
import CitationFlowVisualizer from '../components/visualization/CitationFlowVisualizer';
import PaperDetailsPanel from '../components/panels/PaperDetailsPanel';
import SearchModal from '../components/modals/SearchModal';
import BibTeXUploadModal from '../components/modals/BibTeXUploadModal';
import ExportModal from '../components/modals/ExportModal';
import RealtimeStatusIndicator from '../components/ui/RealtimeStatusIndicator';

// Import services and stores
import { paperStore } from '../store/paperStore';
import { authStore } from '../store/authStore';
import { uiStore } from '../store/uiStore';
import { getWebSocketService } from '../services/websocketService';
import * as api from '../services/api';

// Mock external dependencies
vi.mock('../services/api');
vi.mock('../services/websocketService');
vi.mock('react-force-graph', () => ({
  default: vi.fn().mockImplementation(() => <div data-testid="force-graph" />)
}));
vi.mock('d3', () => ({
  select: vi.fn().mockReturnThis(),
  selectAll: vi.fn().mockReturnThis(),
  append: vi.fn().mockReturnThis(),
  attr: vi.fn().mockReturnThis(),
  style: vi.fn().mockReturnThis(),
  data: vi.fn().mockReturnThis(),
  enter: vi.fn().mockReturnThis(),
  exit: vi.fn().mockReturnThis(),
  remove: vi.fn().mockReturnThis(),
}));

// Test utilities
const createQueryClient = () => new QueryClient({
  defaultOptions: {
    queries: { retry: false },
    mutations: { retry: false },
  },
});

const TestWrapper: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const queryClient = createQueryClient();
  
  return (
    <BrowserRouter>
      <QueryClientProvider client={queryClient}>
        {children}
      </QueryClientProvider>
    </BrowserRouter>
  );
};

// Sample data for testing
const mockPapers = [
  {
    id: 'paper-1',
    title: 'Machine Learning in Healthcare: A Comprehensive Survey',
    authors: [
      { id: 'author-1', name: 'Dr. Jane Smith', affiliation: 'MIT' }
    ],
    publicationYear: 2023,
    doi: '10.1000/test1',
    abstract: 'This paper provides a comprehensive survey of machine learning applications in healthcare...',
    journal: { name: 'Nature Medicine', issn: '1234-5678' },
    citationCount: { total: 150, recent: 25 },
    isOpenAccess: true
  },
  {
    id: 'paper-2',
    title: 'Deep Learning for Medical Image Analysis',
    authors: [
      { id: 'author-2', name: 'Prof. John Doe', affiliation: 'Stanford' }
    ],
    publicationYear: 2022,
    doi: '10.1000/test2',
    abstract: 'This work explores deep learning approaches for medical image analysis...',
    journal: { name: 'Medical Image Analysis', issn: '8765-4321' },
    citationCount: { total: 89, recent: 12 },
    isOpenAccess: false
  }
];

const mockNetwork = {
  networkId: 'network-1',
  nodes: [
    { id: 'paper-1', type: 'paper', title: 'Paper 1', x: 0, y: 0 },
    { id: 'paper-2', type: 'paper', title: 'Paper 2', x: 100, y: 100 }
  ],
  edges: [
    { source: 'paper-1', target: 'paper-2', type: 'cites', weight: 1 }
  ],
  statistics: {
    totalNodes: 2,
    totalEdges: 1,
    communities: 1
  }
};

describe('Core Component Functionality', () => {
  let queryClient: QueryClient;

  beforeEach(() => {
    queryClient = createQueryClient();
    
    // Reset stores
    paperStore.getState().reset?.();
    authStore.getState().reset?.();
    uiStore.getState().reset?.();
    
    // Mock API responses
    vi.mocked(api.searchPapers).mockResolvedValue({
      papers: mockPapers,
      total: 2,
      page: 1,
      pageSize: 20,
      totalPages: 1
    });
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  describe('App Component', () => {
    it('renders main application layout', async () => {
      render(
        <TestWrapper>
          <App />
        </TestWrapper>
      );

      // Check for main layout elements
      expect(screen.getByTestId('app-layout')).toBeInTheDocument();
      expect(screen.getByTestId('sidebar')).toBeInTheDocument();
      expect(screen.getByTestId('main-content')).toBeInTheDocument();
    });

    it('handles authentication state changes', async () => {
      render(
        <TestWrapper>
          <App />
        </TestWrapper>
      );

      // Initially should show login/register options
      expect(screen.getByTestId('auth-section')).toBeInTheDocument();

      // Simulate user authentication
      act(() => {
        authStore.getState().setUser({
          id: 'user-1',
          email: 'test@example.com',
          fullName: 'Test User'
        });
        authStore.getState().setToken('mock-jwt-token');
      });

      await waitFor(() => {
        expect(screen.getByTestId('user-profile')).toBeInTheDocument();
      });
    });
  });

  describe('DashboardPage Component', () => {
    it('renders dashboard with search functionality', async () => {
      render(
        <TestWrapper>
          <DashboardPage />
        </TestWrapper>
      );

      expect(screen.getByTestId('dashboard-page')).toBeInTheDocument();
      expect(screen.getByTestId('search-input')).toBeInTheDocument();
      expect(screen.getByTestId('search-filters')).toBeInTheDocument();
    });

    it('performs paper search and displays results', async () => {
      const user = userEvent.setup();
      
      render(
        <TestWrapper>
          <DashboardPage />
        </TestWrapper>
      );

      // Enter search query
      const searchInput = screen.getByTestId('search-input');
      await user.type(searchInput, 'machine learning');

      // Click search button
      const searchButton = screen.getByTestId('search-button');
      await user.click(searchButton);

      // Wait for results to appear
      await waitFor(() => {
        expect(screen.getByTestId('search-results')).toBeInTheDocument();
        expect(screen.getByText('Machine Learning in Healthcare')).toBeInTheDocument();
      });
    });

    it('handles search errors gracefully', async () => {
      // Mock API error
      vi.mocked(api.searchPapers).mockRejectedValue(new Error('Search failed'));
      
      const user = userEvent.setup();
      
      render(
        <TestWrapper>
          <DashboardPage />
        </TestWrapper>
      );

      // Perform search
      await user.type(screen.getByTestId('search-input'), 'test query');
      await user.click(screen.getByTestId('search-button'));

      // Should show error message
      await waitFor(() => {
        expect(screen.getByTestId('error-message')).toBeInTheDocument();
        expect(screen.getByText(/search failed/i)).toBeInTheDocument();
      });
    });
  });

  describe('NetworkView Component', () => {
    it('renders network visualization', async () => {
      render(
        <TestWrapper>
          <NetworkView network={mockNetwork} />
        </TestWrapper>
      );

      expect(screen.getByTestId('network-view')).toBeInTheDocument();
      expect(screen.getByTestId('force-graph')).toBeInTheDocument();
      expect(screen.getByTestId('network-controls')).toBeInTheDocument();
    });

    it('handles node selection and displays details', async () => {
      const user = userEvent.setup();
      
      render(
        <TestWrapper>
          <NetworkView network={mockNetwork} />
        </TestWrapper>
      );

      // Simulate node click
      const networkView = screen.getByTestId('network-view');
      fireEvent.click(networkView, {
        detail: { node: { id: 'paper-1', title: 'Paper 1' } }
      });

      await waitFor(() => {
        expect(screen.getByTestId('node-details-panel')).toBeInTheDocument();
      });
    });

    it('updates visualization when network data changes', async () => {
      const { rerender } = render(
        <TestWrapper>
          <NetworkView network={mockNetwork} />
        </TestWrapper>
      );

      const updatedNetwork = {
        ...mockNetwork,
        nodes: [...mockNetwork.nodes, { id: 'paper-3', type: 'paper', title: 'Paper 3', x: 200, y: 200 }]
      };

      rerender(
        <TestWrapper>
          <NetworkView network={updatedNetwork} />
        </TestWrapper>
      );

      // Verify component re-renders with new data
      expect(screen.getByTestId('network-view')).toBeInTheDocument();
    });
  });

  describe('SearchModal Component', () => {
    it('renders advanced search form', async () => {
      render(
        <TestWrapper>
          <SearchModal isOpen={true} onClose={() => {}} />
        </TestWrapper>
      );

      expect(screen.getByTestId('advanced-search-modal')).toBeInTheDocument();
      expect(screen.getByTestId('year-range-filter')).toBeInTheDocument();
      expect(screen.getByTestId('author-filter')).toBeInTheDocument();
      expect(screen.getByTestId('journal-filter')).toBeInTheDocument();
      expect(screen.getByTestId('open-access-filter')).toBeInTheDocument();
    });

    it('submits advanced search with filters', async () => {
      const user = userEvent.setup();
      const onSearchMock = vi.fn();
      
      render(
        <TestWrapper>
          <SearchModal isOpen={true} onClose={() => {}} onSearch={onSearchMock} />
        </TestWrapper>
      );

      // Fill in search filters
      await user.type(screen.getByTestId('author-input'), 'Smith');
      await user.selectOptions(screen.getByTestId('year-from-select'), '2020');
      await user.selectOptions(screen.getByTestId('year-to-select'), '2023');
      await user.click(screen.getByTestId('open-access-checkbox'));

      // Submit search
      await user.click(screen.getByTestId('search-submit-button'));

      expect(onSearchMock).toHaveBeenCalledWith({
        author: 'Smith',
        yearFrom: 2020,
        yearTo: 2023,
        isOpenAccess: true
      });
    });
  });
});

describe('State Management Tests', () => {
  beforeEach(() => {
    // Reset all stores before each test
    paperStore.getState().reset?.();
    authStore.getState().reset?.();
    uiStore.getState().reset?.();
  });

  describe('Paper Store', () => {
    it('manages search results correctly', () => {
      const { setPapers, setSearchQuery, setFilters } = paperStore.getState();

      // Set search query
      setSearchQuery('machine learning');
      expect(paperStore.getState().searchQuery).toBe('machine learning');

      // Set papers
      setPapers(mockPapers);
      expect(paperStore.getState().papers).toEqual(mockPapers);

      // Set filters
      const filters = { yearFrom: 2020, yearTo: 2023, isOpenAccess: true };
      setFilters(filters);
      expect(paperStore.getState().filters).toEqual(filters);
    });

    it('manages selected papers', () => {
      const { selectPaper, unselectPaper } = paperStore.getState();

      // Select paper
      selectPaper('paper-1');
      expect(paperStore.getState().selectedPapers).toContain('paper-1');

      // Unselect paper
      unselectPaper('paper-1');
      expect(paperStore.getState().selectedPapers).not.toContain('paper-1');
    });

    it('manages network data', () => {
      const { setNetwork, updateNetworkNode } = paperStore.getState();

      // Set network
      setNetwork(mockNetwork);
      expect(paperStore.getState().network).toEqual(mockNetwork);

      // Update node
      const updatedNode = { id: 'paper-1', type: 'paper', title: 'Updated Paper 1', x: 50, y: 50 };
      updateNetworkNode('paper-1', updatedNode);
      
      const network = paperStore.getState().network;
      const node = network?.nodes.find(n => n.id === 'paper-1');
      expect(node?.title).toBe('Updated Paper 1');
    });
  });

  describe('Auth Store', () => {
    it('manages user authentication', () => {
      const { setUser, setToken, logout } = authStore.getState();

      // Set user and token
      const user = { id: 'user-1', email: 'test@example.com', fullName: 'Test User' };
      setUser(user);
      setToken('mock-token');

      expect(authStore.getState().user).toEqual(user);
      expect(authStore.getState().token).toBe('mock-token');
      expect(authStore.getState().isAuthenticated).toBe(true);

      // Logout
      logout();
      expect(authStore.getState().user).toBeNull();
      expect(authStore.getState().token).toBeNull();
      expect(authStore.getState().isAuthenticated).toBe(false);
    });
  });

  describe('UI Store', () => {
    it('manages modal states', () => {
      const { openModal, closeModal } = uiStore.getState();

      // Open modal
      openModal('advancedSearch');
      expect(uiStore.getState().modals.advancedSearch).toBe(true);

      // Close modal
      closeModal('advancedSearch');
      expect(uiStore.getState().modals.advancedSearch).toBe(false);
    });

    it('manages loading states', () => {
      const { setLoading } = uiStore.getState();

      // Set loading
      setLoading('papers', true);
      expect(uiStore.getState().loading.papers).toBe(true);

      // Clear loading
      setLoading('papers', false);
      expect(uiStore.getState().loading.papers).toBe(false);
    });
  });
});

describe('Real-time Features Tests', () => {
  let mockWebSocketService: any;

  beforeEach(() => {
    mockWebSocketService = {
      connect: vi.fn().mockResolvedValue(undefined),
      disconnect: vi.fn(),
      isConnected: vi.fn().mockReturnValue(true),
      setHandlers: vi.fn(),
      emit: vi.fn(),
      subscribeToTask: vi.fn(),
      subscribeToCollaboration: vi.fn(),
      getConnectionState: vi.fn().mockReturnValue('connected')
    };
    
    vi.mocked(getWebSocketService).mockReturnValue(mockWebSocketService);
  });

  describe('RealtimeStatusIndicator Component', () => {
    it('displays connection status', () => {
      render(
        <TestWrapper>
          <RealtimeStatusIndicator />
        </TestWrapper>
      );

      expect(screen.getByTestId('realtime-status')).toBeInTheDocument();
      expect(screen.getByTestId('connection-status')).toHaveTextContent('connected');
    });

    it('handles connection state changes', async () => {
      mockWebSocketService.getConnectionState.mockReturnValue('connecting');
      
      const { rerender } = render(
        <TestWrapper>
          <RealtimeStatusIndicator />
        </TestWrapper>
      );

      expect(screen.getByTestId('connection-status')).toHaveTextContent('connecting');

      // Simulate connection established
      mockWebSocketService.getConnectionState.mockReturnValue('connected');
      rerender(
        <TestWrapper>
          <RealtimeStatusIndicator />
        </TestWrapper>
      );

      expect(screen.getByTestId('connection-status')).toHaveTextContent('connected');
    });
  });

  describe('Real-time Updates', () => {
    it('handles real-time paper updates', async () => {
      render(
        <TestWrapper>
          <DashboardPage />
        </TestWrapper>
      );

      // Simulate WebSocket message
      const updateMessage = {
        type: 'paper_update',
        data: {
          paperId: 'paper-1',
          field: 'citationCount',
          value: 200
        }
      };

      // Trigger WebSocket message handler
      const setHandlersCall = mockWebSocketService.setHandlers.mock.calls[0];
      if (setHandlersCall && setHandlersCall[0].onMessage) {
        act(() => {
          setHandlersCall[0].onMessage(updateMessage);
        });
      }

      // Verify update was processed
      await waitFor(() => {
        const updatedPaper = paperStore.getState().papers.find(p => p.id === 'paper-1');
        expect(updatedPaper?.citationCount.total).toBe(200);
      });
    });

    it('handles collaboration updates', async () => {
      render(
        <TestWrapper>
          <NetworkView network={mockNetwork} />
        </TestWrapper>
      );

      const collaborationUpdate = {
        type: 'collaboration',
        data: {
          action: 'node_moved',
          nodeId: 'paper-1',
          position: { x: 150, y: 150 },
          userId: 'other-user'
        }
      };

      // Trigger collaboration message
      const setHandlersCall = mockWebSocketService.setHandlers.mock.calls[0];
      if (setHandlersCall && setHandlersCall[0].onCollaboration) {
        act(() => {
          setHandlersCall[0].onCollaboration(collaborationUpdate.data);
        });
      }

      // Verify collaboration update was processed
      await waitFor(() => {
        const network = paperStore.getState().network;
        const node = network?.nodes.find(n => n.id === 'paper-1');
        expect(node?.x).toBe(150);
        expect(node?.y).toBe(150);
      });
    });
  });
});

describe('User Workflow Tests', () => {
  beforeEach(() => {
    vi.mocked(api.searchPapers).mockResolvedValue({
      papers: mockPapers,
      total: 2,
      page: 1,
      pageSize: 20,
      totalPages: 1
    });

    vi.mocked(api.buildCitationNetwork).mockResolvedValue({
      networkId: 'network-1',
      taskId: 'task-1',
      status: 'pending'
    });

    vi.mocked(api.exportNetwork).mockResolvedValue({
      exportId: 'export-1',
      downloadUrl: '/api/v1/exports/export-1.json',
      format: 'json'
    });
  });

  describe('Complete Search to Analysis Workflow', () => {
    it('completes full workflow: search → select → analyze → export', async () => {
      const user = userEvent.setup();
      
      render(
        <TestWrapper>
          <App />
        </TestWrapper>
      );

      // Step 1: Authenticate user
      act(() => {
        authStore.getState().setUser({
          id: 'user-1',
          email: 'test@example.com',
          fullName: 'Test User'
        });
        authStore.getState().setToken('mock-token');
      });

      // Step 2: Search for papers
      const searchInput = screen.getByTestId('search-input');
      await user.type(searchInput, 'machine learning healthcare');
      await user.click(screen.getByTestId('search-button'));

      // Wait for search results
      await waitFor(() => {
        expect(screen.getByTestId('search-results')).toBeInTheDocument();
      });

      // Step 3: Select papers for analysis
      const paperCheckboxes = screen.getAllByTestId(/paper-checkbox-/);
      await user.click(paperCheckboxes[0]);
      await user.click(paperCheckboxes[1]);

      // Step 4: Build citation network
      await user.click(screen.getByTestId('build-network-button'));

      // Wait for network to be built
      await waitFor(() => {
        expect(screen.getByTestId('network-view')).toBeInTheDocument();
      });

      // Step 5: Export results
      await user.click(screen.getByTestId('export-button'));
      
      // Wait for export modal
      await waitFor(() => {
        expect(screen.getByTestId('export-modal')).toBeInTheDocument();
      });

      await user.click(screen.getByTestId('export-json-button'));
      
      // Verify export was initiated
      expect(api.exportNetwork).toHaveBeenCalled();
    });
  });

  describe('BibTeX Import Workflow', () => {
    it('imports BibTeX file and processes papers', async () => {
      const user = userEvent.setup();
      
      // Mock file upload
      vi.mocked(api.uploadBibTeX).mockResolvedValue({
        taskId: 'bibtex-task-1',
        status: 'processing'
      });

      render(
        <TestWrapper>
          <BibTeXUploadModal isOpen={true} onClose={() => {}} />
        </TestWrapper>
      );

      // Create mock file
      const bibtexContent = `
        @article{test2023,
          title={Test Paper},
          author={Test Author},
          year={2023}
        }
      `;
      const file = new File([bibtexContent], 'test.bib', { type: 'text/plain' });

      // Upload file
      const fileInput = screen.getByTestId('bibtex-file-input');
      await user.upload(fileInput, file);

      await user.click(screen.getByTestId('upload-submit-button'));

      // Verify upload was initiated
      expect(api.uploadBibTeX).toHaveBeenCalledWith(expect.any(FormData));
    });
  });
});

describe('Performance and Accessibility Tests', () => {
  describe('Performance Tests', () => {
    it('renders large paper lists efficiently', async () => {
      const largePaperList = Array.from({ length: 1000 }, (_, i) => ({
        ...mockPapers[0],
        id: `paper-${i}`,
        title: `Paper ${i}`
      }));

      vi.mocked(api.searchPapers).mockResolvedValue({
        papers: largePaperList,
        total: 1000,
        page: 1,
        pageSize: 50,
        totalPages: 20
      });

      const startTime = performance.now();
      
      render(
        <TestWrapper>
          <DashboardPage />
        </TestWrapper>
      );

      // Trigger search
      const user = userEvent.setup();
      await user.type(screen.getByTestId('search-input'), 'test');
      await user.click(screen.getByTestId('search-button'));

      await waitFor(() => {
        expect(screen.getByTestId('search-results')).toBeInTheDocument();
      });

      const renderTime = performance.now() - startTime;
      expect(renderTime).toBeLessThan(1000); // Should render within 1 second
    });

    it('handles large network visualization efficiently', () => {
      const largeNetwork = {
        ...mockNetwork,
        nodes: Array.from({ length: 500 }, (_, i) => ({
          id: `node-${i}`,
          type: 'paper',
          title: `Paper ${i}`,
          x: Math.random() * 1000,
          y: Math.random() * 1000
        })),
        edges: Array.from({ length: 1000 }, (_, i) => ({
          source: `node-${i % 500}`,
          target: `node-${(i + 1) % 500}`,
          type: 'cites',
          weight: 1
        }))
      };

      const startTime = performance.now();
      
      render(
        <TestWrapper>
          <NetworkView network={largeNetwork} />
        </TestWrapper>
      );

      const renderTime = performance.now() - startTime;
      expect(renderTime).toBeLessThan(2000); // Should render within 2 seconds
    });
  });

  describe('Accessibility Tests', () => {
    it('provides proper ARIA labels for interactive elements', () => {
      render(
        <TestWrapper>
          <DashboardPage />
        </TestWrapper>
      );

      // Check for ARIA labels
      expect(screen.getByLabelText(/search papers/i)).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /search/i })).toBeInTheDocument();
    });

    it('supports keyboard navigation', async () => {
      const user = userEvent.setup();
      
      render(
        <TestWrapper>
          <DashboardPage />
        </TestWrapper>
      );

      // Tab through interactive elements
      await user.tab();
      expect(screen.getByTestId('search-input')).toHaveFocus();

      await user.tab();
      expect(screen.getByTestId('search-button')).toHaveFocus();
    });

    it('provides proper heading hierarchy', () => {
      render(
        <TestWrapper>
          <DashboardPage />
        </TestWrapper>
      );

      // Check heading hierarchy
      const headings = screen.getAllByRole('heading');
      expect(headings[0]).toHaveTextContent(/intellectual lineage tracer/i);
      expect(headings[0].tagName).toBe('H1');
    });
  });
});

describe('Error Handling and Edge Cases', () => {
  it('handles API failures gracefully', async () => {
    vi.mocked(api.searchPapers).mockRejectedValue(new Error('API Error'));
    
    const user = userEvent.setup();
    
    render(
      <TestWrapper>
        <DashboardPage />
      </TestWrapper>
    );

    await user.type(screen.getByTestId('search-input'), 'test');
    await user.click(screen.getByTestId('search-button'));

    await waitFor(() => {
      expect(screen.getByTestId('error-message')).toBeInTheDocument();
    });
  });

  it('handles network disconnection', async () => {
    const mockWebSocketService = {
      getConnectionState: vi.fn().mockReturnValue('disconnected'),
      connect: vi.fn(),
      disconnect: vi.fn(),
      isConnected: vi.fn().mockReturnValue(false),
      setHandlers: vi.fn(),
    };
    
    vi.mocked(getWebSocketService).mockReturnValue(mockWebSocketService);

    render(
      <TestWrapper>
        <RealtimeStatusIndicator />
      </TestWrapper>
    );

    expect(screen.getByTestId('connection-status')).toHaveTextContent('disconnected');
    expect(screen.getByTestId('reconnect-button')).toBeInTheDocument();
  });

  it('validates form inputs properly', async () => {
    const user = userEvent.setup();
    
    render(
      <TestWrapper>
        <SearchModal isOpen={true} onClose={() => {}} />
      </TestWrapper>
    );

    // Try to submit without required fields
    await user.click(screen.getByTestId('search-submit-button'));

    await waitFor(() => {
      expect(screen.getByTestId('validation-error')).toBeInTheDocument();
    });
  });
});