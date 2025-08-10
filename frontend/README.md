# Citation Network Explorer - Frontend

A modern React TypeScript frontend for exploring intellectual lineage through citation networks.

## Features

- **Interactive Visualization**: Explore citation networks in 2D or 3D using React Force Graph
- **Modern Architecture**: Built with React 18, TypeScript, and Vite for fast development
- **State Management**: Zustand for predictable state management
- **Responsive Design**: Mobile-first approach with Tailwind CSS
- **Accessibility**: WCAG 2.2 compliant components
- **Multiple Views**: Network, List, and Table views for different use cases
- **Data Integration**: Support for CrossRef, Semantic Scholar, OpenAlex, and Zotero
- **Export Options**: BibTeX, CSV, JSON, GraphML formats
- **Real-time Updates**: WebSocket integration for live updates

## Tech Stack

- **Framework**: React 18 with TypeScript
- **Build Tool**: Vite
- **Styling**: Tailwind CSS
- **State Management**: Zustand
- **Data Fetching**: TanStack Query (React Query)
- **Visualization**: React Force Graph (2D/3D)
- **Testing**: Vitest + React Testing Library
- **Linting**: ESLint + TypeScript ESLint

## Prerequisites

- Node.js 18+ 
- npm or yarn
- Running FastAPI backend (see backend README)

## Installation

1. **Install dependencies:**
   ```bash
   npm install
   ```

2. **Set up environment variables:**
   ```bash
   cp .env.example .env
   ```
   
   Edit `.env` with your configuration:
   ```env
   VITE_API_URL=http://localhost:8000/api/v1
   VITE_WS_URL=ws://localhost:8000/ws
   ```

3. **Start development server:**
   ```bash
   npm run dev
   ```

   The app will be available at http://localhost:3000

## Development

### Available Scripts

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run preview` - Preview production build
- `npm run lint` - Run ESLint
- `npm run lint:fix` - Fix ESLint issues
- `npm run type-check` - Run TypeScript checks
- `npm run test` - Run tests
- `npm run test:ui` - Run tests with UI
- `npm run test:coverage` - Run tests with coverage

### Project Structure

```
src/
├── components/          # React components
│   ├── ui/             # Reusable UI components
│   ├── layout/         # Layout components
│   ├── visualization/  # Network visualization
│   ├── papers/         # Paper-related components
│   ├── modals/         # Modal dialogs
│   └── forms/          # Form components
├── hooks/              # Custom React hooks
├── services/           # API services
├── store/              # Zustand stores
├── types/              # TypeScript type definitions
├── utils/              # Utility functions
├── styles/             # Global styles
└── test/               # Test utilities
```

### Key Components

**NetworkView**: Interactive force-directed graph visualization
- 2D/3D modes using React Force Graph
- Node coloring by year, citations, or type
- Interactive controls for filtering and layout
- Tooltips and selection handling

**ListView**: Card-based paper listing
- Search and filtering capabilities
- Sorting by various metrics
- Bulk actions for seed management
- Responsive design

**TableView**: Tabular data presentation
- Sortable columns
- Pagination
- Bulk selection
- Compact information display

### State Management

The app uses Zustand for state management with three main stores:

- **paperStore**: Manages papers, edges, and graph configuration
- **uiStore**: Handles UI state, modals, and user preferences  
- **authStore**: Manages authentication and user data

### API Integration

Services are organized by domain:
- **PaperService**: Paper CRUD operations
- **AuthService**: Authentication operations
- **ZoteroService**: Zotero integration
- **SearchService**: External API searches

### Styling

- **Tailwind CSS** for utility-first styling
- **CSS Custom Properties** for theming
- **Responsive Design** with mobile-first approach
- **Dark/Light Mode** support
- **Accessibility** features built-in

## Building for Production

```bash
npm run build
```

The build will be output to the `dist/` directory.

### Build Optimization

- **Code Splitting**: Automatic splitting by route and vendor
- **Tree Shaking**: Removes unused code
- **Asset Optimization**: Images and fonts are optimized
- **Bundle Analysis**: Use `npm run build && npx vite-bundle-analyzer dist`

## Testing

### Unit Testing

```bash
npm run test
```

Tests use Vitest and React Testing Library:
- Component testing
- Hook testing  
- Service testing
- Integration testing

### Test Coverage

```bash
npm run test:coverage
```

## Docker Development

A Dockerfile is provided for containerized development:

```bash
docker build -t citation-explorer-frontend .
docker run -p 3000:3000 citation-explorer-frontend
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `VITE_API_URL` | `http://localhost:8000/api/v1` | Backend API URL |
| `VITE_WS_URL` | `ws://localhost:8000/ws` | WebSocket URL |
| `VITE_ENABLE_ANALYTICS` | `false` | Enable analytics |
| `VITE_ENABLE_WEBSOCKETS` | `true` | Enable WebSocket features |
| `VITE_ENABLE_3D_VISUALIZATION` | `true` | Enable 3D graph mode |

### Theming

The app supports light/dark themes with CSS custom properties:

```css
:root {
  --color-primary: 14 165 233;
  --color-secondary: 100 116 139;
  --color-accent: 239 68 68;
  /* ... */
}
```

## Performance

### Optimization Strategies

- **Lazy Loading**: Components are lazy-loaded by route
- **Memoization**: Expensive calculations are memoized
- **Virtual Scrolling**: Large lists use virtual scrolling
- **Debounced Search**: Search inputs are debounced
- **Optimistic Updates**: UI updates optimistically

### Monitoring

- **React DevTools**: Component inspection
- **TanStack Query DevTools**: Data fetching inspection
- **Zustand DevTools**: State management inspection

## Accessibility

- **ARIA Labels**: Screen reader support
- **Keyboard Navigation**: Full keyboard accessibility
- **Focus Management**: Proper focus handling
- **Color Contrast**: WCAG AA compliant colors
- **Semantic HTML**: Proper HTML semantics

## Browser Support

- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Run linting and tests
6. Submit a pull request

### Code Style

- Use TypeScript for all new code
- Follow the existing component patterns
- Write tests for new features
- Use semantic commit messages

## Troubleshooting

### Common Issues

**Build fails with TypeScript errors:**
```bash
npm run type-check
```

**Tests fail:**
```bash
npm run test -- --reporter=verbose
```

**Development server issues:**
```bash
rm -rf node_modules package-lock.json
npm install
npm run dev
```

## License

This project is licensed under the MIT License.