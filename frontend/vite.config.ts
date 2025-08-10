import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
      '@components': path.resolve(__dirname, './src/components'),
      '@hooks': path.resolve(__dirname, './src/hooks'),
      '@services': path.resolve(__dirname, './src/services'),
      '@types': path.resolve(__dirname, './src/types'),
      '@utils': path.resolve(__dirname, './src/utils'),
      '@store': path.resolve(__dirname, './src/store'),
      '@pages': path.resolve(__dirname, './src/pages'),
      '@styles': path.resolve(__dirname, './src/styles')
    }
  },
  server: {
    port: 3000,
    host: true,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        secure: false
      }
    }
  },
  build: {
    outDir: 'dist',
    sourcemap: true,
    rollupOptions: {
      external: ['three/examples/jsm/renderers/webgpu/WebGPURenderer.js'],
      output: {
        manualChunks: {
          vendor: ['react', 'react-dom'],
          visualization: ['react-force-graph', 'react-force-graph-2d', 'react-force-graph-3d', 'd3'],
          routing: ['react-router-dom'],
          query: ['@tanstack/react-query'],
          ui: ['framer-motion', 'react-hot-toast']
        }
      }
    }
  },
  optimizeDeps: {
    include: ['react-force-graph', 'react-force-graph-2d', 'react-force-graph-3d', 'd3']
  },
  test: {
    globals: true,
    environment: 'jsdom',
    setupFiles: ['./src/test/setup.ts'],
    css: true
  }
})