import { defineConfig, type Plugin } from 'vite'
import { readFileSync } from 'node:fs'
import { join } from 'node:path'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import { deploymentHeaders } from './src/utils/deploymentHeaders.ts'

const BACKEND = 'http://localhost:8000'
let productionHeaders = ''
const exactBackendHeaders: Plugin = {
  name: 'lumin-exact-backend-headers',
  apply: 'build',
  configResolved(config) {
    productionHeaders = deploymentHeaders(readFileSync(join(config.publicDir, '_headers'), 'utf8'), config.env.VITE_API_URL ?? '')
  },
  generateBundle() {
    this.emitFile({ type: 'asset', fileName: '_headers', source: productionHeaders })
  },
}

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss(), exactBackendHeaders],
  optimizeDeps: { include: ['react-pdf', 'docx-preview'] },
  server: {
    // Same-origin proxying keeps the HttpOnly SameSite=Strict session cookie first-party.
    proxy: {
      '/api': {
        target: BACKEND,
        changeOrigin: true,
      },
      '/static/illustrations': {
        target: BACKEND,
        changeOrigin: true,
      },
    },
  },
})
