import { createServer } from 'vite'

/** SSR-only fixtures must not load local configuration, keys, or background servers. */
export function createTestServer(options = {}) {
  return createServer({
    ...options,
    configFile: false,
    envDir: false,
    optimizeDeps: { noDiscovery: true },
    server: { middlewareMode: true, watch: null, hmr: false, ws: false },
    appType: 'custom',
  })
}
