/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_API_URL?: string
  readonly VITE_PAPER_PDF?: string
  readonly VITE_PAPER_TITLE?: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}
