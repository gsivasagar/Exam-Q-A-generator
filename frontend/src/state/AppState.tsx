import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import { ApiError, type Api, type Credentials, type IngestedDocument, type IngestProgress, type Provider } from '../api'

export interface Upload {
  key: string
  name: string
  status: 'reading' | 'ready' | 'error'
  progress?: IngestProgress
  doc?: IngestedDocument
  error?: string
}

interface AppState {
  api: Api
  creds: Credentials
  setProvider: (p: Provider) => void
  setApiKey: (k: string) => void
  uploads: Upload[]
  /** True until the files already stored for this account have been loaded. */
  loadingFiles: boolean
  readyDocs: IngestedDocument[]
  addFiles: (files: File[]) => void
  addUrl: (url: string) => void
  removeUpload: (key: string) => void
}

const Ctx = createContext<AppState | null>(null)

export function AppStateProvider({ api, children }: { api: Api; children: ReactNode }) {
  const [provider, setProvider] = useState<Provider>('Gemini')
  // The key lives in memory only: closing the tab forgets it.
  const [apiKey, setApiKey] = useState('')
  const [uploads, setUploads] = useState<Upload[]>([])
  const [loadingFiles, setLoadingFiles] = useState(true)
  const credsRef = useRef<Credentials>({ provider, apiKey })
  credsRef.current = { provider, apiKey }
  const counter = useRef(0)

  const patch = useCallback((key: string, change: Partial<Upload>) => {
    setUploads((list) => list.map((u) => (u.key === key ? { ...u, ...change } : u)))
  }, [])

  const run = useCallback(
    (name: string, work: (onProgress: (p: IngestProgress) => void, creds: Credentials) => Promise<IngestedDocument>) => {
      const key = `up-${++counter.current}`
      setUploads((list) => [...list, { key, name, status: 'reading' }])
      work((progress) => patch(key, { progress }), credsRef.current).then(
        (doc) => patch(key, { status: 'ready', doc, progress: undefined }),
        (err: unknown) =>
          patch(key, {
            status: 'error',
            progress: undefined,
            error: err instanceof ApiError ? err.message : 'Something went wrong reading this file.',
          }),
      )
    },
    [patch],
  )

  // Files this account already has: after a reload they are still there.
  useEffect(() => {
    let current = true
    api.listDocuments().then(
      (docs) => {
        if (!current) return
        setUploads((list) => [
          ...docs.filter((d) => !list.some((u) => u.doc?.id === d.id)).map((doc) => ({ key: `doc-${doc.id}`, name: doc.name, status: 'ready' as const, doc })),
          ...list,
        ])
        setLoadingFiles(false)
      },
      () => current && setLoadingFiles(false),
    )
    return () => {
      current = false
    }
  }, [api])

  const value = useMemo<AppState>(
    () => ({
      api,
      creds: { provider, apiKey },
      setProvider,
      setApiKey,
      uploads,
      loadingFiles,
      readyDocs: uploads.flatMap((u) => (u.status === 'ready' && u.doc ? [u.doc] : [])),
      addFiles: (files) => files.forEach((f) => run(f.name, (onProgress, creds) => api.ingestFile(f, creds, onProgress))),
      addUrl: (url) => run(url, (_p, creds) => api.ingestUrl(url, creds)),
      removeUpload: (key) => {
        const doc = uploads.find((u) => u.key === key)?.doc
        setUploads((list) => list.filter((u) => u.key !== key))
        if (doc) void api.deleteDocument(doc.id).catch(() => undefined) // the stored file goes too
      },
    }),
    [api, provider, apiKey, uploads, loadingFiles, run],
  )

  return <Ctx.Provider value={value}>{children}</Ctx.Provider>
}

export function useApp(): AppState {
  const ctx = useContext(Ctx)
  if (!ctx) throw new Error('useApp must be used inside AppStateProvider')
  return ctx
}
