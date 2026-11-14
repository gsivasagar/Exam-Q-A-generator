import { SessionStore } from './session'
import {
  ApiError,
  type Api,
  type ApiErrorCode,
  type AuthResult,
  type Credentials,
  type IngestedDocument,
  type IngestProgress,
} from './types'

const CODES: ApiErrorCode[] = [
  'ocr_limit',
  'needs_gemini',
  'invalid_key',
  'model_error',
  'unreadable',
  'unsupported',
  'no_documents',
  'not_found',
  'rate_limited',
  'not_guest',
  'no_password',
  'confirmation_mismatch',
  'unauthorized',
  'invalid_credentials',
  'email_taken',
  'invalid_email',
  'weak_password',
  'wrong_password',
  'guest_limit',
  'csrf',
]

function asCode(value: unknown): ApiErrorCode {
  return CODES.includes(value as ApiErrorCode) ? (value as ApiErrorCode) : 'unknown'
}

/** Headers for calls that may reach a model. The key travels in a header, never in a URL or body. */
function credentialHeaders(creds?: Credentials): Record<string, string> {
  if (!creds) return {}
  const headers: Record<string, string> = { 'X-Provider': creds.provider }
  if (creds.apiKey) headers['X-Api-Key'] = creds.apiKey
  return headers
}

function codeForStatus(status: number): ApiErrorCode {
  return status === 401 ? 'unauthorized' : status === 404 ? 'not_found' : status === 429 ? 'rate_limited' : 'unknown'
}

async function errorFrom(res: Response): Promise<ApiError> {
  try {
    const body = (await res.json()) as { code?: string; message?: string; detail?: string }
    return new ApiError(
      body.code ? asCode(body.code) : codeForStatus(res.status),
      body.message ?? body.detail ?? `Request failed (${res.status}).`,
    )
  } catch {
    return new ApiError(codeForStatus(res.status), `Request failed (${res.status}).`)
  }
}

type Fetch = typeof fetch

interface SendOptions {
  creds?: Credentials
  /** Send the access token and, on 401, refresh once and retry. False for sign-in, sign-up, guest, refresh, logout. */
  authed?: boolean
}

export function createHttpApi(
  baseUrl = '/api',
  fetchImpl: Fetch = (...args) => fetch(...args),
  store: SessionStore = new SessionStore(),
): Api {
  async function rawSend(path: string, init: RequestInit, opts: SendOptions, token: string | null): Promise<Response> {
    const headers: Record<string, string> = { ...(init.headers as Record<string, string>), ...credentialHeaders(opts.creds) }
    if (opts.authed && token) headers.Authorization = `Bearer ${token}`
    try {
      return await fetchImpl(baseUrl + path, { ...init, headers, credentials: 'include' })
    } catch {
      throw new ApiError('network', 'Could not reach the server. Is the back end running?')
    }
  }

  // One refresh at a time: calls that fail together wait for the same answer.
  let refreshing: Promise<string | null> | null = null
  function refreshOnce(): Promise<string | null> {
    refreshing ??= (async () => {
      try {
        const result = await api.refresh()
        store.token = result.accessToken
        return result.accessToken
      } catch {
        store.expire()
        return null
      }
    })().finally(() => {
      refreshing = null
    })
    return refreshing
  }

  async function send(path: string, init: RequestInit = {}, opts: SendOptions = {}): Promise<Response> {
    let res = await rawSend(path, init, opts, store.token)
    if (res.status === 401 && opts.authed) {
      const fresh = await refreshOnce()
      if (!fresh) throw new ApiError('unauthorized', 'Your session ended. Sign in again.')
      res = await rawSend(path, init, opts, fresh)
      if (res.status === 401) {
        store.expire()
        throw new ApiError('unauthorized', 'Your session ended. Sign in again.')
      }
    }
    if (!res.ok) throw await errorFrom(res)
    return res
  }

  const authed: SendOptions = { authed: true }
  const anonymous: SendOptions = { authed: false }

  const json = async <T>(path: string, init?: RequestInit, opts: SendOptions = authed): Promise<T> =>
    (await send(path, init, opts)).json() as Promise<T>

  const post = (body?: unknown): RequestInit => ({
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: body === undefined ? undefined : JSON.stringify(body),
  })

  const del = (body?: unknown): RequestInit => ({
    method: 'DELETE',
    headers: body === undefined ? {} : { 'Content-Type': 'application/json' },
    body: body === undefined ? undefined : JSON.stringify(body),
  })

  // The two calls that act on the refresh cookie must say they come from this page.
  const cookieCall: RequestInit = { method: 'POST', headers: { 'X-Requested-With': 'quizforge' } }

  /** Reads newline-delimited JSON: progress events, then one done or error event. */
  async function readIngestStream(res: Response, onProgress: (p: IngestProgress) => void): Promise<IngestedDocument> {
    const reader = res.body?.getReader()
    const text = reader ? null : await res.text()
    const decoder = new TextDecoder()
    let buffer = ''
    let done: IngestedDocument | null = null

    const handle = (line: string) => {
      if (!line.trim()) return
      const event = JSON.parse(line) as
        | ({ event: 'progress' } & IngestProgress)
        | { event: 'done'; document: IngestedDocument }
        | { event: 'error'; code?: string; message?: string }
      if (event.event === 'progress') onProgress({ page: event.page, pages: event.pages, tables: event.tables, figures: event.figures })
      else if (event.event === 'done') done = event.document
      else if (event.event === 'error') throw new ApiError(asCode(event.code), event.message ?? 'Could not read this file.')
    }

    if (text !== null) {
      text.split('\n').forEach(handle)
    } else if (reader) {
      for (;;) {
        const { value, done: finished } = await reader.read()
        if (value) buffer += decoder.decode(value, { stream: !finished })
        const lines = buffer.split('\n')
        buffer = lines.pop() ?? ''
        lines.forEach(handle)
        if (finished) break
      }
      handle(buffer)
    }
    if (!done) throw new ApiError('unknown', 'The server stopped before the file was read.')
    return done
  }

  const api: Api = {
    signup: (email, password, remember) => json<AuthResult>('/auth/signup', post({ email, password, remember }), anonymous),
    login: (email, password, remember) => json<AuthResult>('/auth/login', post({ email, password, remember }), anonymous),
    guest: () => json<AuthResult>('/auth/guest', post(), anonymous),
    upgrade: (email, password) => json<AuthResult>('/auth/upgrade', post({ email, password })),
    refresh: () => json<AuthResult>('/auth/refresh', cookieCall, anonymous),
    logout: async () => {
      await send('/auth/logout', cookieCall, anonymous)
    },
    me: () => json('/me'),
    changePassword: async (current, next) => {
      await send('/me/password', post({ current, new: next }), authed)
    },
    devices: () => json('/me/sessions'),
    signOutDevice: async (id) => {
      await send(`/me/sessions/${encodeURIComponent(id)}`, del(), authed)
    },
    signOutOtherDevices: async () => (await json<{ signedOut: number }>('/me/sessions/others', del())).signedOut,
    exportData: async () => (await send('/me/export', {}, authed)).blob(),
    deleteAccount: async (email) => {
      await send('/me', del({ email }), authed)
    },

    listDocuments: () => json('/documents'),
    deleteDocument: async (id) => {
      await send(`/documents/${encodeURIComponent(id)}`, del(), authed)
    },
    async ingestFile(file, creds, onProgress) {
      const form = new FormData()
      form.append('file', file, file.name)
      const res = await send('/documents', { method: 'POST', body: form }, { authed: true, creds })
      return readIngestStream(res, onProgress)
    },
    ingestUrl: async (url, creds) => {
      const res = await send('/documents/url', post({ url }), { authed: true, creds })
      return readIngestStream(res, () => {})
    },
    createQuiz: (req) => {
      const { provider, apiKey, ...rest } = req
      return json('/quizzes', post(rest), { authed: true, creds: { provider, apiKey } })
    },
    listQuizzes: () => json('/quizzes'),
    getQuiz: (id) => json(`/quizzes/${encodeURIComponent(id)}`),
    gradeQuiz: (id, answers, creds) => json(`/quizzes/${encodeURIComponent(id)}/grade`, post({ answers }), { authed: true, creds }),
    getResult: (quizId) => json(`/quizzes/${encodeURIComponent(quizId)}/result`),
    chat: (documentIds, message, creds) => json('/chat', post({ documentIds, message }), { authed: true, creds }),
  }
  return api
}
