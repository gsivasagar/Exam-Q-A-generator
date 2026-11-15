import { describe, expect, it, vi } from 'vitest'
import { createHttpApi } from './http'
import { SessionStore } from './session'
import { ApiError } from './types'

const user = { id: 'u1', email: 'a@b.co', isGuest: false, expiresAt: null }
const auth = (token: string) => new Response(JSON.stringify({ accessToken: token, expiresIn: 900, user }), { status: 200 })
const status = (code: number, body: object = {}) => new Response(JSON.stringify(body), { status: code })

interface Call {
  path: string
  headers: Record<string, string>
  credentials?: string
}

/** A fake server: `/auth/refresh` hands out new tokens, other paths accept only the newest token. */
function server(options: { refreshWorks?: boolean; accepted?: () => string } = {}) {
  const { refreshWorks = true } = options
  const calls: Call[] = []
  let valid = 'token-1'
  const fetchMock = vi.fn(async (url: string, init: RequestInit = {}) => {
    const path = url.replace('/api', '')
    const headers = (init.headers ?? {}) as Record<string, string>
    calls.push({ path, headers, credentials: init.credentials })
    if (path === '/auth/refresh') {
      if (!refreshWorks) return status(401, { code: 'unauthorized', message: 'Your session ended. Sign in again.' })
      valid = 'token-2'
      return auth('token-2')
    }
    if (path === '/auth/login') return status(401, { code: 'invalid_credentials', message: 'Email or password is wrong.' })
    if (headers.Authorization !== `Bearer ${valid}`) return status(401, { code: 'unauthorized', message: 'Sign in to continue.' })
    return new Response(JSON.stringify([]), { status: 200 })
  })
  return { fetchMock: fetchMock as unknown as typeof fetch, calls }
}

describe('sending the access token', () => {
  it('adds it to calls that need a session and keeps it off sign-in and sign-up', async () => {
    const { fetchMock, calls } = server()
    const store = new SessionStore()
    store.token = 'token-1'
    const api = createHttpApi('/api', fetchMock, store)
    await api.listQuizzes()
    await api.login('a@b.co', 'x', true).catch(() => undefined)
    expect(calls[0].headers.Authorization).toBe('Bearer token-1')
    expect(calls[1].headers.Authorization).toBeUndefined()
  })

  it('always lets the browser send and receive the cookie', async () => {
    const { fetchMock, calls } = server()
    const api = createHttpApi('/api', fetchMock, new SessionStore())
    await api.refresh()
    expect(calls[0].credentials).toBe('include')
  })
})

describe('an expired access token', () => {
  it('refreshes once, retries with the new token and remembers it', async () => {
    const { fetchMock, calls } = server()
    const store = new SessionStore()
    store.token = 'stale'
    const api = createHttpApi('/api', fetchMock, store)
    expect(await api.listQuizzes()).toEqual([])
    expect(calls.map((c) => c.path)).toEqual(['/quizzes', '/auth/refresh', '/quizzes'])
    expect(calls[2].headers.Authorization).toBe('Bearer token-2')
    expect(store.token).toBe('token-2')
  })

  it('sends the cross-site header on refresh and no access token', async () => {
    const { fetchMock, calls } = server()
    const store = new SessionStore()
    store.token = 'stale'
    await createHttpApi('/api', fetchMock, store).listQuizzes()
    const refresh = calls.find((c) => c.path === '/auth/refresh')!
    expect(refresh.headers['X-Requested-With']).toBe('quizforge')
    expect(refresh.headers.Authorization).toBeUndefined()
  })

  it('shares one refresh between calls that fail together', async () => {
    const { fetchMock, calls } = server()
    const store = new SessionStore()
    store.token = 'stale'
    const api = createHttpApi('/api', fetchMock, store)
    await Promise.all([api.listQuizzes(), api.listDocuments(), api.me().catch(() => undefined)])
    expect(calls.filter((c) => c.path === '/auth/refresh')).toHaveLength(1)
  })

  it('ends the session when refresh fails, and does not loop', async () => {
    const { fetchMock, calls } = server({ refreshWorks: false })
    const store = new SessionStore()
    store.token = 'stale'
    const expired = vi.fn()
    store.onExpired(expired)
    const err = await createHttpApi('/api', fetchMock, store).listQuizzes().catch((e) => e)
    expect(err).toBeInstanceOf(ApiError)
    expect(err.code).toBe('unauthorized')
    expect(expired).toHaveBeenCalledTimes(1)
    expect(store.token).toBeNull()
    expect(calls.map((c) => c.path)).toEqual(['/quizzes', '/auth/refresh'])
  })

  it('ends the session when the retry is refused as well', async () => {
    const fetchMock = vi.fn(async (url: string) =>
      url.endsWith('/auth/refresh') ? auth('token-2') : status(401, { code: 'unauthorized', message: 'no' }),
    ) as unknown as typeof fetch
    const store = new SessionStore()
    const expired = vi.fn()
    store.onExpired(expired)
    store.token = 'stale'
    await expect(createHttpApi('/api', fetchMock, store).listQuizzes()).rejects.toMatchObject({ code: 'unauthorized' })
    expect(expired).toHaveBeenCalledTimes(1)
  })

  it('a wrong password is an error, not a reason to refresh', async () => {
    const { fetchMock, calls } = server()
    const store = new SessionStore()
    const expired = vi.fn()
    store.onExpired(expired)
    const err = await createHttpApi('/api', fetchMock, store).login('a@b.co', 'nope', true).catch((e) => e)
    expect(err.code).toBe('invalid_credentials')
    expect(calls).toHaveLength(1)
    expect(expired).not.toHaveBeenCalled()
  })

  it('does not try to refresh the refresh call itself', async () => {
    const { fetchMock, calls } = server({ refreshWorks: false })
    await expect(createHttpApi('/api', fetchMock, new SessionStore()).refresh()).rejects.toMatchObject({ code: 'unauthorized' })
    expect(calls).toHaveLength(1)
  })
})

describe('account calls', () => {
  it('sends the cookie header on sign-out and returns the downloaded data as a blob', async () => {
    const fetchMock = vi.fn(async (url: string) => (url.endsWith('/auth/logout') ? new Response(null, { status: 204 }) : new Response('{"a":1}', { status: 200 }))) as unknown as typeof fetch
    const store = new SessionStore()
    store.token = 't'
    const api = createHttpApi('/api', fetchMock, store)
    await api.logout()
    expect((fetchMock as unknown as ReturnType<typeof vi.fn>).mock.calls[0][1].headers['X-Requested-With']).toBe('quizforge')
    const blob = await api.exportData()
    expect(await blob.text()).toBe('{"a":1}')
  })

  it('changes a password with the right field names', async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(null, { status: 204 }))
    const store = new SessionStore()
    store.token = 't'
    await createHttpApi('/api', fetchMock, store).changePassword('old one here', 'new one here!')
    expect(JSON.parse(fetchMock.mock.calls[0][1].body)).toEqual({ current: 'old one here', new: 'new one here!' })
  })

  it('confirms an account deletion with the email', async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(null, { status: 204 }))
    const store = new SessionStore()
    store.token = 't'
    await createHttpApi('/api', fetchMock, store).deleteAccount('a@b.co')
    expect(fetchMock.mock.calls[0][0]).toBe('/api/me')
    expect(fetchMock.mock.calls[0][1].method).toBe('DELETE')
    expect(JSON.parse(fetchMock.mock.calls[0][1].body)).toEqual({ email: 'a@b.co' })
  })
})
