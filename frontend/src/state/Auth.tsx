import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import type { Api, AuthResult, User } from '../api'
import type { SessionStore } from '../api/session'

export type AuthStatus = 'loading' | 'signedOut' | 'signedIn'

interface AuthState {
  api: Api
  status: AuthStatus
  user: User | null
  /** Set when the server ended the session (for example it expired), shown on the sign-in page. */
  notice: string
  signIn: (email: string, password: string, remember: boolean) => Promise<void>
  signUp: (email: string, password: string, remember: boolean) => Promise<void>
  /** A guest gives an email and password and keeps everything. */
  upgrade: (email: string, password: string) => Promise<void>
  startGuest: () => Promise<void>
  signOut: () => Promise<void>
  /** After the account was deleted or something else ended the session from this page. */
  forget: () => void
}

const Ctx = createContext<AuthState | null>(null)

export function AuthProvider({ api, store, children }: { api: Api; store: SessionStore; children: ReactNode }) {
  const [status, setStatus] = useState<AuthStatus>('loading')
  const [user, setUser] = useState<User | null>(null)
  const [notice, setNotice] = useState('')
  const started = useRef(false)

  const adopt = useCallback(
    (result: AuthResult) => {
      store.token = result.accessToken
      setUser(result.user)
      setStatus('signedIn')
      setNotice('')
    },
    [store],
  )

  const forget = useCallback(() => {
    store.token = null
    setUser(null)
    setStatus('signedOut')
  }, [store])

  // Come back to the site: the refresh cookie, if there is one, gives a new access token.
  useEffect(() => {
    if (started.current) return
    started.current = true
    api.refresh().then(adopt, forget)
  }, [api, adopt, forget])

  // The server said the session is over (expired, signed out elsewhere, or a copied token was spotted).
  useEffect(
    () =>
      store.onExpired(() => {
        setUser(null)
        setStatus('signedOut')
        setNotice('Your session ended. Sign in again.')
      }),
    [store],
  )

  const value = useMemo<AuthState>(
    () => ({
      api,
      status,
      user,
      notice,
      signIn: async (email, password, remember) => adopt(await api.login(email, password, remember)),
      signUp: async (email, password, remember) => adopt(await api.signup(email, password, remember)),
      upgrade: async (email, password) => adopt(await api.upgrade(email, password)),
      startGuest: async () => adopt(await api.guest()),
      signOut: async () => {
        try {
          await api.logout()
        } catch {
          // The server could not be reached. The person still leaves on this page; the session then ends on its own.
        } finally {
          forget()
        }
      },
      forget,
    }),
    [api, status, user, notice, adopt, forget],
  )
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>
}

export function useAuth(): AuthState {
  const ctx = useContext(Ctx)
  if (!ctx) throw new Error('useAuth must be used inside AuthProvider')
  return ctx
}
