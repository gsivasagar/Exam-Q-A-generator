import { render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { App } from '../App'
import { SessionStore, type Api } from '../api'
import { createMockApi } from '../api/mock'
import { AuthProvider } from '../state/Auth'

interface Options {
  /** Start signed out (no refresh cookie). The default is a returning, signed-in visitor. */
  signedIn?: boolean
  api?: Api
}

/** Renders the whole app at `route` on sample data with no waiting, and returns once it has settled. */
export async function renderApp(route = '/', { signedIn = true, api }: Options = {}) {
  const store = new SessionStore()
  const theApi = api ?? createMockApi({ delayMs: 0, signedInAs: signedIn ? 'tester@example.com' : undefined })
  render(
    <MemoryRouter initialEntries={[route]}>
      <AuthProvider api={theApi} store={store}>
        <App />
      </AuthProvider>
    </MemoryRouter>,
  )
  if (signedIn) await screen.findByRole('navigation', { name: 'Main' })
  return { api: theApi, store }
}

export const GOOD = 'a long test passphrase'

/** A mock API with one registered account that is currently signed out. */
export async function apiWithAccount(email = 'me@example.com', password = GOOD) {
  const api = createMockApi({ delayMs: 0 })
  await api.signup(email, password, true)
  await api.logout()
  return api
}

export const pdf = (name: string) => new File(['%PDF-1.4'], name, { type: 'application/pdf' })
