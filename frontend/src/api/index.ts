import { createHttpApi } from './http'
import { createMockApi } from './mock'
import type { SessionStore } from './session'
import type { Api } from './types'

export * from './types'
export { SessionStore } from './session'

/** The real Python back end (through the dev proxy) unless VITE_API=mock asks for the built-in sample data. */
export function createApi(store: SessionStore): Api {
  return import.meta.env.VITE_API === 'mock' ? createMockApi() : createHttpApi('/api', undefined, store)
}
