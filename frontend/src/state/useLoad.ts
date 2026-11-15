import { useEffect, useState } from 'react'
import { ApiError } from '../api'

export type Loaded<T> = { status: 'loading' } | { status: 'error'; message: string } | { status: 'ready'; data: T }

/** Runs `load` when `key` changes and reports loading, error or data. Ignores answers that arrive late. */
export function useLoad<T>(load: () => Promise<T>, key: string): Loaded<T> {
  const [state, setState] = useState<Loaded<T>>({ status: 'loading' })
  useEffect(() => {
    let current = true
    setState({ status: 'loading' })
    load().then(
      (data) => current && setState({ status: 'ready', data }),
      (err: unknown) =>
        current && setState({ status: 'error', message: err instanceof ApiError ? err.message : 'Something went wrong.' }),
    )
    return () => {
      current = false
    }
    // `load` is a new closure on every render; `key` says when the request really changes.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key])
  return state
}
