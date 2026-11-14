/** The access token, held in memory only (never localStorage). A page reload forgets it; the refresh cookie
 * then brings it back. */
export class SessionStore {
  token: string | null = null
  private listeners = new Set<() => void>()

  /** Called when the session is over and the person has to sign in again. Returns an unsubscribe function. */
  onExpired(fn: () => void): () => void {
    this.listeners.add(fn)
    return () => this.listeners.delete(fn)
  }

  expire(): void {
    this.token = null
    this.listeners.forEach((fn) => fn())
  }
}
