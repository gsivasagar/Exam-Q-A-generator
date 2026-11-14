import { Navigate, useLocation } from 'react-router-dom'
import { useAuth } from '../state/Auth'
import { Layout } from './Layout'

/** Everything inside the app needs a session (a guest counts). Signed-out visitors go to the sign-in page. */
export function RequireSession() {
  const auth = useAuth()
  const location = useLocation()
  if (auth.status === 'loading') return <p className="splash" role="status">Loading…</p>
  if (auth.status === 'signedOut' || !auth.user) {
    return <Navigate to="/signin" replace state={{ from: location.pathname + location.search }} />
  }
  return <Layout />
}
