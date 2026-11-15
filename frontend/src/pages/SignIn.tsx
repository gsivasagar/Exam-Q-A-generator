import { useState, type FormEvent } from 'react'
import { Link, Navigate, useLocation, useNavigate } from 'react-router-dom'
import { ApiError } from '../api'
import { AuthShell } from '../components/AuthShell'
import { PasswordField } from '../components/PasswordField'
import { ErrorNote } from '../components/ui'
import { useAuth } from '../state/Auth'

export function SignInPage() {
  const auth = useAuth()
  const navigate = useNavigate()
  const location = useLocation() as { state?: { from?: string } }
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [remember, setRemember] = useState(true)
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)

  if (auth.status === 'signedIn') return <Navigate to={location.state?.from ?? '/'} replace />

  const go = () => navigate(location.state?.from ?? '/', { replace: true })
  const fail = (err: unknown) => {
    setError(err instanceof ApiError ? err.message : 'Something went wrong. Try again.')
    setBusy(false)
  }

  const submit = async (e: FormEvent) => {
    e.preventDefault()
    setBusy(true)
    setError('')
    try {
      await auth.signIn(email, password, remember)
      go()
    } catch (err) {
      fail(err)
    }
  }

  const guest = async () => {
    setBusy(true)
    setError('')
    try {
      await auth.startGuest()
      go()
    } catch (err) {
      fail(err)
    }
  }

  const message = error || auth.notice
  return (
    <AuthShell
      title="Turn your notes into practice that points back to the page."
      message="Upload PDFs, slides, scans or images. Get a quiz, see which key points you missed, and ask questions with every answer cited."
    >
      <form className="auth__card" onSubmit={submit}>
        <header>
          <h1>Sign in</h1>
          <p>
            New here? <Link to="/signup">Create an account</Link>
          </p>
        </header>
        {auth.api.sample && (
          <div className="note note--warn">
            <strong>Sample data mode</strong>
            Accounts and quizzes here are examples kept in this tab only; nothing reads your files.
          </div>
        )}
        {message && <ErrorNote message={message} />}
        <div className="field">
          <label htmlFor="email">Email</label>
          <input id="email" type="email" autoComplete="email" value={email} onChange={(e) => setEmail(e.target.value)} required />
        </div>
        <PasswordField id="password" label="Password" value={password} onChange={setPassword} autoComplete="current-password" />
        <label className="check">
          <input type="checkbox" checked={remember} onChange={(e) => setRemember(e.target.checked)} />
          Keep me signed in on this device
        </label>
        <button type="submit" className="button button--primary" disabled={busy || auth.status === 'loading'}>
          {busy ? 'Signing in…' : 'Sign in'}
        </button>
        <div className="divider">or</div>
        <button type="button" className="button button--secondary" onClick={guest} disabled={busy}>
          Try it without an account
        </button>
        <p className="hint hint--center">A guest session keeps your work for 7 days. Create an account later to keep it.</p>
      </form>
    </AuthShell>
  )
}
