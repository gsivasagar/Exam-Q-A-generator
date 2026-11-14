import { useState, type FormEvent } from 'react'
import { Link, Navigate, useNavigate } from 'react-router-dom'
import { ApiError, type Me } from '../api'
import { AuthShell } from '../components/AuthShell'
import { PasswordField } from '../components/PasswordField'
import { ErrorNote } from '../components/ui'
import { useAuth } from '../state/Auth'
import { useLoad } from '../state/useLoad'

export function SignUpPage() {
  const auth = useAuth()
  const navigate = useNavigate()
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const isGuest = auth.user?.isGuest ?? false
  // A guest is upgrading: say what moves into the account.
  const mine = useLoad<Me | null>(() => (isGuest ? auth.api.me() : Promise.resolve(null)), String(isGuest))

  if (auth.status === 'signedIn' && !isGuest) return <Navigate to="/" replace />

  const submit = async (e: FormEvent) => {
    e.preventDefault()
    setBusy(true)
    setError('')
    try {
      await (isGuest ? auth.upgrade(email, password) : auth.signUp(email, password, true))
      navigate('/', { replace: true })
    } catch (err) {
      setError(err instanceof ApiError ? err.message : 'Something went wrong. Try again.')
      setBusy(false)
    }
  }

  const counts = mine.status === 'ready' && mine.data ? mine.data.counts : null
  return (
    <AuthShell
      title="Keep your quizzes, scores and review queue."
      message="An account saves your work between visits and devices. We store your email and a hash of your password, never the password itself, and never your Gemini key."
    >
      <form className="auth__card" onSubmit={submit}>
        <header>
          <h1>Create an account</h1>
          <p>
            Already have one? <Link to="/signin">Sign in</Link>
          </p>
        </header>
        {counts && (
          <div className="note note--accent" role="status">
            Your guest work moves into this account: {counts.files} {counts.files === 1 ? 'file' : 'files'} and {counts.quizzes}{' '}
            {counts.quizzes === 1 ? 'quiz' : 'quizzes'}.
          </div>
        )}
        {error && <ErrorNote message={error} />}
        <div className="field">
          <label htmlFor="email">Email</label>
          <input id="email" type="email" autoComplete="email" value={email} onChange={(e) => setEmail(e.target.value)} required />
        </div>
        <PasswordField id="password" label="Password" value={password} onChange={setPassword} autoComplete="new-password" showLength />
        <button type="submit" className="button button--primary" disabled={busy}>
          {busy ? 'Creating account…' : 'Create account'}
        </button>
        <p className="hint">You can delete your account and everything in it from the Account page at any time.</p>
      </form>
    </AuthShell>
  )
}
