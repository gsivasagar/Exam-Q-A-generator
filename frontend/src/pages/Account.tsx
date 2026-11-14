import { useState, type FormEvent } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { ApiError, type Device, type Me } from '../api'
import { PasswordField } from '../components/PasswordField'
import { Chip, ErrorNote } from '../components/ui'
import { describeAgent } from '../lib/agent'
import { useAuth } from '../state/Auth'
import { useLoad } from '../state/useLoad'

const messageOf = (err: unknown) => (err instanceof ApiError ? err.message : 'Something went wrong. Try again.')

function PasswordSection() {
  const { api } = useAuth()
  const [current, setCurrent] = useState('')
  const [next, setNext] = useState('')
  const [error, setError] = useState('')
  const [done, setDone] = useState(false)
  const [busy, setBusy] = useState(false)

  const submit = async (e: FormEvent) => {
    e.preventDefault()
    setBusy(true)
    setError('')
    setDone(false)
    try {
      await api.changePassword(current, next)
      setCurrent('')
      setNext('')
      setDone(true)
    } catch (err) {
      setError(messageOf(err))
    } finally {
      setBusy(false)
    }
  }

  return (
    <form className="panel" onSubmit={submit} aria-labelledby="pw-h">
      <h2 id="pw-h">Password</h2>
      <div className="two-up">
        <PasswordField id="current" label="Current password" value={current} onChange={setCurrent} autoComplete="current-password" />
        <PasswordField id="new" label="New password" value={next} onChange={setNext} autoComplete="new-password" showLength />
      </div>
      {error && <ErrorNote message={error} />}
      {done && <div className="note note--success" role="status">Password changed. You were signed out on your other devices.</div>}
      <div className="row">
        <button type="submit" className="button button--secondary" disabled={busy || !current || !next}>
          {busy ? 'Changing…' : 'Change password'}
        </button>
        <span className="hint">Changing it signs you out on every other device.</span>
      </div>
    </form>
  )
}

function DevicesSection() {
  const { api } = useAuth()
  const [version, setVersion] = useState(0)
  const [error, setError] = useState('')
  const loaded = useLoad<Device[]>(() => api.devices(), `devices-${version}`)

  const run = async (work: () => Promise<unknown>) => {
    setError('')
    try {
      await work()
      setVersion((v) => v + 1)
    } catch (err) {
      setError(messageOf(err))
    }
  }

  return (
    <section className="panel" aria-labelledby="dev-h">
      <div className="panel__head">
        <h2 id="dev-h">Where you are signed in</h2>
        {loaded.status === 'ready' && loaded.data.length > 1 && (
          <button type="button" className="button button--secondary button--small" onClick={() => run(() => api.signOutOtherDevices())}>
            Sign out everywhere else
          </button>
        )}
      </div>
      {error && <ErrorNote message={error} />}
      {loaded.status === 'loading' && <p className="status" role="status">Loading…</p>}
      {loaded.status === 'error' && <ErrorNote message={loaded.message} />}
      {loaded.status === 'ready' && (
        <ul className="devices">
          {loaded.data.map((d) => (
            <li key={d.id}>
              <span>
                <strong>{describeAgent(d.userAgent)}</strong>
                <span className="muted-line">{d.current ? 'This browser. Active now.' : `Last used ${new Date(d.lastUsedAt).toLocaleDateString()}`}</span>
              </span>
              {d.current ? (
                <Chip tone="success">Current</Chip>
              ) : (
                <button type="button" className="button button--secondary button--small" onClick={() => run(() => api.signOutDevice(d.id))}>
                  Sign out
                </button>
              )}
            </li>
          ))}
        </ul>
      )}
    </section>
  )
}

function DataSection({ me }: { me: Me }) {
  const { api } = useAuth()
  const [error, setError] = useState('')
  const download = async () => {
    setError('')
    try {
      const blob = await api.exportData()
      const url = URL.createObjectURL(blob)
      const link = document.createElement('a')
      link.href = url
      link.download = 'quizforge-data.json'
      link.click()
      URL.revokeObjectURL(url)
    } catch (err) {
      setError(messageOf(err))
    }
  }
  return (
    <section className="panel" aria-labelledby="data-h">
      <h2 id="data-h">Your data</h2>
      <div className="counts">
        <div>
          <strong>{me.counts.files}</strong> files
        </div>
        <div>
          <strong>{me.counts.quizzes}</strong> quizzes
        </div>
        <div>
          <strong>{me.counts.reviewCards}</strong> review cards
        </div>
      </div>
      <p className="hint">Only you can see these. Your Gemini key is never saved: you enter it each time you open QuizForge.</p>
      {error && <ErrorNote message={error} />}
      <div>
        <button type="button" className="button button--secondary" onClick={download}>
          Download my data (JSON)
        </button>
      </div>
    </section>
  )
}

function DeleteSection({ email }: { email: string }) {
  const { api, forget } = useAuth()
  const navigate = useNavigate()
  const [typed, setTyped] = useState('')
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const matches = typed.trim().toLowerCase() === email.toLowerCase()

  const remove = async (e: FormEvent) => {
    e.preventDefault()
    if (!matches) return
    setBusy(true)
    setError('')
    try {
      await api.deleteAccount(typed)
      forget()
      navigate('/signin', { replace: true })
    } catch (err) {
      setError(messageOf(err))
      setBusy(false)
    }
  }

  return (
    <form className="panel panel--danger" onSubmit={remove} aria-labelledby="del-h">
      <h2 id="del-h">Delete account</h2>
      <p>This removes your account, your files and their search index, your quizzes, results and review cards. It cannot be undone.</p>
      <div className="field field--narrow">
        <label htmlFor="confirm">Type your email to confirm</label>
        <input id="confirm" type="email" placeholder={email} value={typed} onChange={(e) => setTyped(e.target.value)} />
      </div>
      {error && <ErrorNote message={error} />}
      <div>
        <button type="submit" className="button button--danger" disabled={!matches || busy}>
          {busy ? 'Deleting…' : 'Delete my account and data'}
        </button>
      </div>
    </form>
  )
}

export function AccountPage() {
  const { api, user } = useAuth()
  const me = useLoad<Me>(() => api.me(), user?.id ?? 'none')

  if (user?.isGuest) {
    return (
      <div className="page">
        <header className="page__head">
          <h1>Account</h1>
          <p>You are using a guest session, so there is no password or email yet.</p>
        </header>
        <div className="card empty">
          <h2>Keep your work</h2>
          <p>Create an account to keep your files and quizzes beyond the guest period.</p>
          <Link to="/signup" className="button button--primary">
            Create account
          </Link>
        </div>
      </div>
    )
  }

  return (
    <div className="page page--narrow">
      <header className="page__head">
        <h1>Account</h1>
        <p>{user?.email}</p>
      </header>
      <PasswordSection />
      <DevicesSection />
      {me.status === 'ready' && <DataSection me={me.data} />}
      {me.status === 'error' && <ErrorNote message={me.message} />}
      {user?.email && <DeleteSection email={user.email} />}
    </div>
  )
}
