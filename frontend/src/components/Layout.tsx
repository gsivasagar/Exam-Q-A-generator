import { useState } from 'react'
import { Link, NavLink, Outlet, useNavigate } from 'react-router-dom'
import { shortDate } from '../lib/agent'
import { useAuth } from '../state/Auth'

const icons = {
  plus: <path d="M12 5v14M5 12h14" />,
  quizzes: (
    <>
      <rect x="5" y="3" width="14" height="18" rx="2" />
      <path d="M9 8h6M9 12h6M9 16h3" />
    </>
  ),
  chat: <path d="M21 12a8 8 0 0 1-11.5 7.2L4 20l1-4.5A8 8 0 1 1 21 12z" />,
  out: <path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4M16 17l5-5-5-5M21 12H9" />,
}

function Icon({ name, size = 18 }: { name: keyof typeof icons; size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      {icons[name]}
    </svg>
  )
}

function AccountBox() {
  const { user, signOut } = useAuth()
  const navigate = useNavigate()
  if (!user) return null

  const leave = async () => {
    await signOut()
    navigate('/signin', { replace: true })
  }

  if (user.isGuest) {
    return (
      <div className="navbox">
        <div className="navbox__who">
          <span className="avatar avatar--guest" aria-hidden="true">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="12" cy="8" r="4" />
              <path d="M4 21a8 8 0 0 1 16 0" />
            </svg>
          </span>
          <span className="navbox__text">
            <strong>Guest</strong>
            <span>Kept until {shortDate(user.expiresAt)}</span>
          </span>
        </div>
        <Link to="/signup" className="button button--primary button--small">
          Create account
        </Link>
        <button type="button" className="link-button link-button--plain" onClick={leave}>
          Leave guest session
        </button>
      </div>
    )
  }

  return (
    <div className="navbox">
      <NavLink to="/account" className={({ isActive }) => `navbox__who navbox__link${isActive ? ' navbox__link--active' : ''}`}>
        <span className="avatar" aria-hidden="true">
          {(user.email ?? '?').charAt(0).toUpperCase()}
        </span>
        <span className="navbox__text">
          <strong>Account</strong>
          <span>{user.email}</span>
        </span>
      </NavLink>
      <button type="button" className="navbox__signout" onClick={leave}>
        <Icon name="out" size={16} />
        Sign out
      </button>
    </div>
  )
}

function SampleBanner() {
  const { api } = useAuth()
  if (!api.sample) return null
  return (
    <div className="banner banner--sample">
      <div>
        <strong>Sample data mode.</strong> Questions, pages and answers come from a built-in example set, not from your
        files. Start the back end and run <code>npm run dev</code> to use the real app.
      </div>
    </div>
  )
}

function GuestBanner() {
  const { user } = useAuth()
  const [hidden, setHidden] = useState(false)
  if (!user?.isGuest || hidden) return null
  return (
    <div className="banner" role="status">
      <div>
        <strong>You are using a guest session.</strong> Your files and quizzes are deleted on {shortDate(user.expiresAt)}. Create an
        account to keep them.
      </div>
      <div className="row">
        <Link to="/signup" className="button button--warn button--small">
          Keep my work
        </Link>
        <button type="button" className="button button--ghost button--small" onClick={() => setHidden(true)}>
          Not now
        </button>
      </div>
    </div>
  )
}

export function Layout() {
  const link = ({ isActive }: { isActive: boolean }) => `nav__link${isActive ? ' nav__link--active' : ''}`
  return (
    <div className="shell">
      <nav className="nav" aria-label="Main">
        <div className="brand">
          <div className="brand__mark">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#FFFFFF" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
              <path d="M4 5h16M4 12h10M4 19h16" />
            </svg>
          </div>
          <span className="brand__name">QuizForge</span>
        </div>
        <ul className="nav__list">
          <li>
            <NavLink to="/" end className={link}>
              <Icon name="plus" />
              New quiz
            </NavLink>
          </li>
          <li>
            <NavLink to="/quizzes" className={link}>
              <Icon name="quizzes" />
              My quizzes
            </NavLink>
          </li>
          <li>
            <NavLink to="/chat" className={link}>
              <Icon name="chat" />
              Chat with documents
            </NavLink>
          </li>
        </ul>
        <AccountBox />
      </nav>
      <div className="main">
        <GuestBanner />
        <main>
          <SampleBanner />
          <Outlet />
        </main>
      </div>
    </div>
  )
}
