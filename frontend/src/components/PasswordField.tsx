import { useState } from 'react'

export const MIN_PASSWORD = 12

interface Props {
  id: string
  label: string
  value: string
  onChange: (value: string) => void
  autoComplete: 'current-password' | 'new-password'
  /** Show how long the password is against the minimum (sign-up and change password). */
  showLength?: boolean
}

export function PasswordField({ id, label, value, onChange, autoComplete, showLength }: Props) {
  const [shown, setShown] = useState(false)
  const long = value.length >= MIN_PASSWORD
  return (
    <div className="field">
      <label htmlFor={id}>{label}</label>
      <div className="password">
        <input
          id={id}
          type={shown ? 'text' : 'password'}
          autoComplete={autoComplete}
          value={value}
          aria-describedby={showLength ? `${id}-help` : undefined}
          onChange={(e) => onChange(e.target.value)}
        />
        <button type="button" className="password__toggle" aria-label={shown ? 'Hide password' : 'Show password'} aria-pressed={shown} onClick={() => setShown(!shown)}>
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
            {shown ? (
              <>
                <path d="M3 3l18 18M10.6 5.1A10 10 0 0 1 12 5c6.5 0 10 7 10 7a17 17 0 0 1-3.1 4M6.6 6.6A17 17 0 0 0 2 12s3.5 7 10 7a9.8 9.8 0 0 0 5.4-1.6" />
                <path d="M9.9 9.9a3 3 0 0 0 4.2 4.2" />
              </>
            ) : (
              <>
                <path d="M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7S2 12 2 12z" />
                <circle cx="12" cy="12" r="3" />
              </>
            )}
          </svg>
        </button>
      </div>
      {showLength && (
        <div id={`${id}-help`} className={`length-hint${long ? ' length-hint--ok' : ''}`} role="status">
          {value.length === 0
            ? `At least ${MIN_PASSWORD} characters. A few words with spaces or dashes works well.`
            : long
              ? `${value.length} characters. Long enough.`
              : `${value.length} characters. At least ${MIN_PASSWORD} needed.`}
        </div>
      )}
    </div>
  )
}
