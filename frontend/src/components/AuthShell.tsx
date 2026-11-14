import type { ReactNode } from 'react'

/** The two-column frame shared by sign-in and sign-up: a brand panel and the form. */
export function AuthShell({ title, message, children }: { title: string; message: string; children: ReactNode }) {
  return (
    <div className="auth">
      <section className="auth__brand" aria-label="About QuizForge">
        <div className="brand brand--light">
          <div className="brand__mark brand__mark--light">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#0F766E" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
              <path d="M4 5h16M4 12h10M4 19h16" />
            </svg>
          </div>
          <span className="brand__name">QuizForge</span>
        </div>
        <div className="auth__pitch">
          <h2>{title}</h2>
          <p>{message}</p>
        </div>
      </section>
      <main className="auth__form">{children}</main>
    </div>
  )
}
