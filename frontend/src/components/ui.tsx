import type { ReactNode } from 'react'
import type { SourceRef } from '../api'

export type Tone = 'neutral' | 'accent' | 'success' | 'warn' | 'danger'

export function Chip({ tone = 'neutral', children }: { tone?: Tone; children: ReactNode }) {
  return <span className={`chip chip--${tone}`}>{children}</span>
}

export function ProgressBar({ value, max, label }: { value: number; max: number; label: string }) {
  const pct = max > 0 ? Math.min(100, Math.round((value / max) * 100)) : 0
  return (
    <div className="progress" role="progressbar" aria-valuemin={0} aria-valuemax={max} aria-valuenow={value} aria-label={label}>
      <div className="progress__fill" style={{ width: `${pct}%` }} />
    </div>
  )
}

export function sourceLabel(source: SourceRef): string {
  return source.page > 0 ? `${source.file}, page ${source.page}` : source.file
}

export function SourceChip({ source }: { source: SourceRef }) {
  return (
    <span className="source-chip">
      <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
        <path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z" />
        <path d="M14 3v5h5" />
      </svg>
      {sourceLabel(source)}
    </span>
  )
}

export function ErrorNote({ message }: { message: string }) {
  return (
    <div className="note note--danger" role="alert">
      {message}
    </div>
  )
}

export function percent(score: number): string {
  return `${Math.round(score * 100)}%`
}
