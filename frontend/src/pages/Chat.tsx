import { Fragment, useState, type FormEvent, type KeyboardEvent } from 'react'
import { Link } from 'react-router-dom'
import { ApiError, type ChatAnswer, type Citation } from '../api'
import { ErrorNote, sourceLabel } from '../components/ui'
import { useApp } from '../state/AppState'

type Message =
  | { id: number; role: 'user'; text: string }
  | { id: number; role: 'assistant'; answer: ChatAnswer }
  | { id: number; role: 'error'; text: string }

function CitedText({ text, onCite }: { text: string; onCite: (n: number) => void }) {
  const parts = text.split(/\[(\d+)\]/)
  return (
    <p>
      {parts.map((part, i) =>
        i % 2 === 1 ? (
          <button key={i} type="button" className="cite" aria-label={`Source ${part}`} onClick={() => onCite(Number(part))}>
            {part}
          </button>
        ) : (
          <Fragment key={i}>{part}</Fragment>
        ),
      )}
    </p>
  )
}

function SourceCard({ c, active }: { c: Citation; active: boolean }) {
  return (
    <article className={`card source${active ? ' source--active' : ''}`} id={`source-${c.n}`}>
      <div className="source__head">
        <span className="source__title">
          <span className="cite cite--solid">{c.n}</span>
          {sourceLabel(c.source)}
        </span>
        {c.table && c.readWith === 'gemini' && <span className="tag">Table, read with Gemini</span>}
      </div>
      {c.table ? (
        <>
          <div className="source__caption">{c.table.caption}</div>
          <div className="table-scroll">
            <table>
              <thead>
                <tr>
                  {c.table.header.map((h) => (
                    <th key={h} scope="col">
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {c.table.rows.map((row, ri) => (
                  <tr key={ri} className={ri === c.table!.highlightRow ? 'is-highlight' : ''}>
                    {row.map((cell, ci) => (
                      <td key={ci}>{cell}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      ) : (
        <p className="source__passage">{c.passage}</p>
      )}
    </article>
  )
}

export function ChatPage() {
  const app = useApp()
  const [excluded, setExcluded] = useState<Set<string>>(new Set())
  const [messages, setMessages] = useState<Message[]>([])
  const [draft, setDraft] = useState('')
  const [pending, setPending] = useState(false)
  const [activeId, setActiveId] = useState<number | null>(null)
  const [activeN, setActiveN] = useState<number | null>(null)

  const scope = app.readyDocs.filter((d) => !excluded.has(d.id))
  const answers = messages.filter((m): m is Extract<Message, { role: 'assistant' }> => m.role === 'assistant')
  const shown = answers.find((m) => m.id === activeId) ?? answers[answers.length - 1]
  const citations = shown && shown.answer.kind === 'answer' ? shown.answer.citations : []

  const send = async (e?: FormEvent) => {
    e?.preventDefault()
    const text = draft.trim()
    if (!text || pending || scope.length === 0) return
    const base = messages.length
    setMessages((m) => [...m, { id: base, role: 'user', text }])
    setDraft('')
    setPending(true)
    try {
      const answer = await app.api.chat(scope.map((d) => d.id), text, app.creds)
      setMessages((m) => [...m, { id: base + 1, role: 'assistant', answer }])
      setActiveId(base + 1)
      setActiveN(null)
    } catch (err) {
      setMessages((m) => [...m, { id: base + 1, role: 'error', text: err instanceof ApiError ? err.message : 'Something went wrong.' }])
    } finally {
      setPending(false)
    }
  }

  const onKey = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      void send()
    }
  }

  const toggle = (id: string) =>
    setExcluded((s) => {
      const next = new Set(s)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return next
    })

  return (
    <div className="page page--chat">
      <header className="page__head">
        <h1>Chat with your documents</h1>
        <div className="scope" role="group" aria-label="Files to search">
          <span className="scope__label">Searching in</span>
          {app.readyDocs.map((d) => (
            <label key={d.id} className={`scope__file${excluded.has(d.id) ? '' : ' is-on'}`}>
              <input type="checkbox" checked={!excluded.has(d.id)} onChange={() => toggle(d.id)} />
              {d.name}
            </label>
          ))}
          <Link to="/" className="scope__add">
            Add files
          </Link>
        </div>
      </header>

      {app.readyDocs.length === 0 ? (
        <div className="card empty">
          <h2>No files yet</h2>
          <p>Add a file first, then ask questions about it. Every answer points to the page it came from.</p>
          <Link to="/" className="button button--primary">
            Add files
          </Link>
        </div>
      ) : (
        <div className="chat">
          <section className="card chat__main" aria-label="Conversation">
            <div className="chat__log" aria-live="polite">
              {messages.length === 0 && <p className="muted-line">Ask a question about the selected files.</p>}
              {messages.map((m) => {
                if (m.role === 'user') return <div key={m.id} className="bubble bubble--user">{m.text}</div>
                if (m.role === 'error') return <ErrorNote key={m.id} message={m.text} />
                if (m.answer.kind === 'refusal') {
                  return (
                    <div key={m.id} className="bubble bubble--bot">
                      <div className="note note--warn">
                        <strong>Not in your documents</strong>
                        {m.answer.text}
                      </div>
                      <div className="row">
                        <Link to="/" className="button button--secondary button--small">
                          Add more files
                        </Link>
                      </div>
                    </div>
                  )
                }
                const { text, citations: cites } = m.answer
                return (
                  <div key={m.id} className="bubble bubble--bot">
                    <CitedText
                      text={text}
                      onCite={(n) => {
                        setActiveId(m.id)
                        setActiveN(n)
                      }}
                    />
                    <div className="chips">
                      {cites.map((c) => (
                        <button
                          key={c.n}
                          type="button"
                          className="source-chip source-chip--button"
                          onClick={() => {
                            setActiveId(m.id)
                            setActiveN(c.n)
                          }}
                        >
                          {sourceLabel(c.source)}
                        </button>
                      ))}
                    </div>
                  </div>
                )
              })}
              {pending && <p className="muted-line" role="status">Searching your files…</p>}
            </div>

            <form className="composer" onSubmit={send}>
              <div className="composer__field">
                <label htmlFor="ask" className="sr-only">
                  Ask a question about your documents
                </label>
                <textarea id="ask" rows={2} placeholder="Ask about your documents" value={draft} onChange={(e) => setDraft(e.target.value)} onKeyDown={onKey} />
                <span className="hint">
                  {scope.length === 0 ? 'Select at least one file to search.' : 'Answers use only the selected files. Every claim links to a page.'}
                </span>
              </div>
              <button type="submit" className="button button--primary composer__send" aria-label="Send" disabled={pending || !draft.trim() || scope.length === 0}>
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                  <path d="M5 12h14M13 6l6 6-6 6" />
                </svg>
              </button>
            </form>
          </section>

          <aside className="chat__side stack" aria-label="Sources">
            <h2 className="section-title">Sources for this answer</h2>
            {citations.length === 0 ? (
              <p className="muted-line">
                {shown?.answer.kind === 'refusal'
                  ? 'No sources for this reply: nothing in the selected files answered the question.'
                  : 'Sources appear here when an answer cites your files.'}
              </p>
            ) : (
              citations.map((c) => <SourceCard key={c.n} c={c} active={activeN === c.n} />)
            )}
            {citations.length > 0 && <p className="hint">Passages are shown as they were read from your files, so you can check the answer yourself.</p>}
          </aside>
        </div>
      )}
    </div>
  )
}
