import { useRef, useState, type DragEvent, type FormEvent } from 'react'
import { useNavigate, useSearchParams } from 'react-router-dom'
import { ApiError, type Difficulty, type QuestionType } from '../api'
import { Chip, ErrorNote, ProgressBar } from '../components/ui'
import { useApp, type Upload } from '../state/AppState'

const TYPES: { value: QuestionType; label: string }[] = [
  { value: 'short', label: 'Short answer' },
  { value: 'mcq', label: 'Multiple choice' },
  { value: 'mixed', label: 'Mixed' },
  { value: 'tf', label: 'True / False' },
  { value: 'cloze', label: 'Fill in the blank' },
]

const LEVELS: { value: Difficulty; label: string }[] = [
  { value: 'any', label: 'Any' },
  { value: 'easy', label: 'Easy' },
  { value: 'medium', label: 'Medium' },
  { value: 'hard', label: 'Hard' },
]

function FileRow({ upload, onRemove }: { upload: Upload; onRemove: () => void }) {
  const { status, progress, doc, error } = upload
  const detail =
    status === 'error'
      ? error
      : status === 'reading'
        ? progress
          ? `${progress.pages} pages. No text layer, so each page is being read as an image.`
          : 'Reading…'
        : doc?.method === 'image'
          ? 'Image read with Gemini. Text and labels found.'
          : doc?.method === 'ocr'
            ? `${doc.pages} pages. Scanned, read with Gemini.`
            : `${doc?.pages ? `${doc.pages} pages. ` : ''}Text read directly.`

  return (
    <li className={`file${status === 'error' ? ' file--error' : ''}`}>
      <div className="file__body">
        <div className="file__head">
          <span className="file__name">{upload.name}</span>
          {status === 'ready' && <Chip tone="success">Ready</Chip>}
          {status === 'reading' && <Chip tone="accent">{progress ? 'Reading with Gemini' : 'Reading'}</Chip>}
          {status === 'error' && <Chip tone="danger">Not added</Chip>}
        </div>
        <div className={`file__detail${status === 'error' ? ' file__detail--error' : ''}`}>{detail}</div>
        {status === 'reading' && progress && (
          <>
            <ProgressBar value={progress.page} max={progress.pages} label={`Pages read in ${upload.name}`} />
            <div className="file__stats">
              <span>
                Page {progress.page} of {progress.pages}
              </span>
              {progress.tables > 0 && <span className="tag">{progress.tables} {progress.tables === 1 ? 'table' : 'tables'} kept</span>}
              {progress.figures > 0 && <span className="tag">{progress.figures} {progress.figures === 1 ? 'figure' : 'figures'} described</span>}
            </div>
          </>
        )}
        {status === 'ready' && doc && (doc.tables > 0 || doc.figures > 0) && (
          <div className="file__stats">
            {doc.tables > 0 && <span className="tag">{doc.tables} tables kept</span>}
            {doc.figures > 0 && <span className="tag">{doc.figures} figures described</span>}
          </div>
        )}
      </div>
      <button type="button" className="icon-button" onClick={onRemove} aria-label={`Remove ${upload.name}`}>
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" aria-hidden="true">
          <path d="M6 6l12 12M18 6L6 18" />
        </svg>
      </button>
    </li>
  )
}

export function NewQuiz() {
  const app = useApp()
  const navigate = useNavigate()
  const fileInput = useRef<HTMLInputElement>(null)
  const [dragging, setDragging] = useState(false)
  const [url, setUrl] = useState('')
  const [count, setCount] = useState(10)
  const [type, setType] = useState<QuestionType>('mcq')
  const [difficulty, setDifficulty] = useState<Difficulty>('any')
  const [params] = useSearchParams()
  const [topic, setTopic] = useState(params.get('topic') ?? '')
  const [creating, setCreating] = useState(false)
  const [error, setError] = useState('')

  const reading = app.uploads.some((u) => u.status === 'reading')
  const needsKey = app.creds.provider === 'Gemini' && !app.creds.apiKey.trim()
  const canCreate = app.readyDocs.length > 0 && !reading && !creating && !needsKey

  const onDrop = (e: DragEvent) => {
    e.preventDefault()
    setDragging(false)
    app.addFiles([...e.dataTransfer.files])
  }

  const addUrl = (e: FormEvent) => {
    e.preventDefault()
    if (!url.trim()) return
    app.addUrl(url.trim())
    setUrl('')
  }

  const create = async (e: FormEvent) => {
    e.preventDefault()
    if (!canCreate) return
    setCreating(true)
    setError('')
    try {
      const quiz = await app.api.createQuiz({
        documentIds: app.readyDocs.map((d) => d.id),
        count,
        type,
        difficulty,
        topic,
        ...app.creds,
      })
      navigate(`/quiz/${quiz.id}`)
    } catch (err) {
      setError(err instanceof ApiError ? err.message : 'Could not create the quiz.')
      setCreating(false)
    }
  }

  return (
    <div className="page">
      <header className="page__head">
        <h1>New quiz</h1>
        <p>Add your study material, then choose what to practise. Every question will point back to a file and page.</p>
      </header>

      <div className="split">
        <section className="split__main" aria-label="Study material">
          <div
            className={`dropzone${dragging ? ' dropzone--active' : ''}`}
            onDragOver={(e) => {
              e.preventDefault()
              setDragging(true)
            }}
            onDragLeave={() => setDragging(false)}
            onDrop={onDrop}
          >
            <div className="dropzone__icon">
              <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                <path d="M12 16V4M7 9l5-5 5 5M4 20h16" />
              </svg>
            </div>
            <div className="dropzone__title">
              Drop files here, or{' '}
              <button type="button" className="link-button" onClick={() => fileInput.current?.click()}>
                choose files
              </button>
            </div>
            <div className="dropzone__hint">PDF, Word, PowerPoint, Markdown, text, or images (PNG, JPEG, WebP). Up to 25 MB each.</div>
            <input
              ref={fileInput}
              type="file"
              multiple
              hidden
              aria-label="Choose files"
              accept=".pdf,.docx,.pptx,.md,.markdown,.txt,.png,.jpg,.jpeg,.webp"
              onChange={(e) => {
                app.addFiles([...(e.target.files ?? [])])
                e.target.value = ''
              }}
            />
          </div>

          <form className="field" onSubmit={addUrl}>
            <label htmlFor="url">Or paste a web page address</label>
            <div className="row">
              <input id="url" type="url" placeholder="https://" value={url} onChange={(e) => setUrl(e.target.value)} />
              <button type="submit" className="button button--secondary" disabled={!url.trim()}>
                Add
              </button>
            </div>
          </form>

          {app.loadingFiles && app.uploads.length === 0 && (
            <p className="status" role="status">
              Loading your files…
            </p>
          )}

          {app.uploads.length > 0 && (
            <>
              <h2 className="section-title">
                Your files <span>({app.uploads.length})</span>
              </h2>
              <ul className="files">
                {app.uploads.map((u) => (
                  <FileRow key={u.key} upload={u} onRemove={() => app.removeUpload(u.key)} />
                ))}
              </ul>
            </>
          )}
        </section>

        <form className="panel split__side" aria-label="Quiz options" onSubmit={create}>
          <h2>Quiz options</h2>

          <fieldset className="field">
            <legend>Model</legend>
            <div className="segmented">
              {(['Gemini', 'Ollama'] as const).map((p) => (
                <label key={p} className={app.creds.provider === p ? 'is-on' : ''}>
                  <input type="radio" name="provider" checked={app.creds.provider === p} onChange={() => app.setProvider(p)} />
                  {p}
                </label>
              ))}
            </div>
          </fieldset>

          {app.creds.provider === 'Gemini' && (
            <div className="field">
              <label htmlFor="key">Gemini API key</label>
              <input id="key" type="password" autoComplete="off" value={app.creds.apiKey} onChange={(e) => app.setApiKey(e.target.value)} />
              <p className="hint">Kept in this browser tab only. Never saved.</p>
            </div>
          )}

          <div className="field">
            <label htmlFor="count" className="field__split">
              Number of questions <strong>{count}</strong>
            </label>
            <input id="count" type="range" min={5} max={25} value={count} onChange={(e) => setCount(Number(e.target.value))} />
          </div>

          <fieldset className="field">
            <legend>Question type</legend>
            <div className="pills">
              {TYPES.map((t) => (
                <label key={t.value} className={type === t.value ? 'is-on' : ''}>
                  <input type="radio" name="type" checked={type === t.value} onChange={() => setType(t.value)} />
                  {t.label}
                </label>
              ))}
            </div>
          </fieldset>

          <fieldset className="field">
            <legend>Difficulty</legend>
            <div className="segmented">
              {LEVELS.map((l) => (
                <label key={l.value} className={difficulty === l.value ? 'is-on' : ''}>
                  <input type="radio" name="difficulty" checked={difficulty === l.value} onChange={() => setDifficulty(l.value)} />
                  {l.label}
                </label>
              ))}
            </div>
          </fieldset>

          <div className="field">
            <label htmlFor="topic">
              Topic filter <span className="muted">(optional)</span>
            </label>
            <input id="topic" type="text" placeholder="e.g. mitosis" value={topic} onChange={(e) => setTopic(e.target.value)} />
          </div>

          {error && <ErrorNote message={error} />}

          <button type="submit" className="button button--primary" disabled={!canCreate}>
            {creating ? 'Creating quiz…' : 'Create quiz'}
          </button>
          {!canCreate && !creating && (
            <p className="hint" role="status">
              {app.readyDocs.length === 0
                ? 'Add a file and wait for it to finish reading.'
                : reading
                  ? 'Waiting for files to finish reading.'
                  : 'Enter your Gemini API key to continue.'}
            </p>
          )}
          <p className="hint hint--note">Scanned pages and images are read with Gemini. Each page is sent to Google and counts against your quota.</p>
        </form>
      </div>
    </div>
  )
}
