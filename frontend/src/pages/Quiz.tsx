import { useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import { ApiError, type Quiz } from '../api'
import { Chip, ErrorNote, ProgressBar, SourceChip } from '../components/ui'
import { useApp } from '../state/AppState'
import { useLoad } from '../state/useLoad'

const TYPE_LABEL = { mcq: 'Multiple choice', tf: 'True / False', short: 'Short answer', cloze: 'Fill in the blank' } as const

function Runner({ quiz }: { quiz: Quiz }) {
  const app = useApp()
  const navigate = useNavigate()
  const [index, setIndex] = useState(0)
  const [answers, setAnswers] = useState<Record<string, string>>({})
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState('')

  const q = quiz.questions[index]
  const answered = quiz.questions.filter((x) => (answers[x.id] ?? '').trim() !== '').length
  const set = (value: string) => setAnswers((a) => ({ ...a, [q.id]: value }))

  const submit = async () => {
    setSubmitting(true)
    setError('')
    try {
      await app.api.gradeQuiz(quiz.id, answers, app.creds)
      navigate(`/results/${quiz.id}`)
    } catch (err) {
      setError(err instanceof ApiError ? err.message : 'Could not grade the quiz.')
      setSubmitting(false)
    }
  }

  return (
    <div className="page">
      <header className="page__head page__head--row">
        <div>
          <div className="eyebrow">{quiz.title}</div>
          <h1>
            Question {index + 1} of {quiz.questions.length}
          </h1>
        </div>
        <div className="head-progress">
          <div className="head-progress__labels">
            <span>{answered} answered</span>
            <span>{quiz.questions.length - answered} to go</span>
          </div>
          <ProgressBar value={answered} max={quiz.questions.length} label="Questions answered" />
        </div>
      </header>

      <div className="split">
        <section className="card split__main" aria-label="Question">
          <div className="chips">
            <Chip tone="accent">{TYPE_LABEL[q.type]}</Chip>
            <Chip>Topic: {q.topic}</Chip>
            <SourceChip source={q.source} />
          </div>

          <h2 className="question">{q.question}</h2>

          {q.options ? (
            <div role="radiogroup" aria-label="Answer choices" className="choices">
              {q.options.map((option, i) => {
                const selected = answers[q.id] === option
                return (
                  <button
                    key={option}
                    type="button"
                    role="radio"
                    aria-checked={selected}
                    className={`choice${selected ? ' choice--on' : ''}`}
                    onClick={() => set(option)}
                  >
                    <span className="choice__key">{String.fromCharCode(65 + i)}</span>
                    {option}
                  </button>
                )
              })}
            </div>
          ) : (
            <div className="field">
              <label htmlFor="answer">Your answer</label>
              <textarea id="answer" rows={5} value={answers[q.id] ?? ''} onChange={(e) => set(e.target.value)} />
            </div>
          )}

          <div className="card__foot">
            <button type="button" className="button button--secondary" disabled={index === 0} onClick={() => setIndex(index - 1)}>
              Previous
            </button>
            <button
              type="button"
              className="button button--primary"
              disabled={index === quiz.questions.length - 1}
              onClick={() => setIndex(index + 1)}
            >
              Next
            </button>
          </div>
        </section>

        <aside className="split__side stack" aria-label="Question navigator">
          <div className="panel">
            <h3>Questions</h3>
            <div className="navigator">
              {quiz.questions.map((x, i) => {
                const done = (answers[x.id] ?? '').trim() !== ''
                const state = i === index ? 'current' : done ? 'answered' : 'open'
                return (
                  <button
                    key={x.id}
                    type="button"
                    className={`navigator__item navigator__item--${state}`}
                    aria-label={`Question ${i + 1}, ${i === index ? 'current' : done ? 'answered' : 'not answered'}`}
                    aria-current={i === index ? 'step' : undefined}
                    onClick={() => setIndex(i)}
                  >
                    {i + 1}
                  </button>
                )
              })}
            </div>
          </div>
          {error && <ErrorNote message={error} />}
          <button type="button" className="button button--dark" onClick={submit} disabled={submitting}>
            {submitting ? 'Grading…' : 'Submit answers and grade me'}
          </button>
          <p className="hint">You can submit with questions left open. They score zero.</p>
        </aside>
      </div>
    </div>
  )
}

export function QuizPage() {
  const { id = '' } = useParams()
  const app = useApp()
  const loaded = useLoad(() => app.api.getQuiz(id), id)
  if (loaded.status === 'loading') return <p className="page status" role="status">Loading quiz…</p>
  if (loaded.status === 'error') {
    return (
      <div className="page">
        <ErrorNote message={loaded.message} />
        <p>
          <Link to="/">Start a new quiz</Link>
        </p>
      </div>
    )
  }
  return <Runner quiz={loaded.data} />
}
