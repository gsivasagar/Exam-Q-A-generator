import { Link, useParams } from 'react-router-dom'
import type { GradeResult, QuestionResult } from '../api'
import { Chip, ErrorNote, SourceChip, percent } from '../components/ui'
import { useApp } from '../state/AppState'
import { useLoad } from '../state/useLoad'

function scoreTone(score: number): 'success' | 'warn' | 'danger' {
  return score >= 0.8 ? 'success' : score >= 0.5 ? 'warn' : 'danger'
}

function ResultCard({ r, index }: { r: QuestionResult; index: number }) {
  return (
    <article className="card result">
      <div className="result__head">
        {r.status === 'correct' && <Chip tone="success">Correct</Chip>}
        {r.status === 'partial' && <Chip tone="warn">Partly right, {percent(r.score)}</Chip>}
        {r.status === 'incorrect' && <Chip tone="danger">Incorrect</Chip>}
        {r.status === 'ungraded' && <Chip tone="warn">Could not be graded</Chip>}
        <SourceChip source={r.source} />
      </div>
      <div className="result__question">
        {index + 1}. {r.question}
      </div>

      {r.keyPoints ? (
        <>
          <div className="quoted">Your answer: {r.yourAnswer || <em>No answer</em>}</div>
          <div className="result__sub">Key points</div>
          <ul className="keypoints">
            {r.keyPoints.map((k) => (
              <li key={k.text} className={k.covered ? 'is-covered' : 'is-missed'}>
                <span className="sr-only">{k.covered ? 'Covered: ' : 'Missed: '}</span>
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                  {k.covered ? <path d="M5 12l5 5 9-10" /> : <path d="M6 6l12 12M18 6L6 18" />}
                </svg>
                {k.text}
              </li>
            ))}
          </ul>
          <p className="hint">The model only says which points were covered. The score is worked out from that, so a re-grade gives the same number.</p>
        </>
      ) : r.status === 'correct' ? (
        <div className="muted-line">Your answer: {r.yourAnswer}</div>
      ) : (
        <>
          <div className="answers">
            <span className="answers__yours">Your answer: {r.yourAnswer || 'No answer'}</span>
            {r.correctAnswer && <span className="answers__right">Correct: {r.correctAnswer}</span>}
          </div>
          {r.explanation && <p className="explanation">{r.explanation}</p>}
        </>
      )}
    </article>
  )
}

function Summary({ result }: { result: GradeResult }) {
  const weakest = result.topics[0]
  const partly = result.results.filter((r) => r.status === 'partial').length
  const wrong = result.results.filter((r) => r.status === 'incorrect').length
  return (
    <div className="page">
      <header className="page__head page__head--row">
        <div>
          <div className="eyebrow">{result.title}</div>
          <h1>Your results</h1>
        </div>
        <div className="row">
          <Link to={`/quiz/${result.quizId}`} className="button button--secondary">
            Retake quiz
          </Link>
          <Link to="/" className="button button--primary">
            New quiz
          </Link>
        </div>
      </header>

      <section className="stats" aria-label="Summary">
        <div className="card stat">
          <div className="stat__label">Average score</div>
          <div className="stat__value">{percent(result.average)}</div>
          <div className="stat__note">Computed from the key points each answer covered</div>
        </div>
        <div className="card stat">
          <div className="stat__label">Fully correct</div>
          <div className="stat__value">
            {result.correct} <small>of {result.total}</small>
          </div>
          <div className="stat__note">
            {partly} partly right, {wrong} wrong
          </div>
        </div>
        <div className="card stat">
          <div className="stat__label">Weakest topic</div>
          <div className="stat__value stat__value--text">{weakest ? weakest.topic : 'None'}</div>
          <div className="stat__note">{weakest ? `${percent(weakest.score)} correct` : ''}</div>
        </div>
      </section>

      <div className="split">
        <section className="split__main stack" aria-label="Question feedback">
          <h2 className="section-title">Question by question</h2>
          {result.results.map((r, i) => (
            <ResultCard key={r.questionId} r={r} index={i} />
          ))}
        </section>

        <aside className="split__side stack" aria-label="Next steps">
          <div className="panel">
            <h3>Practise your weak topics</h3>
            <p className="hint">Make a new quiz from the parts of your notes that cover the topics you missed.</p>
            <ul className="topics">
              {result.topics.map((t) => (
                <li key={t.topic}>
                  <span>{t.topic}</span>
                  <strong className={`tone-${scoreTone(t.score)}`}>{percent(t.score)}</strong>
                </li>
              ))}
            </ul>
            {weakest && (
              <Link to={`/?topic=${encodeURIComponent(weakest.topic)}`} className="button button--primary">
                Practise {weakest.topic}
              </Link>
            )}
          </div>
        </aside>
      </div>
    </div>
  )
}

export function ResultsPage() {
  const { id = '' } = useParams()
  const app = useApp()
  const loaded = useLoad(() => app.api.getResult(id), id)
  if (loaded.status === 'loading') return <p className="page status" role="status">Loading results…</p>
  if (loaded.status === 'error') {
    return (
      <div className="page">
        <ErrorNote message={loaded.message} />
        <p>
          <Link to={`/quiz/${id}`}>Back to the quiz</Link>
        </p>
      </div>
    )
  }
  return <Summary result={loaded.data} />
}
