import { Link } from 'react-router-dom'
import { ErrorNote, percent } from '../components/ui'
import { useApp } from '../state/AppState'
import { useLoad } from '../state/useLoad'

export function QuizzesPage() {
  const app = useApp()
  const loaded = useLoad(() => app.api.listQuizzes(), 'list')
  return (
    <div className="page">
      <header className="page__head">
        <h1>My quizzes</h1>
        <p>Quizzes you made in this session. Open one to retake it or see its results.</p>
      </header>
      {loaded.status === 'loading' && <p className="status" role="status">Loading quizzes…</p>}
      {loaded.status === 'error' && <ErrorNote message={loaded.message} />}
      {loaded.status === 'ready' && loaded.data.length === 0 && (
        <div className="card empty">
          <h2>No quizzes yet</h2>
          <p>Add a file and create your first quiz.</p>
          <Link to="/" className="button button--primary">
            New quiz
          </Link>
        </div>
      )}
      {loaded.status === 'ready' && loaded.data.length > 0 && (
        <ul className="quiz-list">
          {loaded.data.map((q) => (
            <li key={q.id} className="card quiz-list__item">
              <div>
                <div className="quiz-list__title">{q.title}</div>
                <div className="muted-line">
                  {q.questionCount} questions
                  {q.lastScore !== null && <> · last score {percent(q.lastScore)}</>}
                </div>
              </div>
              <div className="row">
                <Link to={`/quiz/${q.id}`} className="button button--secondary button--small">
                  {q.lastScore === null ? 'Start' : 'Retake'}
                </Link>
                {q.lastScore !== null && (
                  <Link to={`/results/${q.id}`} className="button button--secondary button--small">
                    Results
                  </Link>
                )}
              </div>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}
