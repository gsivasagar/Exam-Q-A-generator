import { Route, Routes } from 'react-router-dom'
import { AppStateProvider } from './state/AppState'
import { useAuth } from './state/Auth'
import { RequireSession } from './components/RequireSession'
import { AccountPage } from './pages/Account'
import { ChatPage } from './pages/Chat'
import { NewQuiz } from './pages/NewQuiz'
import { QuizPage } from './pages/Quiz'
import { QuizzesPage } from './pages/Quizzes'
import { ResultsPage } from './pages/Results'
import { SignInPage } from './pages/SignIn'
import { SignUpPage } from './pages/SignUp'

export function App() {
  const { user, api } = useAuth()
  const routes = (
    <Routes>
      <Route path="signin" element={<SignInPage />} />
      <Route path="signup" element={<SignUpPage />} />
      <Route element={<RequireSession />}>
        <Route index element={<NewQuiz />} />
        <Route path="quizzes" element={<QuizzesPage />} />
        <Route path="quiz/:id" element={<QuizPage />} />
        <Route path="results/:id" element={<ResultsPage />} />
        <Route path="chat" element={<ChatPage />} />
        <Route path="account" element={<AccountPage />} />
        <Route
          path="*"
          element={
            <div className="page">
              <h1>Page not found</h1>
            </div>
          }
        />
      </Route>
    </Routes>
  )
  // Keyed by account, so switching accounts never shows the previous person's files or Gemini key. It sits above
  // the routes so a guest who goes to create an account keeps what they were doing.
  return user ? (
    <AppStateProvider key={user.id} api={api}>
      {routes}
    </AppStateProvider>
  ) : (
    routes
  )
}
