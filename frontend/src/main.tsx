import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { BrowserRouter } from 'react-router-dom'
import { App } from './App'
import { createApi, SessionStore } from './api'
import { AuthProvider } from './state/Auth'
import './styles.css'

const store = new SessionStore()
const api = createApi(store)

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <BrowserRouter>
      <AuthProvider api={api} store={store}>
        <App />
      </AuthProvider>
    </BrowserRouter>
  </StrictMode>,
)
