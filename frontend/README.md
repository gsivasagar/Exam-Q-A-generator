# QuizForge front end

React, TypeScript and Vite. It is the new interface for QuizForge: upload files (including scans and
images), make a quiz, take it, see graded feedback, and chat with the documents with page citations.

```bash
npm ci
npm run dev          # the real app: needs the back end running (below)
npm run dev:sample   # built-in sample data, no back end; a banner says so on every page
```

Start the back end first, from the project root (Python 3.11 or newer):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --only-binary=:all: -r requirements.txt
python -m uvicorn app.api.main:app --port 8000
```

Two details that matter:

- **`--only-binary=:all:`** installs prebuilt packages only. Without it, pip on an Intel Mac picks the newest
  `cryptography`, which has no wheel for that chip and has to be compiled with Rust; if that build fails, pip
  aborts the whole install and the venv ends up empty.
- **`python -m uvicorn`**, not `uvicorn`: it runs the copy inside the active venv. A bare `uvicorn` can resolve to a
  globally installed one (check with `which uvicorn`; it should point inside `.venv`), which then fails with
  "Form data requires python-multipart" or similar because the venv's packages are not in use.

`/api` is proxied to `http://localhost:8000`, so the page, the API and the refresh cookie share one origin.
`APP_ORIGIN` (default `http://localhost:5173`) must match the address you open. See `API.md` for the contract.

**Sample data mode** answers with a fixed set of example questions (cells, mitosis) whatever you upload, with
made-up page numbers, and keeps everything in the tab. It exists for trying the screens and for tests; it does not
read your files.

## Commands

| | |
|---|---|
| `npm run dev` | dev server |
| `npm test` | unit and screen-flow tests (Vitest, Testing Library) |
| `npm run typecheck` | TypeScript |
| `npm run lint` | ESLint |
| `npm run build` | production build into `dist/` |

## Layout

```
src/api/        types (the contract), mock (sample data and rules), http (the real client), session (the token)
src/state/      auth (who is signed in), app state (files, model settings), a small loading hook
src/components/ layout with sidebar, small shared pieces
src/pages/      NewQuiz, Quiz, Results, Chat, Quizzes, SignIn, SignUp, Account
```

## Notes

- The Gemini key is held in memory only and sent in a header; closing the tab forgets it, and so does signing out.
- The access token is also held in memory only. A reload brings it back through the refresh cookie, which the page
  cannot read. When a call is refused as expired, one shared refresh runs and the call is retried once.
- Not connected to a real server yet: the sign-in code has run only against the sample data and a fake `fetch`.
- Scanned pages and images are read with Gemini: each page goes to Google. The upload screen says so.
- Not built yet: exports (PDF, Anki, CSV), the spaced-repetition review queue and the progress charts. They
  were in the Streamlit app and are not in this first version.
- The sample data is cleared on reload, so a quiz link does not survive a refresh until you use the real back end.
