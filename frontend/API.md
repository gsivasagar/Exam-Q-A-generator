# API contract

What the front end expects from the Python back end. The types are in `src/api/types.ts`; the sample
implementation of every rule is `src/api/mock.ts`; the real client is `src/api/http.ts`. JSON uses
camelCase field names.

The back end is `app/api/` (run with `uvicorn app.api.main:app`). It is a thin layer over functions QuizForge
already had (`ingest_document`, `ingest_url`, `generate_qa_pairs`, `grade_all`), plus accounts. Not built yet:
`POST /api/chat` answers `501 not_implemented` (it checks file ownership first). The front end uses every
endpoint below except chat (`src/api/http.ts`, `src/state/Auth.tsx`).

## Signing in

Every endpoint except the ones marked public needs `Authorization: Bearer <access token>`; without a valid one
the answer is `401 unauthorized`.

- The **access token** is a JWT that lives 15 minutes. Keep it in memory only (not localStorage).
- The **refresh token** is an httpOnly cookie (`qf_refresh`, path `/api/auth`) the page cannot read. When an
  access token expires or a call answers 401, call `POST /api/auth/refresh` once, then retry. Two calls that
  fail together should share one refresh. If refresh answers 401, the person must sign in again.
- `POST /api/auth/refresh` and `/api/auth/logout` act on the cookie, so they also need the header
  `X-Requested-With: quizforge` and an `Origin` equal to `APP_ORIGIN`; otherwise `403 csrf`.
- Send credentials with `fetch(..., { credentials: 'include' })` on the auth calls.

| Method and path | Body | Returns |
|---|---|---|
| `POST /api/auth/signup` (public) | `{ email, password, remember }` | `{ accessToken, expiresIn, user }` and the cookie |
| `POST /api/auth/login` (public) | `{ email, password, remember }` | same |
| `POST /api/auth/guest` (public) | | same, for a guest user |
| `POST /api/auth/upgrade` | `{ email, password }` (guest only) | same user id, now with an email |
| `POST /api/auth/refresh` (public, cookie) | | `{ accessToken, expiresIn, user }` |
| `POST /api/auth/logout` (public, cookie) | | `204`, cookie cleared |
| `GET /api/me` | | `{ id, email, isGuest, expiresAt, counts: { files, quizzes, reviewCards } }` |
| `POST /api/me/password` | `{ current, new }` | `204`; other devices are signed out |
| `GET /api/me/sessions` | | `[{ id, userAgent, createdAt, lastUsedAt, current }]` |
| `DELETE /api/me/sessions/{id}`, `DELETE /api/me/sessions/others` | | `204`, `{ signedOut }` |
| `GET /api/me/export` | | a JSON download of everything stored for the account |
| `DELETE /api/me` | `{ email }` | `204`; removes the account and all its data |
| `GET /api/documents`, `DELETE /api/documents/{id}` | | the account's files, `204` |

`user` is `{ id, email, isGuest, expiresAt }`; `email` is null for a guest, who also has an `expiresAt`.
Passwords: 12 to 128 characters, not on a short list of very common ones. A wrong email and a wrong password
give the same `invalid_credentials` message. Limits: 5 login attempts a minute per address and email, 20 an
hour per address; 5 sign-ups and 5 guest sessions an hour per address (`429 rate_limited` with `Retry-After`).
Guests: 3 files of 10 MB at most (`403 guest_limit`), deleted after 7 days.
Another account's file, quiz or session is always `404 not_found`, the same as one that does not exist.

## Model credentials

Calls that may reach a model carry them in headers, never in a URL or body (this is separate from signing in):

| Header | Value |
|---|---|
| `X-Provider` | `Gemini` or `Ollama` |
| `X-Api-Key` | the user's Gemini key (omitted for Ollama) |

The server must not log, store or echo the key. The front end keeps it in memory only.

## Errors

Any non-2xx response is JSON `{ "code": "<code>", "message": "<text for the user>" }`. Codes the front end
understands: `ocr_limit`, `needs_gemini`, `unreadable`, `unsupported`, `no_documents`, `not_found`,
`rate_limited`, `unauthorized`, `invalid_credentials`, `email_taken`, `invalid_email`, `weak_password`,
`wrong_password`, `guest_limit`, `csrf`. Anything else is shown with its message as "unknown". Messages are shown as written, so they
must be safe and plain.

## Endpoints

| Method and path | Body | Returns |
|---|---|---|
| `POST /api/documents` | multipart `file` | stream (below) |
| `POST /api/documents/url` | `{ "url": "https://..." }` | stream (below) |
| `POST /api/quizzes` | `{ documentIds, count, type, difficulty, topic }` | `Quiz` |
| `GET /api/quizzes` | | `QuizSummary[]` |
| `GET /api/quizzes/{id}` | | `Quiz` |
| `POST /api/quizzes/{id}/grade` | `{ answers: { [questionId]: string } }` | `GradeResult` |
| `GET /api/quizzes/{id}/result` | | `GradeResult` (latest attempt) |
| `POST /api/chat` | `{ documentIds, message }` | `ChatAnswer` |

`type` is `short`, `mcq`, `mixed`, `tf` or `cloze`; `difficulty` is `any`, `easy`, `medium` or `hard`.
Question pages are 1-based; `0` means the file has no pages.

### Document stream

`POST /api/documents` answers with newline-delimited JSON, one object per line, so a scan can report
progress while it is read:

```
{"event":"progress","page":4,"pages":12,"tables":2,"figures":1}
{"event":"done","document":{"id":"...","name":"Lecture 3 scan.pdf","pages":12,"method":"ocr","tables":4,"figures":2}}
```

or, on failure, `{"event":"error","code":"ocr_limit","message":"..."}`. Text files may send only the `done`
line. `method` is `text`, `ocr` or `image`.

## Rules the back end must follow (the front end relies on them)

- A scan over `OCR_MAX_PAGES` (30) is refused with `ocr_limit` before any page is read.
- An image, or a scan with no text layer, without Gemini and a key is refused (`needs_gemini` or `unreadable`).
- Every question carries its `source` (file and page). Every chat citation does too.
- Grading: the model says which key points a short answer covers and the score is computed in code, so a
  re-grade gives the same number. `GradeResult.topics` is sorted weakest first. An answer the model could not grade has status `ungraded`
  and score 0, with the reason in `explanation`.
- Chat answers only from the selected documents. When they do not answer the question it returns
  `{ "kind": "refusal", "text": "..." }`, never a guess. Citation markers in `text` are `[1]`, `[2]`, and each
  matches a `citations[].n`.
- A table read from a scan or image is returned as `citation.table` (header and rows), with
  `readWith: "gemini"`, so the page can draw it as a table.
