// The contract between the front end and the Python back end. docs: frontend/API.md

export type Provider = 'Gemini' | 'Ollama'
export type QuestionType = 'short' | 'mcq' | 'mixed' | 'tf' | 'cloze'
export type Difficulty = 'any' | 'easy' | 'medium' | 'hard'

/** What every call that may use a model needs to know. The key is sent in a header, never stored. */
export interface Credentials {
  provider: Provider
  apiKey: string
}

export interface SourceRef {
  file: string
  /** 1-based page, or 0 when the file has no pages (Word, text, web page, image). */
  page: number
}

export type ReadMethod = 'text' | 'ocr' | 'image'

export interface IngestedDocument {
  id: string
  name: string
  pages: number | null
  method: ReadMethod
  tables: number
  figures: number
}

export interface IngestProgress {
  page: number
  pages: number
  tables: number
  figures: number
}

export interface Question {
  id: string
  type: Exclude<QuestionType, 'mixed'>
  question: string
  /** Present for mcq and tf. */
  options?: string[]
  topic: string
  source: SourceRef
}

export interface Quiz {
  id: string
  title: string
  createdAt: string
  questions: Question[]
}

export interface QuizSummary {
  id: string
  title: string
  createdAt: string
  questionCount: number
  /** Average score of the latest attempt, 0 to 1, or null if never graded. */
  lastScore: number | null
}

export interface CreateQuizRequest extends Credentials {
  documentIds: string[]
  count: number
  type: QuestionType
  difficulty: Difficulty
  topic: string
}

export interface KeyPoint {
  text: string
  covered: boolean
}

export interface QuestionResult {
  questionId: string
  question: string
  source: SourceRef
  status: 'correct' | 'partial' | 'incorrect' | 'ungraded'
  /** 0 to 1. */
  score: number
  yourAnswer: string
  correctAnswer?: string
  keyPoints?: KeyPoint[]
  explanation?: string
}

export interface TopicScore {
  topic: string
  /** 0 to 1. */
  score: number
}

export interface GradeResult {
  quizId: string
  title: string
  /** 0 to 1. */
  average: number
  correct: number
  total: number
  results: QuestionResult[]
  topics: TopicScore[]
}

export interface TableData {
  caption: string
  header: string[]
  rows: string[][]
  highlightRow?: number
}

export interface Citation {
  /** The number shown in the answer text as [n]. */
  n: number
  source: SourceRef
  passage: string
  /** Set when the passage is a table that was read from a scan or image. */
  table?: TableData
  readWith?: 'gemini'
}

export type ChatAnswer =
  | { kind: 'answer'; text: string; citations: Citation[] }
  | { kind: 'refusal'; text: string }

export type ApiErrorCode =
  | 'ocr_limit'
  | 'needs_gemini'
  | 'invalid_key'
  | 'model_error'
  | 'unreadable'
  | 'unsupported'
  | 'no_documents'
  | 'not_found'
  | 'rate_limited'
  | 'not_guest'
  | 'no_password'
  | 'confirmation_mismatch'
  | 'unauthorized'
  | 'invalid_credentials'
  | 'email_taken'
  | 'invalid_email'
  | 'weak_password'
  | 'wrong_password'
  | 'guest_limit'
  | 'csrf'
  | 'network'
  | 'unknown'

export class ApiError extends Error {
  readonly code: ApiErrorCode
  constructor(code: ApiErrorCode, message: string) {
    super(message)
    this.name = 'ApiError'
    this.code = code
  }
}

export interface User {
  id: string
  /** Null for a guest. */
  email: string | null
  isGuest: boolean
  /** When a guest's data is deleted; null for a full account. */
  expiresAt: string | null
}

/** What sign-in, sign-up, guest, upgrade and refresh all return. The refresh token travels in a cookie. */
export interface AuthResult {
  accessToken: string
  /** Seconds. */
  expiresIn: number
  user: User
}

export interface Me extends User {
  counts: { files: number; quizzes: number; reviewCards: number }
}

export interface Device {
  id: string
  userAgent: string
  createdAt: string
  lastUsedAt: string
  current: boolean
}

export interface Api {
  /** True for the built-in sample data: questions do not come from the uploaded files. */
  readonly sample?: boolean
  // Accounts
  signup(email: string, password: string, remember: boolean): Promise<AuthResult>
  login(email: string, password: string, remember: boolean): Promise<AuthResult>
  guest(): Promise<AuthResult>
  /** A guest gives an email and password; the account, its files and quizzes stay. */
  upgrade(email: string, password: string): Promise<AuthResult>
  /** Uses the refresh cookie. Rejects with `unauthorized` when there is no valid session. */
  refresh(): Promise<AuthResult>
  logout(): Promise<void>
  me(): Promise<Me>
  changePassword(current: string, next: string): Promise<void>
  devices(): Promise<Device[]>
  signOutDevice(id: string): Promise<void>
  signOutOtherDevices(): Promise<number>
  exportData(): Promise<Blob>
  deleteAccount(email: string): Promise<void>
  // Files
  listDocuments(): Promise<IngestedDocument[]>
  deleteDocument(id: string): Promise<void>
  // Work
  ingestFile(
    file: File,
    creds: Credentials,
    onProgress: (p: IngestProgress) => void,
  ): Promise<IngestedDocument>
  ingestUrl(url: string, creds: Credentials): Promise<IngestedDocument>
  createQuiz(req: CreateQuizRequest): Promise<Quiz>
  listQuizzes(): Promise<QuizSummary[]>
  getQuiz(id: string): Promise<Quiz>
  gradeQuiz(id: string, answers: Record<string, string>, creds: Credentials): Promise<GradeResult>
  getResult(quizId: string): Promise<GradeResult>
  chat(documentIds: string[], message: string, creds: Credentials): Promise<ChatAnswer>
}
