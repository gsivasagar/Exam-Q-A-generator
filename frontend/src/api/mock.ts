import {
  ApiError,
  type Api,
  type AuthResult,
  type ChatAnswer,
  type CreateQuizRequest,
  type Credentials,
  type GradeResult,
  type Device,
  type IngestedDocument,
  type Me,
  type Question,
  type Quiz,
  type QuizSummary,
  type QuestionResult,
  type User,
} from './types'

/** Same limit as the back end's OCR_MAX_PAGES. */
export const OCR_MAX_PAGES = 30

interface Seed {
  question: string
  topic: string
  /** mcq */
  options?: string[]
  /** The right option (mcq), or the model answer (short). */
  answer: string
  /** Short answers: the key points a good answer covers, with words that signal each one. */
  keyPoints?: { text: string; words: string[] }[]
  explanation?: string
}

// Sample content only. The real questions come from the user's own documents.
const BANK: Seed[] = [
  {
    question: "Which organelle produces most of the cell's ATP?",
    topic: 'Organelles',
    options: ['Nucleus', 'Mitochondrion', 'Golgi apparatus', 'Lysosome'],
    answer: 'Mitochondrion',
  },
  {
    question: 'Explain why cells need to control the cell cycle.',
    topic: 'Cell cycle control',
    answer: 'So division happens only when conditions are right and DNA has been copied accurately.',
    keyPoints: [
      { text: 'Division happens only when conditions are right', words: ['ready', 'right', 'when'] },
      { text: 'DNA must be copied accurately first', words: ['dna', 'copied', 'copy'] },
      { text: 'Uncontrolled division can lead to tumours', words: ['tumour', 'tumor', 'cancer'] },
    ],
  },
  {
    question: 'During which phase of mitosis do sister chromatids separate and move to opposite poles?',
    topic: 'Mitosis phases',
    options: ['Prophase', 'Metaphase', 'Anaphase', 'Telophase'],
    answer: 'Anaphase',
    explanation:
      'In metaphase the chromosomes line up at the middle. The sister chromatids are pulled apart in anaphase.',
  },
  {
    question: 'In which phase does the nuclear envelope re-form around each set of chromosomes?',
    topic: 'Mitosis phases',
    options: ['Prophase', 'Anaphase', 'Telophase', 'Interphase'],
    answer: 'Telophase',
  },
  {
    question: 'What happens to an enzyme at a temperature far above its optimum?',
    topic: 'Enzymes',
    answer: 'It loses its shape (denatures) and can no longer bind its substrate.',
    keyPoints: [
      { text: 'The enzyme loses its shape', words: ['shape', 'denature', 'unfold'] },
      { text: 'It can no longer bind its substrate', words: ['substrate', 'bind', 'active site'] },
    ],
  },
  {
    question: 'Which molecule carries amino acids to the ribosome?',
    topic: 'Protein synthesis',
    options: ['mRNA', 'tRNA', 'rRNA', 'DNA'],
    answer: 'tRNA',
  },
]

const TRUE_FALSE: Seed[] = [
  { question: 'Ribosomes are surrounded by a membrane.', topic: 'Organelles', answer: 'False' },
  { question: 'DNA is copied during the S phase of interphase.', topic: 'Cell cycle control', answer: 'True' },
]

function wait(ms: number): Promise<void> {
  return ms > 0 ? new Promise((resolve) => setTimeout(resolve, ms)) : Promise.resolve()
}

const uid = (() => {
  let n = 0
  return (prefix: string) => `${prefix}-${++n}`
})()

interface StoredQuiz {
  owner: string
  quiz: Quiz
  seeds: Map<string, Seed>
  result?: GradeResult
}

export interface MockOptions {
  /** Pause per step, to make loading states visible. Tests pass 0. */
  delayMs?: number
  /** Start already signed in as this email (with the refresh cookie set), as if returning to the site. */
  signedInAs?: string
}

/** Mirrors the server's password rules, in short form. */
const COMMON = new Set(['password1234', 'qwertyuiop12', '123456789012', 'letmein12345', 'iloveyou1234'])
export function mockPasswordProblem(password: string): string | null {
  if (password.length < 12) return 'Use at least 12 characters. A few words with spaces or dashes works well.'
  if (password.length > 128) return 'Use at most 128 characters.'
  if (COMMON.has(password.toLowerCase()) || /^(.)\1+$/.test(password)) return 'That password is too common. Choose something less guessable.'
  return null
}

interface MockUser extends User {
  password: string | null
}

const GUEST_DAYS = 7
const GUEST_MAX_FILES = 3

export function createMockApi({ delayMs = 250, signedInAs }: MockOptions = {}): Api {
  const docs = new Map<string, { owner: string; doc: IngestedDocument }>()
  const quizzes = new Map<string, StoredQuiz>()
  const users = new Map<string, MockUser>()
  const otherDevices = new Map<string, Device[]>()
  let current: MockUser | null = null // who the access token belongs to
  let cookie: string | null = null // who the refresh cookie belongs to

  const needsGemini = (c: Credentials) => c.provider !== 'Gemini' || !c.apiKey.trim()
  const publicUser = (u: MockUser): User => ({ id: u.id, email: u.email, isGuest: u.isGuest, expiresAt: u.expiresAt })
  const requireUser = (): MockUser => {
    if (!current) throw new ApiError('unauthorized', 'Sign in to continue.')
    return current
  }
  const addDoc = (owner: string, doc: IngestedDocument) => {
    docs.set(doc.id, { owner, doc })
    return doc
  }
  const ownedDocs = (owner: string) => [...docs.values()].filter((d) => d.owner === owner).map((d) => d.doc)
  const ownedDoc = (owner: string, id: string) => {
    const found = docs.get(id)
    return found && found.owner === owner ? found.doc : undefined
  }
  const issue = (u: MockUser): AuthResult => {
    current = u
    cookie = u.id
    return { accessToken: `mock-token-${u.id}`, expiresIn: 900, user: publicUser(u) }
  }
  const newUser = (email: string | null, password: string | null, isGuest: boolean): MockUser => {
    const u: MockUser = {
      id: uid('user'),
      email,
      password,
      isGuest,
      expiresAt: isGuest ? new Date(Date.now() + GUEST_DAYS * 86400000).toISOString() : null,
    }
    users.set(u.id, u)
    const now = new Date().toISOString()
    otherDevices.set(u.id, [{ id: uid('device'), userAgent: 'Safari on iPhone', createdAt: now, lastUsedAt: now, current: false }])
    return u
  }
  const checkEmail = (email: string): string => {
    const e = email.trim().toLowerCase()
    if (!/^[^@\s]+@[^@\s]+\.[^@\s]+$/.test(e)) throw new ApiError('invalid_email', 'That does not look like an email address.')
    return e
  }
  const byEmail = (email: string) => [...users.values()].find((u) => u.email === email)

  if (signedInAs) issue(newUser(signedInAs.toLowerCase(), 'a long test passphrase', false))

  return {
    sample: true,
    async signup(email, password) {
      await wait(delayMs)
      const e = checkEmail(email)
      const problem = mockPasswordProblem(password)
      if (problem) throw new ApiError('weak_password', problem)
      if (byEmail(e)) throw new ApiError('email_taken', 'An account with this email already exists. Try signing in.')
      return issue(newUser(e, password, false))
    },
    async login(email, password) {
      await wait(delayMs)
      const u = byEmail(email.trim().toLowerCase())
      if (!u || u.password !== password) throw new ApiError('invalid_credentials', 'Email or password is wrong. Check both and try again.')
      return issue(u)
    },
    async guest() {
      await wait(delayMs)
      return issue(newUser(null, null, true))
    },
    async upgrade(email, password) {
      await wait(delayMs)
      const me = requireUser()
      if (!me.isGuest) throw new ApiError('not_guest', 'This account already has an email and password.')
      const e = checkEmail(email)
      const problem = mockPasswordProblem(password)
      if (problem) throw new ApiError('weak_password', problem)
      if (byEmail(e)) throw new ApiError('email_taken', 'An account with this email already exists. Try signing in.')
      Object.assign(me, { email: e, password, isGuest: false, expiresAt: null })
      return issue(me)
    },
    async refresh() {
      await wait(delayMs)
      const u = cookie ? users.get(cookie) : undefined
      if (!u) throw new ApiError('unauthorized', 'Sign in to continue.')
      return issue(u)
    },
    async logout() {
      await wait(delayMs)
      current = null
      cookie = null
    },
    async me(): Promise<Me> {
      await wait(delayMs)
      const u = requireUser()
      return {
        ...publicUser(u),
        counts: {
          files: ownedDocs(u.id).length,
          quizzes: [...quizzes.values()].filter((q) => q.owner === u.id).length,
          reviewCards: 0,
        },
      }
    },
    async changePassword(currentPassword, next) {
      await wait(delayMs)
      const u = requireUser()
      if (!u.password) throw new ApiError('no_password', 'Guest sessions have no password. Create an account first.')
      if (u.password !== currentPassword) throw new ApiError('wrong_password', 'The current password is wrong.')
      const problem = mockPasswordProblem(next)
      if (problem) throw new ApiError('weak_password', problem)
      u.password = next
      otherDevices.set(u.id, [])
    },
    async devices() {
      await wait(delayMs)
      const u = requireUser()
      const now = new Date().toISOString()
      return [{ id: `${u.id}-here`, userAgent: 'Chrome on macOS', createdAt: now, lastUsedAt: now, current: true }, ...(otherDevices.get(u.id) ?? [])]
    },
    async signOutDevice(id) {
      await wait(delayMs)
      const u = requireUser()
      const list = otherDevices.get(u.id) ?? []
      if (!list.some((d) => d.id === id)) throw new ApiError('not_found', 'That session was not found.')
      otherDevices.set(u.id, list.filter((d) => d.id !== id))
    },
    async signOutOtherDevices() {
      await wait(delayMs)
      const u = requireUser()
      const n = (otherDevices.get(u.id) ?? []).length
      otherDevices.set(u.id, [])
      return n
    },
    async exportData() {
      await wait(delayMs)
      const u = requireUser()
      const body = { account: publicUser(u), files: ownedDocs(u.id), quizzes: [...quizzes.values()].filter((q) => q.owner === u.id).map((q) => q.quiz) }
      return new Blob([JSON.stringify(body, null, 2)], { type: 'application/json' })
    },
    async deleteAccount(email) {
      await wait(delayMs)
      const u = requireUser()
      if (u.email && email.trim().toLowerCase() !== u.email) throw new ApiError('confirmation_mismatch', 'That is not the email on this account.')
      for (const [id, d] of docs) if (d.owner === u.id) docs.delete(id)
      for (const [id, q] of quizzes) if (q.owner === u.id) quizzes.delete(id)
      users.delete(u.id)
      current = null
      cookie = null
    },

    async listDocuments() {
      await wait(delayMs)
      return ownedDocs(requireUser().id)
    },
    async deleteDocument(id) {
      await wait(delayMs)
      const u = requireUser()
      if (!ownedDoc(u.id, id)) throw new ApiError('not_found', 'That file was not found.')
      docs.delete(id)
    },

    async ingestFile(file, creds, onProgress) {
      await wait(delayMs)
      const me = requireUser()
      if (me.isGuest && ownedDocs(me.id).length >= GUEST_MAX_FILES) {
        throw new ApiError('guest_limit', `Guests can add up to ${GUEST_MAX_FILES} files. Create an account to add more.`)
      }
      const name = file.name
      const lower = name.toLowerCase()
      const isImage = /\.(png|jpe?g|webp)$/.test(lower)
      const isPdf = lower.endsWith('.pdf')
      if (!isImage && !isPdf && !/\.(docx|pptx|md|markdown|txt)$/.test(lower)) {
        throw new ApiError('unsupported', `Unsupported file type '${name}'. Upload a PDF, Word, PowerPoint, image, Markdown or text file.`)
      }
      if (isImage) {
        if (needsGemini(creds)) {
          throw new ApiError('needs_gemini', 'Reading an image needs the Gemini provider. Choose Gemini and enter an API key, or upload a PDF, Word or text file.')
        }
        await wait(delayMs)
        const doc: IngestedDocument = { id: uid('doc'), name, pages: null, method: 'image', tables: 0, figures: 1 }
        return addDoc(me.id, doc)
      }
      const scanned = isPdf && lower.includes('scan')
      if (!scanned) {
        const doc: IngestedDocument = { id: uid('doc'), name, pages: isPdf ? 48 : null, method: 'text', tables: 0, figures: 0 }
        return addDoc(me.id, doc)
      }
      const pages = lower.includes('big') ? 42 : 12
      if (pages > OCR_MAX_PAGES) {
        throw new ApiError('ocr_limit', `This scan has ${pages} pages and the limit is ${OCR_MAX_PAGES}. Split the file, or upload a PDF that has a text layer.`)
      }
      if (needsGemini(creds)) {
        throw new ApiError('unreadable', 'This PDF has no text layer. It may be a scanned document. Choose Gemini and enter an API key to read it.')
      }
      let tables = 0
      let figures = 0
      for (let page = 1; page <= pages; page++) {
        await wait(delayMs / 4)
        if (page % 3 === 0) tables++
        if (page % 5 === 0) figures++
        onProgress({ page, pages, tables, figures })
      }
      const doc: IngestedDocument = { id: uid('doc'), name, pages, method: 'ocr', tables, figures }
      return addDoc(me.id, doc)
    },

    async ingestUrl(url) {
      await wait(delayMs)
      const me = requireUser()
      let host: string
      try {
        const u = new URL(url)
        if (u.protocol !== 'http:' && u.protocol !== 'https:') throw new Error('scheme')
        host = u.hostname + (u.pathname.replace(/\/$/, '') || '')
      } catch {
        throw new ApiError('unsupported', 'That does not look like a web address. Start with https://')
      }
      const doc: IngestedDocument = { id: uid('doc'), name: host, pages: null, method: 'text', tables: 0, figures: 0 }
      return addDoc(me.id, doc)
    },

    async createQuiz(req: CreateQuizRequest) {
      await wait(delayMs)
      const me = requireUser()
      const chosen = req.documentIds.map((id) => ownedDoc(me.id, id)).filter((d): d is IngestedDocument => !!d)
      if (chosen.length === 0) throw new ApiError('no_documents', 'Add at least one file first.')
      const topic = req.topic.trim().toLowerCase()
      const pool = (type: string): Seed[] => {
        const base = type === 'tf' ? TRUE_FALSE : BANK.filter((s) => (type === 'mcq' ? !!s.options : type === 'short' || type === 'cloze' ? !s.options : true))
        const filtered = topic ? base.filter((s) => s.topic.toLowerCase().includes(topic) || s.question.toLowerCase().includes(topic)) : base
        return filtered.length > 0 ? filtered : base
      }
      const seeds = new Map<string, Seed>()
      const questions: Question[] = []
      for (let i = 0; i < req.count; i++) {
        const kind = req.type === 'mixed' ? (i % 2 === 0 ? 'mcq' : 'short') : req.type
        const list = pool(kind)
        const seed = list[i % list.length]
        const doc = chosen[i % chosen.length]
        const id = uid('q')
        const isTf = kind === 'tf'
        questions.push({
          id,
          type: isTf ? 'tf' : seed.options ? 'mcq' : kind === 'cloze' ? 'cloze' : 'short',
          question: seed.question,
          options: isTf ? ['True', 'False'] : seed.options,
          topic: seed.topic,
          source: { file: doc.name, page: doc.pages ? ((i * 3 + 4) % doc.pages) + 1 : 0 },
        })
        seeds.set(id, seed)
      }
      const quiz: Quiz = {
        id: uid('quiz'),
        title: `Practice: ${chosen.map((d) => d.name).join(', ')}`.slice(0, 80),
        createdAt: new Date().toISOString(),
        questions,
      }
      quizzes.set(quiz.id, { owner: me.id, quiz, seeds })
      return quiz
    },

    async listQuizzes(): Promise<QuizSummary[]> {
      await wait(delayMs)
      const me = requireUser()
      return [...quizzes.values()]
        .filter((q) => q.owner === me.id)
        .map(({ quiz, result }) => ({
          id: quiz.id,
          title: quiz.title,
          createdAt: quiz.createdAt,
          questionCount: quiz.questions.length,
          lastScore: result ? result.average : null,
        }))
        .reverse()
    },

    async getQuiz(id) {
      await wait(delayMs)
      const me = requireUser()
      const found = quizzes.get(id)
      if (!found || found.owner !== me.id) throw new ApiError('not_found', 'That quiz was not found. The sample data is cleared when the page reloads.')
      return found.quiz
    },

    async gradeQuiz(id, answers) {
      await wait(delayMs)
      const me = requireUser()
      const found = quizzes.get(id)
      if (!found || found.owner !== me.id) throw new ApiError('not_found', 'That quiz was not found.')
      const results: QuestionResult[] = found.quiz.questions.map((q) => {
        const seed = found.seeds.get(q.id)!
        const given = (answers[q.id] ?? '').trim()
        const base = { questionId: q.id, question: q.question, source: q.source, yourAnswer: given }
        if (q.options) {
          const right = given === seed.answer
          return { ...base, status: right ? 'correct' : 'incorrect', score: right ? 1 : 0, correctAnswer: seed.answer, explanation: right ? undefined : seed.explanation }
        }
        const points = seed.keyPoints ?? [{ text: seed.answer, words: seed.answer.toLowerCase().split(/\W+/).filter((w) => w.length > 5) }]
        const text = given.toLowerCase()
        const keyPoints = points.map((p) => ({ text: p.text, covered: text.length > 0 && p.words.some((w) => text.includes(w)) }))
        const score = keyPoints.filter((k) => k.covered).length / keyPoints.length
        return { ...base, status: score >= 0.999 ? 'correct' : score > 0 ? 'partial' : 'incorrect', score, keyPoints, correctAnswer: seed.answer }
      })
      const byTopic = new Map<string, number[]>()
      results.forEach((r, i) => {
        const topic = found.quiz.questions[i].topic
        byTopic.set(topic, [...(byTopic.get(topic) ?? []), r.score])
      })
      const result: GradeResult = {
        quizId: id,
        title: found.quiz.title,
        average: results.reduce((s, r) => s + r.score, 0) / Math.max(results.length, 1),
        correct: results.filter((r) => r.status === 'correct').length,
        total: results.length,
        results,
        topics: [...byTopic].map(([topic, scores]) => ({ topic, score: scores.reduce((a, b) => a + b, 0) / scores.length })).sort((a, b) => a.score - b.score),
      }
      found.result = result
      return result
    },

    async getResult(quizId) {
      await wait(delayMs)
      const me = requireUser()
      const found = quizzes.get(quizId)
      if (!found || found.owner !== me.id || !found.result) throw new ApiError('not_found', 'There are no results for this quiz yet.')
      return found.result
    },

    async chat(documentIds, message): Promise<ChatAnswer> {
      await wait(delayMs)
      const me = requireUser()
      const chosen = documentIds.map((id) => ownedDoc(me.id, id)).filter((d): d is IngestedDocument => !!d)
      if (chosen.length === 0) throw new ApiError('no_documents', 'Add at least one file to search.')
      if (!/enzyme|temperature/i.test(message)) {
        return { kind: 'refusal', text: 'I could not find that in the files you are searching. I would rather say so than guess.' }
      }
      const a = chosen[0]
      const b = chosen[chosen.length > 1 ? 1 : 0]
      return {
        kind: 'answer',
        text: 'Enzyme activity was highest at 37 °C and fell sharply at 60 °C [1]. The notes explain the drop as the enzyme losing its shape at high temperature [2].',
        citations: [
          {
            n: 1,
            source: { file: a.name, page: 9 },
            passage: 'Table 2. Enzyme activity by temperature',
            readWith: a.method === 'ocr' ? 'gemini' : undefined,
            table: {
              caption: 'Table 2. Enzyme activity by temperature',
              header: ['Temperature', 'Activity (sample units)'],
              rows: [['20 °C', '12'], ['37 °C', '45'], ['60 °C', '8']],
              highlightRow: 1,
            },
          },
          {
            n: 2,
            source: { file: b.name, page: 31 },
            passage: 'Enzymes are proteins. At high temperatures the enzyme loses its shape and can no longer bind its substrate, so the reaction slows or stops.',
          },
        ],
      }
    },
  }
}
