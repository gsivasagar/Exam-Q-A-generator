import { describe, expect, it } from 'vitest'
import { createMockApi, OCR_MAX_PAGES } from './mock'
import type { Credentials } from './types'

const gemini: Credentials = { provider: 'Gemini', apiKey: 'k' }
const noKey: Credentials = { provider: 'Gemini', apiKey: '' }
const f = (name: string) => new File(['x'], name)
const api = () => createMockApi({ delayMs: 0, signedInAs: 'tester@example.com' })

describe('mock api rules (they mirror the back end)', () => {
  it('refuses a scan over the page limit with the limit in the message', async () => {
    const err = await api().ingestFile(f('big scan.pdf'), gemini, () => {}).catch((e) => e)
    expect(err.code).toBe('ocr_limit')
    expect(err.message).toContain(String(OCR_MAX_PAGES))
  })

  it('needs Gemini and a key for images and for scans', async () => {
    const image = await api().ingestFile(f('diagram.png'), noKey, () => {}).catch((e) => e)
    expect(image.code).toBe('needs_gemini')
    const scan = await api().ingestFile(f('lecture scan.pdf'), noKey, () => {}).catch((e) => e)
    expect(scan.code).toBe('unreadable')
  })

  it('reports page progress for a scan and counts tables and figures', async () => {
    const pages: number[] = []
    const doc = await api().ingestFile(f('lecture scan.pdf'), gemini, (p) => pages.push(p.page))
    expect(pages).toEqual(Array.from({ length: 12 }, (_, i) => i + 1))
    expect(doc).toMatchObject({ method: 'ocr', pages: 12, tables: 4, figures: 2 })
  })

  it('rejects unsupported types and bad addresses', async () => {
    expect((await api().ingestFile(f('song.mp3'), gemini, () => {}).catch((e) => e)).code).toBe('unsupported')
    expect((await api().ingestUrl('ftp://x', gemini).catch((e) => e)).code).toBe('unsupported')
  })

  it('makes the requested number of questions, of the requested type', async () => {
    const a = api()
    const doc = await a.ingestFile(f('notes.pdf'), gemini, () => {})
    const base = { documentIds: [doc.id], difficulty: 'any' as const, topic: '', ...gemini }
    const mcq = await a.createQuiz({ ...base, count: 7, type: 'mcq' })
    expect(mcq.questions).toHaveLength(7)
    expect(mcq.questions.every((q) => q.type === 'mcq' && q.options && q.options.length === 4)).toBe(true)
    const tf = await a.createQuiz({ ...base, count: 5, type: 'tf' })
    expect(tf.questions.every((q) => q.type === 'tf' && q.options?.join() === 'True,False')).toBe(true)
    const short = await a.createQuiz({ ...base, count: 5, type: 'short' })
    expect(short.questions.every((q) => q.type === 'short' && !q.options)).toBe(true)
  })

  it('needs at least one document to make a quiz', async () => {
    const err = await api().createQuiz({ documentIds: ['nope'], count: 5, type: 'mcq', difficulty: 'any', topic: '', ...gemini }).catch((e) => e)
    expect(err.code).toBe('no_documents')
  })

  it('grades short answers by the key points they cover', async () => {
    const a = api()
    const doc = await a.ingestFile(f('notes.pdf'), gemini, () => {})
    const quiz = await a.createQuiz({ documentIds: [doc.id], count: 2, type: 'short', difficulty: 'any', topic: '', ...gemini })
    const [cycle, enzyme] = quiz.questions
    const result = await a.gradeQuiz(
      quiz.id,
      { [cycle.id]: 'The DNA must be copied when the cell is ready', [enzyme.id]: 'It loses its shape and cannot bind its substrate' },
      gemini,
    )
    const [first, second] = result.results
    expect(first.status).toBe('partial')
    expect(first.score).toBeCloseTo(2 / 3)
    expect(first.keyPoints?.map((k) => k.covered)).toEqual([true, true, false])
    expect(second.status).toBe('correct')
    expect(result.correct).toBe(1)
    expect(result.average).toBeCloseTo((2 / 3 + 1) / 2)
    expect((await a.getResult(quiz.id)).average).toBe(result.average)
  })

  it('grades choices exactly and sorts topics weakest first', async () => {
    const a = api()
    const doc = await a.ingestFile(f('notes.pdf'), gemini, () => {})
    const quiz = await a.createQuiz({ documentIds: [doc.id], count: 6, type: 'mcq', difficulty: 'any', topic: '', ...gemini })
    const wrong = await a.gradeQuiz(quiz.id, Object.fromEntries(quiz.questions.map((q) => [q.id, 'not an option'])), gemini)
    expect(wrong.correct).toBe(0)
    expect(wrong.results.every((r) => r.status === 'incorrect' && r.correctAnswer)).toBe(true)
    // answer only the first question correctly: that topic must not be listed first
    const first = quiz.questions[0]
    const right = await a.gradeQuiz(quiz.id, { [first.id]: wrong.results[0].correctAnswer! }, gemini)
    expect(right.correct).toBe(1)
    expect(right.topics[0].score).toBeLessThanOrEqual(right.topics[right.topics.length - 1].score)
    expect(right.topics.find((t) => t.topic === first.topic)!.score).toBeGreaterThan(0)
  })

  it('scores a blank answer zero', async () => {
    const a = api()
    const doc = await a.ingestFile(f('notes.pdf'), gemini, () => {})
    const quiz = await a.createQuiz({ documentIds: [doc.id], count: 5, type: 'short', difficulty: 'any', topic: '', ...gemini })
    expect((await a.gradeQuiz(quiz.id, {}, gemini)).average).toBe(0)
  })

  it('answers questions about the sample table and refuses the rest', async () => {
    const a = api()
    const doc = await a.ingestFile(f('lecture scan.pdf'), gemini, () => {})
    const yes = await a.chat([doc.id], 'What did the enzyme experiment show about temperature?', gemini)
    expect(yes.kind).toBe('answer')
    if (yes.kind === 'answer') expect(yes.citations[0].table?.rows).toHaveLength(3)
    const no = await a.chat([doc.id], 'Who discovered it?', gemini)
    expect(no.kind).toBe('refusal')
    expect((await a.chat([], 'hi', gemini).catch((e) => e)).code).toBe('no_documents')
  })
})
