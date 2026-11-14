import { describe, expect, it, vi } from 'vitest'
import { createHttpApi } from './http'
import { ApiError, type Credentials } from './types'

const creds: Credentials = { provider: 'Gemini', apiKey: 'secret-key' }
const file = new File(['x'], 'scan.pdf', { type: 'application/pdf' })

function ndjson(lines: object[]): Response {
  return new Response(lines.map((l) => JSON.stringify(l)).join('\n') + '\n', { status: 200 })
}

describe('http api', () => {
  it('reads progress events and the finished document from a streamed upload', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      ndjson([
        { event: 'progress', page: 1, pages: 2, tables: 0, figures: 0 },
        { event: 'progress', page: 2, pages: 2, tables: 1, figures: 0 },
        { event: 'done', document: { id: 'd1', name: 'scan.pdf', pages: 2, method: 'ocr', tables: 1, figures: 0 } },
      ]),
    )
    const seen: number[] = []
    const doc = await createHttpApi('/api', fetchMock).ingestFile(file, creds, (p) => seen.push(p.page))
    expect(seen).toEqual([1, 2])
    expect(doc.id).toBe('d1')
  })

  it('turns an error event in the stream into a typed error', async () => {
    const fetchMock = vi.fn().mockResolvedValue(ndjson([{ event: 'error', code: 'ocr_limit', message: 'Too long: 42 pages' }]))
    const err = await createHttpApi('/api', fetchMock).ingestFile(file, creds, () => {}).catch((e) => e)
    expect(err).toBeInstanceOf(ApiError)
    expect(err.code).toBe('ocr_limit')
    expect(err.message).toBe('Too long: 42 pages')
  })

  it('fails clearly when the stream ends without a result', async () => {
    const fetchMock = vi.fn().mockResolvedValue(ndjson([{ event: 'progress', page: 1, pages: 2, tables: 0, figures: 0 }]))
    await expect(createHttpApi('/api', fetchMock).ingestFile(file, creds, () => {})).rejects.toThrow(/stopped/)
  })

  it('sends the key in headers only, never in the URL or the body', async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify({ kind: 'refusal', text: 'no' }), { status: 200 }))
    await createHttpApi('/api', fetchMock).chat(['d1'], 'hello', creds)
    const [url, init] = fetchMock.mock.calls[0]
    expect(url).toBe('/api/chat')
    expect(init.headers['X-Api-Key']).toBe('secret-key')
    expect(init.headers['X-Provider']).toBe('Gemini')
    expect(String(url)).not.toContain('secret-key')
    expect(init.body).not.toContain('secret-key')
  })

  it('omits the key header when there is no key', async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response('[]', { status: 200 }))
    await createHttpApi('/api', fetchMock).chat(['d1'], 'hi', { provider: 'Ollama', apiKey: '' }).catch(() => {})
    expect(fetchMock.mock.calls[0][1].headers['X-Api-Key']).toBeUndefined()
  })

  it('maps error responses to codes, with a fallback from the status', async () => {
    const api = (res: Response) => createHttpApi('/api', vi.fn().mockResolvedValue(res))
    const a = await api(new Response(JSON.stringify({ code: 'needs_gemini', message: 'Pick Gemini' }), { status: 400 })).getQuiz('q').catch((e) => e)
    expect([a.code, a.message]).toEqual(['needs_gemini', 'Pick Gemini'])
    const b = await api(new Response('{}', { status: 404 })).getQuiz('q').catch((e) => e)
    expect(b.code).toBe('not_found')
    const c = await api(new Response('{}', { status: 429 })).getQuiz('q').catch((e) => e)
    expect(c.code).toBe('rate_limited')
    const d = await api(new Response('<html>', { status: 500 })).getQuiz('q').catch((e) => e)
    expect(d.code).toBe('unknown')
  })

  it('reports an unreachable server as a network error', async () => {
    const err = await createHttpApi('/api', vi.fn().mockRejectedValue(new TypeError('fetch failed'))).listQuizzes().catch((e) => e)
    expect(err.code).toBe('network')
  })

  it('encodes ids in paths', async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response('{}', { status: 200 }))
    await createHttpApi('/api', fetchMock).getQuiz('a/b c')
    expect(fetchMock.mock.calls[0][0]).toBe('/api/quizzes/a%2Fb%20c')
  })
})
