import { screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import axe from 'axe-core'
import { describe, expect, it } from 'vitest'
import { createMockApi } from './api/mock'
import type { Quiz } from './api'
import { pdf, renderApp } from './test/render'

/** Runs axe on the whole page. Colour contrast needs real layout, which jsdom does not have, so it is checked
 * separately (the palette was chosen for 4.5:1). */
async function violations(): Promise<string[]> {
  const result = await axe.run(document.body, { rules: { 'color-contrast': { enabled: false } } })
  return result.violations.map((v) => `${v.id}: ${v.help} (${v.nodes.length}) e.g. ${v.nodes[0]?.html.slice(0, 120)}`)
}

type User = ReturnType<typeof userEvent.setup>

async function quizFlow(user: User) {
  await user.type(screen.getByLabelText('Gemini API key'), 'k')
  await user.upload(screen.getByLabelText('Choose files'), pdf('Notes.pdf'))
  await screen.findByText('Ready')
  await user.click(screen.getByRole('button', { name: 'Create quiz' }))
  await screen.findByRole('heading', { name: /Question 1 of/ })
}

describe('accessibility (axe) on every screen', () => {
  it('every app page has exactly one main landmark', async () => {
    await renderApp('/')
    expect(document.querySelectorAll('main')).toHaveLength(1)
  })

  it('sign in', async () => {
    await renderApp('/signin', { signedIn: false })
    expect(await violations()).toEqual([])
  })

  it('create account', async () => {
    await renderApp('/signup', { signedIn: false })
    expect(await violations()).toEqual([])
  })

  it('new quiz with a ready file, a refused file and a reading scan', async () => {
    const user = userEvent.setup()
    await renderApp('/')
    await user.type(screen.getByLabelText('Gemini API key'), 'k')
    await user.upload(screen.getByLabelText('Choose files'), [pdf('Notes.pdf'), pdf('big scan.pdf')])
    await screen.findByText('Not added')
    expect(await violations()).toEqual([])
  })

  it('taking a quiz', async () => {
    const user = userEvent.setup()
    await renderApp('/')
    await quizFlow(user)
    expect(await violations()).toEqual([])
  })

  it('results', async () => {
    const user = userEvent.setup()
    await renderApp('/')
    await quizFlow(user)
    await user.click(screen.getByRole('button', { name: 'Submit answers and grade me' }))
    await screen.findByRole('heading', { name: 'Your results' })
    expect(await violations()).toEqual([])
  })

  it('chat with an answer, a table and a refusal', async () => {
    const user = userEvent.setup()
    await renderApp('/')
    await user.type(screen.getByLabelText('Gemini API key'), 'k')
    await user.upload(screen.getByLabelText('Choose files'), pdf('Lecture scan.pdf'))
    await screen.findByText('Ready')
    await user.click(screen.getByRole('link', { name: 'Chat with documents' }))
    await user.type(screen.getByLabelText('Ask a question about your documents'), 'What about enzyme temperature?{Enter}')
    await screen.findByRole('table')
    await user.type(screen.getByLabelText('Ask a question about your documents'), 'Who found it?{Enter}')
    await screen.findByText('Not in your documents')
    expect(await violations()).toEqual([])
  })

  it('account page', async () => {
    await renderApp('/account')
    await screen.findByText('Safari on iPhone')
    expect(await violations()).toEqual([])
  })

  it('guest session with its banner', async () => {
    const user = userEvent.setup()
    await renderApp('/signin', { signedIn: false })
    await user.click(screen.getByRole('button', { name: 'Try it without an account' }))
    await screen.findByText(/You are using a guest session/)
    expect(await violations()).toEqual([])
  })

  it('my quizzes, empty and filled', async () => {
    const user = userEvent.setup()
    await renderApp('/quizzes')
    await screen.findByRole('heading', { name: 'No quizzes yet' })
    expect(await violations()).toEqual([])
    await user.click(within(screen.getByRole('main')).getByRole('link', { name: 'New quiz' }))
    await quizFlow(user)
    await user.click(screen.getByRole('link', { name: 'My quizzes' }))
    await screen.findByText(/10 questions/)
    expect(await violations()).toEqual([])
  })
})

describe('text from files and the server is shown as text, never run as HTML', () => {
  const evil = '<img src=x onerror="window.__hacked=1">'

  it('a file name', async () => {
    const user = userEvent.setup()
    await renderApp('/')
    await user.upload(screen.getByLabelText('Choose files'), pdf(`${evil}.pdf`))
    expect(await screen.findByText(`${evil}.pdf`)).toBeInTheDocument()
    expect(document.querySelector('img[src="x"]')).toBeNull()
    expect((window as unknown as { __hacked?: number }).__hacked).toBeUndefined()
  })

  it('a question, its options and its source', async () => {
    const api = createMockApi({ delayMs: 0, signedInAs: 'me@example.com' })
    const quiz: Quiz = {
      id: 'q',
      title: `<b>title</b>`,
      createdAt: '',
      questions: [{ id: 'q0', type: 'mcq', question: `What is ${evil}?`, options: [evil, '<script>x()</script>'], topic: evil, source: { file: evil, page: 1 } }],
    }
    api.getQuiz = async () => quiz
    await renderApp('/quiz/q', { api })
    expect(await screen.findByText(`What is ${evil}?`)).toBeInTheDocument()
    expect(document.querySelector('img[src="x"]')).toBeNull()
    expect(document.querySelector('main script, main b')).toBeNull()
  })
})
