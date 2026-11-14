import { screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it } from 'vitest'
import { pdf, renderApp } from './test/render'

type User = ReturnType<typeof userEvent.setup>

async function addFile(user: User, name: string) {
  await user.upload(screen.getByLabelText('Choose files'), pdf(name))
}

async function enterGeminiKey(user: User) {
  await user.type(screen.getByLabelText('Gemini API key'), 'test-key')
}

async function makeQuiz(user: User, type = 'Multiple choice') {
  await enterGeminiKey(user)
  await addFile(user, 'Notes.pdf')
  await screen.findByText('Ready')
  await user.click(screen.getByLabelText(type))
  await user.click(screen.getByRole('button', { name: 'Create quiz' }))
  await screen.findByRole('heading', { name: /Question 1 of 10/ })
}

describe('upload and options', () => {
  it('only lets you create a quiz once a file is ready and a key is entered', async () => {
    const user = userEvent.setup()
    await renderApp()
    const create = screen.getByRole('button', { name: 'Create quiz' })
    await waitFor(() => expect(screen.queryByText('Loading your files…')).not.toBeInTheDocument())
    expect(create).toBeDisabled()
    expect(screen.getByRole('status')).toHaveTextContent(/Add a file/)

    await addFile(user, 'Notes.pdf')
    await screen.findByText('Ready')
    expect(create).toBeDisabled()
    expect(screen.getByRole('status')).toHaveTextContent(/API key/)

    await enterGeminiKey(user)
    expect(create).toBeEnabled()
  })

  it('shows page-by-page reading of a scan, then what was kept', async () => {
    const user = userEvent.setup()
    await renderApp()
    await enterGeminiKey(user)
    await addFile(user, 'Lecture 3 scan.pdf')
    await screen.findByText('Ready')
    expect(screen.getByText(/12 pages. Scanned, read with Gemini/)).toBeInTheDocument()
    expect(screen.getByText('4 tables kept')).toBeInTheDocument()
    expect(screen.getByText('2 figures described')).toBeInTheDocument()
  })

  it('explains why a scan over the page limit was refused', async () => {
    const user = userEvent.setup()
    await renderApp()
    await enterGeminiKey(user)
    await addFile(user, 'Genetics big scan.pdf')
    expect(await screen.findByText(/42 pages and the limit is 30/)).toBeInTheDocument()
    expect(screen.getByText('Not added')).toBeInTheDocument()
  })

  it('asks for Gemini when an image is added without it, and accepts it with a key', async () => {
    const user = userEvent.setup()
    await renderApp()
    await user.upload(screen.getByLabelText('Choose files'), new File(['x'], 'diagram.png', { type: 'image/png' }))
    expect(await screen.findByText(/needs the Gemini provider/)).toBeInTheDocument()

    await enterGeminiKey(user)
    await user.upload(screen.getByLabelText('Choose files'), new File(['x'], 'cells.png', { type: 'image/png' }))
    expect(await screen.findByText(/Image read with Gemini/)).toBeInTheDocument()
  })

  it('removes a file from the list', async () => {
    const user = userEvent.setup()
    await renderApp()
    await addFile(user, 'Notes.pdf')
    await screen.findByText('Ready')
    await user.click(screen.getByRole('button', { name: 'Remove Notes.pdf' }))
    expect(screen.queryByText('Notes.pdf')).not.toBeInTheDocument()
  })

  it('does not ask for a key with Ollama', async () => {
    const user = userEvent.setup()
    await renderApp()
    await user.click(screen.getByLabelText('Ollama'))
    expect(screen.queryByLabelText('Gemini API key')).not.toBeInTheDocument()
    await addFile(user, 'Notes.pdf')
    await screen.findByText('Ready')
    expect(screen.getByRole('button', { name: 'Create quiz' })).toBeEnabled()
  })

  it('prefills the topic from the link on the results page', async () => {
    await renderApp('/?topic=Mitosis%20phases')
    expect(screen.getByLabelText(/Topic filter/)).toHaveValue('Mitosis phases')
  })
})

describe('taking a quiz and seeing results', () => {
  it('goes from upload to graded results', async () => {
    const user = userEvent.setup()
    await renderApp()
    await makeQuiz(user)

    expect(screen.getByRole('heading', { level: 2, name: /Which organelle/ })).toBeInTheDocument()
    expect(screen.getByText('Notes.pdf, page 5')).toBeInTheDocument()

    const choice = screen.getByRole('radio', { name: /Mitochondrion/ })
    expect(choice).toHaveAttribute('aria-checked', 'false')
    await user.click(choice)
    expect(choice).toHaveAttribute('aria-checked', 'true')
    expect(screen.getByText('1 answered')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Question 1, current' })).toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: 'Next' }))
    await screen.findByRole('heading', { name: /Question 2 of 10/ })
    expect(screen.getByRole('button', { name: 'Question 1, answered' })).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'Previous' }))
    expect(screen.getByRole('radio', { name: /Mitochondrion/ })).toHaveAttribute('aria-checked', 'true')

    await user.click(screen.getByRole('button', { name: 'Submit answers and grade me' }))
    await screen.findByRole('heading', { name: 'Your results' })
    const stats = screen.getByRole('region', { name: 'Summary' })
    expect(within(stats).getByText(/of 10/)).toBeInTheDocument()
    expect(within(stats).getByText(/\d+ partly right, \d+ wrong/)).toBeInTheDocument()
  })

  it('lets you jump with the navigator and shows short answers as a text box', async () => {
    const user = userEvent.setup()
    await renderApp()
    await makeQuiz(user, 'Short answer')
    expect(screen.getByLabelText('Your answer')).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'Question 4, not answered' }))
    await screen.findByRole('heading', { name: /Question 4 of 10/ })
  })

  it('shows key points covered and missed for a short answer, and a link to practise the weakest topic', async () => {
    const user = userEvent.setup()
    await renderApp()
    await makeQuiz(user, 'Short answer')
    await user.type(screen.getByLabelText('Your answer'), 'The DNA must be copied when the cell is ready')
    await user.click(screen.getByRole('button', { name: 'Submit answers and grade me' }))
    await screen.findByRole('heading', { name: 'Your results' })
    expect(screen.getByText(/Partly right, 67%/)).toBeInTheDocument()
    const [first] = screen.getAllByRole('article')
    expect(within(first).getAllByText('Covered:', { exact: false })).toHaveLength(2)
    expect(within(first).getAllByText('Missed:', { exact: false })).toHaveLength(1)
    const practise = screen.getByRole('link', { name: /^Practise / })
    expect(practise.getAttribute('href')).toMatch(/^\/\?topic=/)
  })

  it('shows what the right answer was for a wrong choice', async () => {
    const user = userEvent.setup()
    await renderApp()
    await makeQuiz(user)
    await user.click(screen.getByRole('radio', { name: /Nucleus/ }))
    await user.click(screen.getByRole('button', { name: 'Submit answers and grade me' }))
    await screen.findByRole('heading', { name: 'Your results' })
    // the quiz repeats questions, so the first card is the one for question 1
    const [first] = screen.getAllByRole('article')
    expect(within(first).getByText('Your answer: Nucleus')).toBeInTheDocument()
    expect(within(first).getByText('Correct: Mitochondrion')).toBeInTheDocument()
  })

  it('lists the quiz with its score afterwards', async () => {
    const user = userEvent.setup()
    await renderApp()
    await makeQuiz(user)
    await user.click(screen.getByRole('button', { name: 'Submit answers and grade me' }))
    await screen.findByRole('heading', { name: 'Your results' })
    await user.click(screen.getByRole('link', { name: 'My quizzes' }))
    expect(await screen.findByText(/10 questions · last score 0%/)).toBeInTheDocument()
    expect(screen.getByRole('link', { name: 'Retake' })).toBeInTheDocument()
  })

  it('says so when a quiz or its results cannot be found', async () => {
    await renderApp('/quiz/missing')
    expect(await screen.findByRole('alert')).toHaveTextContent(/not found/)
  })
})

describe('chat with documents', () => {
  it('asks for files first', async () => {
    await renderApp('/chat')
    expect(screen.getByRole('heading', { name: 'No files yet' })).toBeInTheDocument()
  })

  it('answers with citations and a table, then refuses what the files do not cover', async () => {
    const user = userEvent.setup()
    await renderApp()
    await enterGeminiKey(user)
    await addFile(user, 'Lecture 3 scan.pdf')
    await screen.findByText('Ready')
    await user.click(screen.getByRole('link', { name: 'Chat with documents' }))

    const box = screen.getByLabelText('Ask a question about your documents')
    await user.type(box, 'What did the enzyme experiment show about temperature?{Enter}')
    expect(await screen.findByRole('button', { name: 'Source 1' })).toBeInTheDocument()
    const sources = screen.getByRole('complementary', { name: 'Sources' })
    expect(within(sources).getByRole('table')).toBeInTheDocument()
    expect(within(sources).getByText('Table, read with Gemini')).toBeInTheDocument()
    expect(within(sources).getByRole('row', { name: /37 °C 45/ })).toHaveClass('is-highlight')

    await user.type(box, 'Who first discovered it?{Enter}')
    expect(await screen.findByText('Not in your documents')).toBeInTheDocument()
    expect(within(sources).queryByRole('table')).not.toBeInTheDocument()
    expect(within(sources).getByText(/No sources for this reply/)).toBeInTheDocument()

    // clicking a citation in the earlier answer brings its sources back
    await user.click(screen.getByRole('button', { name: 'Source 1' }))
    expect(within(sources).getByRole('table')).toBeInTheDocument()
  })

  it('stops you asking when every file is switched off', async () => {
    const user = userEvent.setup()
    await renderApp()
    await addFile(user, 'Notes.pdf')
    await screen.findByText('Ready')
    await user.click(screen.getByRole('link', { name: 'Chat with documents' }))
    await user.click(screen.getByLabelText('Notes.pdf'))
    await user.type(screen.getByLabelText('Ask a question about your documents'), 'hello')
    expect(screen.getByRole('button', { name: 'Send' })).toBeDisabled()
    expect(screen.getByText('Select at least one file to search.')).toBeInTheDocument()
    await waitFor(() => expect(screen.getByLabelText('Notes.pdf')).not.toBeChecked())
  })
})
