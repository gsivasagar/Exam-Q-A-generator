import { act, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { createMockApi } from './api/mock'
import { apiWithAccount, GOOD, pdf, renderApp } from './test/render'

type User = ReturnType<typeof userEvent.setup>

async function fillSignIn(user: User, email: string, password: string) {
  await user.type(screen.getByLabelText('Email'), email)
  await user.type(screen.getByLabelText('Password'), password)
}

afterEach(() => vi.restoreAllMocks())

describe('signing in', () => {
  it('sends a signed-out visitor to the sign-in page', async () => {
    await renderApp('/quizzes', { signedIn: false })
    expect(await screen.findByRole('heading', { name: 'Sign in' })).toBeInTheDocument()
    expect(screen.queryByRole('navigation', { name: 'Main' })).not.toBeInTheDocument()
  })

  it('signs in and goes back to the page that was asked for', async () => {
    const user = userEvent.setup()
    const api = await apiWithAccount()
    const { store } = await renderApp('/quizzes', { signedIn: false, api })
    await screen.findByRole('heading', { name: 'Sign in' })
    expect(store.token).toBeNull()
    await fillSignIn(user, 'ME@example.com', GOOD)
    await user.click(screen.getByRole('button', { name: 'Sign in' }))
    expect(await screen.findByRole('heading', { name: 'My quizzes' })).toBeInTheDocument()
    expect(screen.getByText('me@example.com')).toBeInTheDocument()
    expect(store.token).toMatch(/^mock-token-/) // later calls will carry it
    await user.click(screen.getByRole('button', { name: 'Sign out' }))
    await screen.findByRole('heading', { name: 'Sign in' })
    expect(store.token).toBeNull() // and signing out forgets it
  })

  it('says the same thing for a wrong password and an unknown email', async () => {
    const user = userEvent.setup()
    const api = await apiWithAccount()
    await renderApp('/signin', { signedIn: false, api })
    await fillSignIn(user, 'me@example.com', 'definitely wrong pass')
    await user.click(screen.getByRole('button', { name: 'Sign in' }))
    const first = (await screen.findByRole('alert')).textContent

    await user.clear(screen.getByLabelText('Email'))
    await user.type(screen.getByLabelText('Email'), 'nobody@example.com')
    await user.click(screen.getByRole('button', { name: 'Sign in' }))
    await waitFor(() => expect(screen.getByRole('alert').textContent).toBe(first))
    expect(first).toMatch(/Email or password is wrong/)
  })

  it('can show and hide the password', async () => {
    const user = userEvent.setup()
    await renderApp('/signin', { signedIn: false })
    const field = screen.getByLabelText('Password')
    expect(field).toHaveAttribute('type', 'password')
    await user.click(screen.getByRole('button', { name: 'Show password' }))
    expect(field).toHaveAttribute('type', 'text')
    await user.click(screen.getByRole('button', { name: 'Hide password' }))
    expect(field).toHaveAttribute('type', 'password')
  })

  it('a signed-in visitor is moved off the sign-in page', async () => {
    await renderApp('/signin')
    expect(await screen.findByRole('heading', { name: 'New quiz' })).toBeInTheDocument()
  })

  it('tells you when the server ended your session', async () => {
    const { store } = await renderApp('/quizzes')
    act(() => store.expire())
    expect(await screen.findByRole('heading', { name: 'Sign in' })).toBeInTheDocument()
    expect(screen.getByRole('alert')).toHaveTextContent('Your session ended. Sign in again.')
  })
})

describe('creating an account', () => {
  it('counts the password as you type', async () => {
    const user = userEvent.setup()
    await renderApp('/signup', { signedIn: false })
    const hint = () => screen.getByRole('status')
    expect(hint()).toHaveTextContent('At least 12 characters')
    await user.type(screen.getByLabelText('Password'), 'short pass')
    expect(hint()).toHaveTextContent('10 characters. At least 12 needed.')
    await user.type(screen.getByLabelText('Password'), 'word here')
    expect(hint()).toHaveTextContent('19 characters. Long enough.')
  })

  it('creates the account and lands in the app', async () => {
    const user = userEvent.setup()
    await renderApp('/signup', { signedIn: false })
    await user.type(screen.getByLabelText('Email'), 'new@example.com')
    await user.type(screen.getByLabelText('Password'), GOOD)
    await user.click(screen.getByRole('button', { name: 'Create account' }))
    expect(await screen.findByRole('heading', { name: 'New quiz' })).toBeInTheDocument()
    expect(screen.getByText('new@example.com')).toBeInTheDocument()
  })

  it('shows the server reason for a weak password, a bad email and a taken email', async () => {
    const user = userEvent.setup()
    const api = await apiWithAccount('taken@example.com')
    await renderApp('/signup', { signedIn: false, api })
    const email = screen.getByLabelText('Email')
    const password = screen.getByLabelText('Password')
    const submit = () => user.click(screen.getByRole('button', { name: 'Create account' }))

    await user.type(email, 'fresh@example.com')
    await user.type(password, 'short')
    await submit()
    expect(await screen.findByRole('alert')).toHaveTextContent(/at least 12/i)

    await user.clear(password)
    await user.type(password, GOOD)
    await user.clear(email)
    await user.type(email, 'taken@example.com')
    await submit()
    await waitFor(() => expect(screen.getByRole('alert')).toHaveTextContent(/already exists/))
  })
})

describe('guests', () => {
  async function asGuest(user: User) {
    await renderApp('/signin', { signedIn: false })
    await user.click(screen.getByRole('button', { name: 'Try it without an account' }))
    await screen.findByRole('navigation', { name: 'Main' })
  }

  it('can use the app and sees when their work is deleted', async () => {
    const user = userEvent.setup()
    await asGuest(user)
    expect(screen.getByText(/You are using a guest session/)).toBeInTheDocument()
    expect(screen.getByText(/are deleted on/)).toBeInTheDocument()
    expect(screen.getByText(/Kept until/)).toBeInTheDocument()
    expect(screen.queryByRole('link', { name: /Account/ })).not.toBeInTheDocument()
  })

  it('can hide the banner', async () => {
    const user = userEvent.setup()
    await asGuest(user)
    await user.click(screen.getByRole('button', { name: 'Not now' }))
    expect(screen.queryByText(/You are using a guest session/)).not.toBeInTheDocument()
  })

  it('keeps their files when they create an account', async () => {
    const user = userEvent.setup()
    await asGuest(user)
    await user.upload(screen.getByLabelText('Choose files'), pdf('Guest notes.pdf'))
    await screen.findByText('Ready')

    await user.click(screen.getByRole('link', { name: 'Keep my work' }))
    expect(await screen.findByText(/moves into this account: 1 file and 0 quizzes/)).toBeInTheDocument()
    await user.type(screen.getByLabelText('Email'), 'kept@example.com')
    await user.type(screen.getByLabelText('Password'), GOOD)
    await user.click(screen.getByRole('button', { name: 'Create account' }))

    await screen.findByRole('heading', { name: 'New quiz' })
    expect(screen.queryByText(/You are using a guest session/)).not.toBeInTheDocument()
    expect(screen.getByText('kept@example.com')).toBeInTheDocument()
    expect(await screen.findByText('Guest notes.pdf')).toBeInTheDocument()
  })

  it('stops at the file limit and says how to add more', async () => {
    const user = userEvent.setup()
    await asGuest(user)
    for (const name of ['a.pdf', 'b.pdf', 'c.pdf', 'd.pdf']) {
      await user.upload(screen.getByLabelText('Choose files'), pdf(name))
    }
    expect(await screen.findByText(/Guests can add up to 3 files/)).toBeInTheDocument()
  })

  it('invites a guest to create an account from the sidebar', async () => {
    const user = userEvent.setup()
    await asGuest(user)
    await user.click(screen.getByRole('link', { name: 'Create account' }))
    expect(await screen.findByRole('heading', { name: 'Create an account' })).toBeInTheDocument()
  })

  it('has no password to change: the account page offers to create an account instead', async () => {
    const user = userEvent.setup()
    await asGuest(user)
    await user.click(screen.getByRole('link', { name: 'Create account' })) // go to sign-up, then back to the app
    await screen.findByRole('heading', { name: 'Create an account' })
    expect(screen.queryByLabelText('Current password')).not.toBeInTheDocument()
  })
})

describe('signing out', () => {
  it('returns to sign-in and protected pages stay closed', async () => {
    const user = userEvent.setup()
    await renderApp('/')
    await user.click(screen.getByRole('button', { name: 'Sign out' }))
    expect(await screen.findByRole('heading', { name: 'Sign in' })).toBeInTheDocument()
    expect(screen.queryByRole('navigation', { name: 'Main' })).not.toBeInTheDocument()
  })

  it('signs out on this page even when the server cannot be reached', async () => {
    const user = userEvent.setup()
    const api = createMockApi({ delayMs: 0, signedInAs: 'me@example.com' })
    api.logout = () => Promise.reject(new Error('offline'))
    await renderApp('/', { api })
    await user.click(screen.getByRole('button', { name: 'Sign out' }))
    expect(await screen.findByRole('heading', { name: 'Sign in' })).toBeInTheDocument()
  })
})

describe('accounts are separate', () => {
  it('shows each person only their own files, and brings them back after signing in again', async () => {
    const user = userEvent.setup()
    const api = await apiWithAccount('alice@example.com')
    await renderApp('/signin', { signedIn: false, api })
    await fillSignIn(user, 'alice@example.com', GOOD)
    await user.click(screen.getByRole('button', { name: 'Sign in' }))
    await screen.findByRole('heading', { name: 'New quiz' })
    await user.upload(screen.getByLabelText('Choose files'), pdf('Alice private notes.pdf'))
    await screen.findByText('Ready')
    await user.click(screen.getByRole('button', { name: 'Sign out' }))

    // Bob signs up on the same device.
    await screen.findByRole('heading', { name: 'Sign in' })
    await user.click(screen.getByRole('link', { name: 'Create an account' }))
    await user.type(screen.getByLabelText('Email'), 'bob@example.com')
    await user.type(screen.getByLabelText('Password'), GOOD)
    await user.click(screen.getByRole('button', { name: 'Create account' }))
    await screen.findByRole('heading', { name: 'New quiz' })
    expect(screen.queryByText('Alice private notes.pdf')).not.toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: 'Sign out' }))
    await screen.findByRole('heading', { name: 'Sign in' })
    await fillSignIn(user, 'alice@example.com', GOOD)
    await user.click(screen.getByRole('button', { name: 'Sign in' }))
    expect(await screen.findByText('Alice private notes.pdf')).toBeInTheDocument()
  })

  it('keeps the Gemini key and files of a guest who goes to create an account', async () => {
    const user = userEvent.setup()
    await renderApp('/signin', { signedIn: false })
    await user.click(screen.getByRole('button', { name: 'Try it without an account' }))
    await screen.findByRole('navigation', { name: 'Main' })
    await user.type(screen.getByLabelText('Gemini API key'), 'guest-key')
    await user.click(screen.getByRole('link', { name: 'Keep my work' }))
    await screen.findByRole('heading', { name: 'Create an account' })
    await user.click(screen.getByRole('link', { name: 'Sign in' })) // any way back into the app is fine
    expect(await screen.findByRole('heading', { name: 'New quiz' })).toBeInTheDocument()
    expect(screen.getByLabelText('Gemini API key')).toHaveValue('guest-key')
  })

  it('clears the Gemini key when someone signs out', async () => {
    const user = userEvent.setup()
    await renderApp('/')
    await user.type(screen.getByLabelText('Gemini API key'), 'my-secret-key')
    await user.click(screen.getByRole('button', { name: 'Sign out' }))
    await screen.findByRole('heading', { name: 'Sign in' })
    await user.click(screen.getByRole('button', { name: 'Try it without an account' }))
    await screen.findByRole('navigation', { name: 'Main' })
    expect(screen.getByLabelText('Gemini API key')).toHaveValue('')
  })
})

describe('the account page', () => {
  it('changes the password and checks the current one', async () => {
    const user = userEvent.setup()
    await renderApp('/account')
    const current = screen.getByLabelText('Current password')
    const next = screen.getByLabelText('New password')
    await user.type(current, 'not the real one')
    await user.type(next, 'another long passphrase')
    await user.click(screen.getByRole('button', { name: 'Change password' }))
    expect(await screen.findByRole('alert')).toHaveTextContent('The current password is wrong.')

    await user.clear(current)
    await user.type(current, GOOD)
    await user.click(screen.getByRole('button', { name: 'Change password' }))
    expect(await screen.findByText(/Password changed/)).toBeInTheDocument()
    expect(screen.getByLabelText('Current password')).toHaveValue('') // the fields are emptied
  })

  it('lists devices and signs out the others', async () => {
    const user = userEvent.setup()
    await renderApp('/account')
    const devices = await screen.findByRole('region', { name: 'Where you are signed in' })
    expect(await within(devices).findByText('Chrome on macOS')).toBeInTheDocument()
    expect(within(devices).getByText('Safari on iPhone')).toBeInTheDocument()
    expect(within(devices).getByText('Current')).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'Sign out everywhere else' }))
    await waitFor(() => expect(screen.queryByText('Safari on iPhone')).not.toBeInTheDocument())
    expect(screen.queryByRole('button', { name: 'Sign out everywhere else' })).not.toBeInTheDocument()
  })

  it('signs out one device', async () => {
    const user = userEvent.setup()
    await renderApp('/account')
    const devices = await screen.findByRole('region', { name: 'Where you are signed in' })
    await within(devices).findByText('Safari on iPhone')
    await user.click(within(devices).getByRole('button', { name: 'Sign out' }))
    await waitFor(() => expect(screen.queryByText('Safari on iPhone')).not.toBeInTheDocument())
  })

  it('shows counts and downloads the data', async () => {
    const user = userEvent.setup()
    await renderApp('/account')
    expect(await screen.findByText('files')).toBeInTheDocument()
    const create = vi.fn(() => 'blob:fake')
    const revoke = vi.fn()
    Object.assign(URL, { createObjectURL: create, revokeObjectURL: revoke })
    const click = vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => undefined)
    await user.click(screen.getByRole('button', { name: 'Download my data (JSON)' }))
    await waitFor(() => expect(click).toHaveBeenCalled())
    expect(create).toHaveBeenCalledTimes(1)
    expect(revoke).toHaveBeenCalledWith('blob:fake')
  })

  it('deletes the account only after the email is typed, and the account is gone', async () => {
    const user = userEvent.setup()
    const api = createMockApi({ delayMs: 0, signedInAs: 'gone@example.com' })
    await renderApp('/account', { api })
    const button = await screen.findByRole('button', { name: 'Delete my account and data' })
    expect(button).toBeDisabled()
    await user.type(screen.getByLabelText('Type your email to confirm'), 'wrong@example.com')
    expect(button).toBeDisabled()
    await user.clear(screen.getByLabelText('Type your email to confirm'))
    await user.type(screen.getByLabelText('Type your email to confirm'), 'GONE@example.com')
    expect(button).toBeEnabled()
    await user.click(button)
    expect(await screen.findByRole('heading', { name: 'Sign in' })).toBeInTheDocument()
    await expect(api.login('gone@example.com', 'a long test passphrase', true)).rejects.toMatchObject({ code: 'invalid_credentials' })
  })

  it('needs sign-in to open', async () => {
    await renderApp('/account', { signedIn: false })
    expect(await screen.findByRole('heading', { name: 'Sign in' })).toBeInTheDocument()
  })
})

describe('sample data mode is impossible to miss', () => {
  it('says so on every app page and on sign-in', async () => {
    await renderApp('/')
    expect(screen.getByText('Sample data mode.')).toBeInTheDocument()
    expect(screen.getByText(/not from your\s+files/)).toBeInTheDocument()
  })

  it('says nothing when the real back end is used', async () => {
    const api = createMockApi({ delayMs: 0, signedInAs: 'me@example.com' })
    const real = { ...api, sample: false }
    await renderApp('/', { api: real })
    expect(screen.queryByText('Sample data mode.')).not.toBeInTheDocument()
  })
})

