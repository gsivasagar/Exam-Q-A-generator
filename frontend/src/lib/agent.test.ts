import { describe, expect, it } from 'vitest'
import { describeAgent, shortDate } from './agent'

describe('describeAgent', () => {
  it('names the browser and system', () => {
    expect(describeAgent('Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36')).toBe('Chrome on macOS')
    expect(describeAgent('Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1')).toBe('Safari on iOS')
    expect(describeAgent('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0 Safari/537.36 Edg/120.0')).toBe('Edge on Windows')
    expect(describeAgent('Mozilla/5.0 (X11; Linux x86_64; rv:121.0) Gecko/20100101 Firefox/121.0')).toBe('Firefox on Linux')
  })
  it('keeps a label that is already readable and handles nothing', () => {
    expect(describeAgent('Chrome on macOS')).toBe('Chrome on macOS')
    expect(describeAgent('')).toBe('Unknown device')
  })
})

describe('shortDate', () => {
  it('formats a date and tolerates nothing or nonsense', () => {
    expect(shortDate('2026-11-21T10:00:00Z')).toMatch(/21/)
    expect(shortDate(null)).toBe('')
    expect(shortDate('not a date')).toBe('')
  })
})
