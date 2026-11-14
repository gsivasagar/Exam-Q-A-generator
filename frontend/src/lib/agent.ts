/** "Chrome on macOS" from a browser's User-Agent string, for the list of devices. */
export function describeAgent(ua: string): string {
  if (!ua) return 'Unknown device'
  if (!/mozilla|applewebkit|gecko/i.test(ua)) return ua // already a readable label
  const browser = /edg\//i.test(ua) ? 'Edge' : /firefox\//i.test(ua) ? 'Firefox' : /chrome\//i.test(ua) ? 'Chrome' : /safari\//i.test(ua) ? 'Safari' : 'Browser'
  const os = /iphone|ipad/i.test(ua) ? 'iOS' : /android/i.test(ua) ? 'Android' : /windows/i.test(ua) ? 'Windows' : /mac os x|macintosh/i.test(ua) ? 'macOS' : /linux/i.test(ua) ? 'Linux' : ''
  return os ? `${browser} on ${os}` : browser
}

export function shortDate(iso: string | null): string {
  if (!iso) return ''
  const d = new Date(iso)
  return Number.isNaN(d.getTime()) ? '' : d.toLocaleDateString(undefined, { day: 'numeric', month: 'short' })
}
