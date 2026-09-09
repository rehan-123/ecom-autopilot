// Single place the frontend talks to the backend.
//
// Every screen reads through here, so there is one definition of the base URL,
// one error shape, and no component inventing its own fetch semantics. The
// frontend holds no business logic: it renders what the engines computed.

const BASE = import.meta.env.VITE_API_BASE ?? ''

export class ApiError extends Error {
  constructor(status, detail, path) {
    super(`${status} ${path}: ${detail}`)
    this.status = status
    this.detail = detail
    this.path = path
  }
}

export async function getJson(path) {
  const response = await fetch(`${BASE}${path}`)
  if (!response.ok) {
    let detail = response.statusText
    try {
      detail = (await response.json()).detail ?? detail
    } catch {
      // Body was not JSON; the status text is the best available message.
    }
    throw new ApiError(response.status, detail, path)
  }
  return response.json()
}

export async function postJson(path, body) {
  const response = await fetch(`${BASE}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body ?? {}),
  })
  if (!response.ok) {
    let detail = response.statusText
    try {
      detail = (await response.json()).detail ?? detail
    } catch {
      // As above.
    }
    throw new ApiError(response.status, detail, path)
  }
  return response.json()
}

// Money arrives from the backend as a decimal string and is never parsed into
// a JS number: floating point is exactly what the backend refuses to use, and
// re-introducing it in the browser would undo that.
export function inr(value) {
  if (value === null || value === undefined || value === '') return '—'
  const [whole, fraction = '00'] = String(value).split('.')
  const negative = whole.startsWith('-')
  const digits = negative ? whole.slice(1) : whole
  let grouped = digits
  if (digits.length > 3) {
    const tail = digits.slice(-3)
    let head = digits.slice(0, -3)
    const parts = []
    while (head.length > 2) {
      parts.unshift(head.slice(-2))
      head = head.slice(0, -2)
    }
    if (head) parts.unshift(head)
    grouped = [...parts, tail].join(',')
  }
  return `${negative ? '-' : ''}₹${grouped}.${fraction.slice(0, 2).padEnd(2, '0')}`
}

export function pct(value) {
  if (value === null || value === undefined) return '—'
  return `${Number(value).toFixed(1)}%`
}
