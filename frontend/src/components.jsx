import React from 'react'

export function Loading() {
  return <div className="state">Loading…</div>
}

// An error is shown, never swallowed. An empty table that actually means
// "the backend is down" is worse than no table.
export function ErrorState({ error, onRetry }) {
  return (
    <div className="state error">
      <p>{error?.detail ?? error?.message ?? 'Request failed'}</p>
      {onRetry && <button className="ghost" onClick={onRetry}>Retry</button>}
    </div>
  )
}

export function Empty({ children }) {
  return <div className="state">{children}</div>
}

export function Tag({ value }) {
  if (value === null || value === undefined) return <span className="muted">—</span>
  return <span className={`tag ${String(value)}`}>{String(value)}</span>
}

export function Card({ label, value, note }) {
  return (
    <div className="card">
      <div className="label">{label}</div>
      <div className="value">{value}</div>
      {note && <div className="note">{note}</div>}
    </div>
  )
}

export function Page({ title, subtitle, children, actions }) {
  return (
    <>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
        <div>
          <h1>{title}</h1>
          {subtitle && <p className="sub">{subtitle}</p>}
        </div>
        {actions}
      </div>
      {children}
    </>
  )
}

// Shown on every screen that could be mistaken for a trading terminal.
export function SafetyBanner() {
  return (
    <div className="banner">
      <strong>PAPER MODE</strong> — nothing here purchases, pays, publishes or
      transfers money. Those capabilities are absent from the system, not
      disabled. Every action requires explicit human approval.
    </div>
  )
}

export function DataTable({ columns, rows, empty = 'No rows.' }) {
  if (!rows || rows.length === 0) return <Empty>{empty}</Empty>
  return (
    <table>
      <thead>
        <tr>{columns.map((c) => <th key={c.key} style={c.align === 'right' ? { textAlign: 'right' } : null}>{c.label}</th>)}</tr>
      </thead>
      <tbody>
        {rows.map((row, i) => (
          <tr key={row.id ?? row.key ?? i}>
            {columns.map((c) => (
              <td key={c.key} className={c.align === 'right' ? 'num' : undefined}>
                {c.render ? c.render(row) : row[c.key] ?? '—'}
              </td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  )
}
