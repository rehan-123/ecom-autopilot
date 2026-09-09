import React, { useState } from 'react'
import { useApi } from '../useApi'
import { getJson } from '../api'
import { Card, DataTable, Empty, ErrorState, Loading, Page, SafetyBanner, Tag } from '../components'

// Phase 3 makes marketplace readiness visible without implying execution.
// The three "not available" states are kept visually distinct because they
// mean different things: UNSUPPORTED can never be fixed, NOT_CONFIGURED needs
// credentials, NOT_READY is about this product rather than the channel.

function StatusTag({ status }) {
  const cls = status === 'SUPPORTED' || status === 'READY' ? 'ok'
    : status === 'NOT_CONFIGURED' || status === 'REQUIRES_HUMAN_REVIEW' ? 'REQUIRES_REVIEW'
    : 'NOT_SUPPORTED'
  return <span className={`tag ${cls}`}>{status}</span>
}

export default function Marketplaces() {
  const q = useApi('/api/marketplaces/registry')
  const opportunities = useApi('/api/opportunities')
  const [validation, setValidation] = useState(null)
  const [busy, setBusy] = useState(false)

  async function check(slug) {
    setBusy(true)
    try {
      setValidation(await getJson(`/api/listings/${slug}/validate`))
    } catch (e) {
      setValidation({ error: e.detail ?? e.message })
    } finally { setBusy(false) }
  }

  if (q.loading) return <Loading />
  if (q.error) return <ErrorState error={q.error} onRetry={q.reload} />

  return (
    <Page title="Marketplaces" subtitle="Readiness only — nothing here publishes, orders or syncs.">
      <SafetyBanner />
      <div className="banner">
        <strong>PHASE 3</strong> — adapter and readiness infrastructure. Write
        operations are disabled even where an adapter and credentials exist.
        Nothing on this screen has any external effect.
      </div>

      <div className="cards">
        <Card label="Marketplaces" value={q.data.marketplaces.length} />
        <Card label="Live execution" value="DISABLED" note="Phase 3 boundary" />
        <Card label="Configured"
          value={q.data.marketplaces.filter((m) => m.credentials_configured).length}
          note="credentials present" />
      </div>

      <h2>Adapter status</h2>
      <DataTable
        columns={[
          { key: 'name', label: 'Marketplace' },
          { key: 'access_model', label: 'Access' },
          { key: 'credentials_configured', label: 'Credentials',
            render: (r) => <StatusTag status={r.credentials_configured ? 'SUPPORTED' : 'NOT_CONFIGURED'} /> },
          { key: 'live_transport', label: 'Transport',
            render: (r) => <StatusTag status={r.live_transport ? 'SUPPORTED' : 'NOT_CONFIGURED'} /> },
          { key: 'supported', label: 'Capabilities', render: (r) => {
            const counts = r.capabilities.reduce((acc, c) => {
              acc[c.status] = (acc[c.status] ?? 0) + 1; return acc
            }, {})
            return <span className="small muted">
              {Object.entries(counts).map(([k, v]) => `${k} ${v}`).join(' · ')}
            </span>
          } },
        ]}
        rows={q.data.marketplaces.map((m) => ({ ...m, id: m.name }))}
      />

      <h2>Capability detail</h2>
      {q.data.marketplaces.map((m) => (
        <div key={m.name} style={{ marginBottom: 22 }}>
          <div className="small muted" style={{ marginBottom: 6 }}>{m.name}</div>
          <DataTable
            columns={[
              { key: 'capability', label: 'Capability' },
              { key: 'status', label: 'Status', render: (r) => <StatusTag status={r.status} /> },
              { key: 'write_blocked_for_phase_3', label: 'Phase 3',
                render: (r) => r.write_blocked_for_phase_3
                  ? <span className="tag NOT_SUPPORTED">WRITE DISABLED</span>
                  : <span className="muted">—</span> },
              { key: 'reason', label: 'Reason',
                render: (r) => <span className="small muted">{r.reason || '—'}</span> },
            ]}
            rows={m.capabilities.map((c, i) => ({ ...c, id: `${m.name}-${i}` }))}
          />
        </div>
      ))}

      <h2>Listing readiness</h2>
      {opportunities.loading ? <Loading /> : (
        <DataTable
          columns={[
            { key: 'title', label: 'Product' },
            { key: 'verdict', label: 'Verdict', render: (r) => <Tag value={r.verdict} /> },
            { key: 'check', label: '', render: (r) => (
              <button className="ghost" disabled={busy} onClick={() => check(r.slug)}>
                Check readiness
              </button>) },
          ]}
          rows={(opportunities.data?.opportunities ?? []).map((o) => ({ ...o, id: o.slug }))}
        />
      )}

      {validation && (validation.error ? <ErrorState error={validation} /> : (
        <>
          <h2>{validation.sku_code} — <StatusTag status={validation.readiness} /></h2>
          {validation.missing_factual_fields.length > 0 && (
            <p className="muted small">
              Missing sourced fields: {validation.missing_factual_fields.join(', ')}.
              These stay missing rather than being filled in.
            </p>
          )}
          {validation.assessments.map((a) => (
            <div key={a.marketplace} style={{ marginBottom: 18 }}>
              <div className="small" style={{ marginBottom: 6 }}>
                {a.marketplace} — <StatusTag status={a.readiness} />{' '}
                <StatusTag status={a.capability} />
              </div>
              {a.findings.length === 0 ? <Empty>No findings.</Empty> : (
                <DataTable
                  columns={[
                    { key: 'code', label: 'Code' },
                    { key: 'field', label: 'Field' },
                    { key: 'severity', label: 'Severity',
                      render: (r) => <span className={`tag ${r.blocks ? 'REJECTED' : 'REQUIRES_REVIEW'}`}>{r.severity}</span> },
                    { key: 'message', label: 'Detail',
                      render: (r) => <span className="small">{r.message}</span> },
                  ]}
                  rows={a.findings.map((f, i) => ({ ...f, id: `${a.marketplace}-${i}` }))}
                />
              )}
              {a.notes.map((n, i) => <p key={i} className="muted small">{n}</p>)}
            </div>
          ))}
        </>
      ))}
    </Page>
  )
}
