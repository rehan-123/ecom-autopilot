import React, { useState } from 'react'
import { useApi } from '../useApi'
import { getJson, postJson } from '../api'
import { Card, DataTable, Empty, ErrorState, Loading, Page, SafetyBanner, Tag } from '../components'

// Execution control. Live actions are visually distinct from dry runs, and a
// result is never shown as successful-with-effect unless the adapter confirmed
// it — the API reports side_effect_occurred separately from status precisely
// so this screen does not have to infer it.

function ModeBadge({ enabled }) {
  return enabled
    ? <span className="tag REJECTED">LIVE ENABLED</span>
    : <span className="tag ok">DRY RUN ONLY</span>
}

export default function Execution() {
  const [filters, setFilters] = useState({ state: '', marketplace: '', q: '' })
  const query = new URLSearchParams(
    Object.entries(filters).filter(([, v]) => v)
  ).toString()
  const history = useApi(`/api/executions/history?limit=200${query ? `&${query}` : ''}`)
  const q = useApi('/api/executions')
  const opportunities = useApi('/api/opportunities')
  const [detail, setDetail] = useState(null)
  const [busy, setBusy] = useState(false)
  const [notice, setNotice] = useState(null)

  async function act(id, verb, body, slug) {
    setBusy(true)
    try {
      const path = verb === 'run'
        ? `/api/executions/${id}/run?slug=${encodeURIComponent(slug)}`
        : `/api/executions/${id}/${verb}`
      const result = await postJson(path, body)
      setNotice(verb === 'run'
        ? `${result.result.mode}: ${result.result.status} — external side effect: ${result.result.side_effect_occurred}`
        : `${verb} recorded`)
      setDetail(verb === 'run' ? result.result : null)
      q.reload()
    } catch (e) {
      setNotice(e.detail ?? e.message)
    } finally { setBusy(false) }
  }

  async function create(slug) {
    setBusy(true)
    try {
      await postJson('/api/executions', {
        operation: 'PRICE_UPDATE', marketplace: 'amazon_in', slug,
        payload: { price_inr: '1299' }, reason: 'operator-initiated',
      })
      q.reload()
    } catch (e) { setNotice(e.detail ?? e.message) } finally { setBusy(false) }
  }

  if (q.loading) return <Loading />
  if (q.error) return <ErrorState error={q.error} onRetry={q.reload} />

  return (
    <Page title="Execution" subtitle="Every operation requires explicit human approval.">
      <SafetyBanner />
      <div className="banner">
        <strong>PHASE 4</strong> — controlled execution. Mode:{' '}
        <ModeBadge enabled={q.data.live_execution_enabled} />. Approval is bound
        to one exact operation, SKU, marketplace and payload; changing the
        payload invalidates it. All twelve gates are rechecked immediately
        before every attempt.
      </div>
      {notice && <div className="banner">{notice}</div>}

      <div className="cards">
        <Card label="Requests" value={q.data.executions.length} />
        <Card label="Default mode" value={q.data.default_mode} />
        <Card label="Live execution"
          value={q.data.live_execution_enabled ? 'ENABLED' : 'DISABLED'}
          note="requires an explicit environment flag" />
      </div>

      <h2>Create a request</h2>
      <DataTable
        columns={[
          { key: 'title', label: 'Product' },
          { key: 'verdict', label: 'Verdict', render: (r) => <Tag value={r.verdict} /> },
          { key: 'act', label: '', render: (r) => (
            <button className="ghost" disabled={busy} onClick={() => create(r.slug)}>
              Request PRICE_UPDATE
            </button>) },
        ]}
        rows={(opportunities.data?.opportunities ?? []).slice(0, 5).map((o) => ({ ...o, id: o.slug }))}
      />

      <h2>Durable execution history</h2>
      <div className="card" style={{ marginBottom: 14 }}>
        <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
          <select value={filters.state}
            onChange={(e) => setFilters({ ...filters, state: e.target.value })}
            style={{ background: '#0f1115', color: 'var(--ink)',
              border: '1px solid var(--line)', borderRadius: 5, padding: '6px 8px' }}>
            <option value="">All states</option>
            {['PENDING_APPROVAL', 'APPROVED', 'EXECUTING', 'SUCCEEDED',
              'FAILED', 'REQUIRES_REVIEW', 'CANCELLED', 'REJECTED'].map((s2) => (
              <option key={s2} value={s2}>{s2}</option>
            ))}
          </select>
          <select value={filters.marketplace}
            onChange={(e) => setFilters({ ...filters, marketplace: e.target.value })}
            style={{ background: '#0f1115', color: 'var(--ink)',
              border: '1px solid var(--line)', borderRadius: 5, padding: '6px 8px' }}>
            <option value="">All marketplaces</option>
            {['amazon_in', 'flipkart', 'meesho', 'ondc'].map((m) => (
              <option key={m} value={m}>{m}</option>
            ))}
          </select>
          <input value={filters.q} placeholder="Search SKU, execution ID, error…"
            onChange={(e) => setFilters({ ...filters, q: e.target.value })}
            style={{ flex: 1, minWidth: 200, background: '#0f1115', color: 'var(--ink)',
              border: '1px solid var(--line)', borderRadius: 5, padding: '6px 10px' }} />
          <button className="ghost" onClick={() => setFilters({ state: '', marketplace: '', q: '' })}>
            Clear
          </button>
        </div>
      </div>

      {history.loading ? <Loading /> : history.error ? <ErrorState error={history.error} /> : (
        (history.data?.rows ?? []).length === 0
          ? <Empty>{history.data?.detail ?? 'No executions recorded yet.'}</Empty>
          : <DataTable
              columns={[
                { key: 'execution_id', label: 'Execution',
                  render: (r) => <span className="small muted">{String(r.execution_id).slice(0, 8)}…</span> },
                { key: 'operation', label: 'Operation' },
                { key: 'sku_code', label: 'SKU' },
                { key: 'marketplace', label: 'Channel' },
                { key: 'execution_state', label: 'State', render: (r) => <Tag value={r.execution_state} /> },
                { key: 'execution_mode', label: 'Mode', render: (r) => <Tag value={r.execution_mode} /> },
                { key: 'side_effect_occurred', label: 'Side effect',
                  render: (r) => r.side_effect_occurred
                    ? <span className="tag REJECTED">YES</span>
                    : <span className="tag ok">NO</span> },
                { key: 'reconciliation_state', label: 'Reconciliation',
                  render: (r) => <Tag value={r.reconciliation_state} /> },
                { key: 'external_reference', label: 'External ref',
                  render: (r) => r.external_reference ?? <span className="muted">—</span> },
                { key: 'created_at', label: 'Created',
                  render: (r) => <span className="small muted">{String(r.created_at).slice(0, 19).replace('T', ' ')}</span> },
              ]}
              rows={(history.data?.rows ?? []).map((r) => ({ ...r, id: r.idempotency_key }))}
            />
      )}

      <h2>Execution requests</h2>
      {q.data.executions.length === 0 ? <Empty>No execution requests.</Empty> : (
        <DataTable
          columns={[
            { key: 'operation', label: 'Operation' },
            { key: 'sku_code', label: 'SKU' },
            { key: 'marketplace', label: 'Channel' },
            { key: 'status', label: 'Status', render: (r) => <Tag value={r.status} /> },
            { key: 'payload_fingerprint', label: 'Payload',
              render: (r) => <span className="small muted">{r.payload_fingerprint.slice(0, 12)}…</span> },
            { key: 'approval', label: 'Approved by',
              render: (r) => r.approval ? r.approval.approved_by : <span className="muted">—</span> },
            { key: 'act', label: '', render: (r) => (
              <>
                {r.status === 'PENDING_APPROVAL' && (
                  <button disabled={busy}
                    onClick={() => act(r.id, 'approve', { actor: 'operator' })}>Approve</button>
                )}
                {r.status === 'APPROVED' && (
                  <button disabled={busy} className="ghost"
                    onClick={() => act(r.id, 'run', { mode: 'DRY_RUN' },
                      r.sku_code.replace('SKU-', ''))}>Dry run</button>
                )}
              </>) },
          ]}
          rows={q.data.executions.map((r) => ({ ...r, id: r.id }))}
        />
      )}

      {detail && (
        <>
          <h2>Result — <Tag value={detail.status} /> <Tag value={detail.mode} /></h2>
          <div className="cards">
            <Card label="External side effect"
              value={detail.side_effect_occurred ? 'YES' : 'NO'}
              note={detail.side_effect_occurred ? 'confirmed by adapter' : 'nothing was changed externally'} />
            <Card label="External reference" value={detail.external_reference ?? '—'} />
            <Card label="Retry class" value={detail.retry_class ?? '—'} />
          </div>
          <h2>Gate results</h2>
          <DataTable
            columns={[
              { key: 'gate', label: 'Gate' },
              { key: 'result', label: 'Result',
                render: (r) => <span className={`tag ${r.result === 'PASS' ? 'ok' : 'REJECTED'}`}>{r.result}</span> },
              { key: 'detail', label: 'Detail',
                render: (r) => <span className="small muted">{r.detail || '—'}</span> },
            ]}
            rows={detail.gates.map((g, i) => ({ ...g, id: i }))}
          />
        </>
      )}
    </Page>
  )
}
