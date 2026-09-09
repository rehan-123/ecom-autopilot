import React, { useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { useApi } from '../useApi'
import { inr, pct, postJson } from '../api'
import { Card, DataTable, Empty, ErrorState, Loading, Page, SafetyBanner, Tag } from '../components'

function view({ loading, error, reload, data }, render) {
  if (loading) return <Loading />
  if (error) return <ErrorState error={error} onRetry={reload} />
  return render(data)
}

export function Dashboard() {
  const q = useApi('/api/dashboard')
  return (
    <Page title="Dashboard" subtitle="Live state of the decision and operating engines.">
      <SafetyBanner />
      {view(q, (d) => (
        <>
          <div className="cards">
            <Card label="Distinct products" value={d.discovery.distinct_products}
              note={`${d.discovery.observations} observations`} />
            <Card label="Discovery feeds" value={`${d.discovery.feeds_available}/${d.discovery.feeds_total}`}
              note="sources with a legitimate route" />
            <Card label="Opportunities" value={d.opportunities.total}
              note={Object.entries(d.opportunities.by_verdict).map(([k, v]) => `${k} ${v}`).join(' · ')} />
            <Card label="Realized contribution" value={inr(d.operations.realized_contribution_inr)}
              note={`${d.operations.orders} settled orders`} />
            <Card label="Capital locked" value={inr(d.operations.capital_locked_inr)}
              note={`${d.operations.products} products`} />
            <Card label="Awaiting approval" value={d.approvals_pending}
              note="nothing executes without one" />
            <Card label="Ledger events" value={d.ledger.events}
              note={`hash chain ${d.ledger.chain_valid === null ? 'not initialised' : d.ledger.chain_valid ? 'valid' : 'BROKEN'}`} />
            <Card label="Low stock" value={d.operations.low_stock} note="below reorder point" />
          </div>
          <h2>Where to look next</h2>
          <p className="muted small">
            <Link to="/opportunities">Opportunities</Link> ranks candidates on worst-case
            profit. <Link to="/approvals">Approval queue</Link> holds everything the
            optimiser proposed. <Link to="/ledger">Audit ledger</Link> proves what happened.
          </p>
        </>
      ))}
    </Page>
  )
}

export function Discovery() {
  const feeds = useApi('/api/discovery/feeds')
  const products = useApi('/api/discovery/products')
  return (
    <Page title="Discovery" subtitle="Candidates from legitimate sources only.">
      {view(feeds, (d) => (
        <>
          <h2>Feed coverage</h2>
          <DataTable
            columns={[
              { key: 'source', label: 'Source' },
              { key: 'access', label: 'Access' },
              { key: 'available', label: 'Status', render: (r) => <Tag value={r.available ? 'ok' : 'REJECTED'} /> },
              { key: 'products', label: 'Products', align: 'right' },
              { key: 'unavailable_reason', label: 'Why not', render: (r) => <span className="muted small">{r.unavailable_reason ?? '—'}</span> },
            ]}
            rows={d.feeds.map((f, i) => ({ ...f, id: i }))}
          />
        </>
      ))}
      {view(products, (d) => (
        <>
          <h2>Distinct products</h2>
          <DataTable
            columns={[
              { key: 'canonical_title', label: 'Product', render: (r) => (
                <Link to={`/products/${r.slug}`}>{r.canonical_title}</Link>) },
              { key: 'size', label: 'Obs', align: 'right' },
              { key: 'sources', label: 'Sources', render: (r) => r.sources.join(', ') },
              { key: 'coverage_pct', label: 'Coverage', align: 'right', render: (r) => pct(r.coverage_pct) },
              { key: 'requires_review', label: 'Identity', render: (r) => <Tag value={r.requires_review ? 'REQUIRES_REVIEW' : 'ok'} /> },
            ]}
            rows={d.products.map((p) => ({ ...p, id: p.key }))}
          />
        </>
      ))}
    </Page>
  )
}

export function Suppliers() {
  const q = useApi('/api/suppliers')
  return (
    <Page title="Suppliers" subtitle="Ranked on true landed cost, not sticker price.">
      {view(q, (d) => d.products.length === 0 ? <Empty>No supplier matches.</Empty> : (
        d.products.map((p) => (
          <div key={p.group_key} style={{ marginBottom: 26 }}>
            <h2>{p.product}</h2>
            <DataTable
              columns={[
                { key: 'supplier_name', label: 'Supplier', render: (r) => (
                  <>{r.supplier_name}{r.supplier_name === p.selected && <strong className="muted"> ← selected</strong>}</>) },
                { key: 'unit_cost_inr', label: 'Unit', align: 'right', render: (r) => inr(r.unit_cost_inr) },
                { key: 'landed_cost_inr', label: 'Landed', align: 'right', render: (r) => inr(r.landed_cost_inr) },
                { key: 'moq_units', label: 'MOQ', align: 'right' },
                { key: 'tax_invoice', label: 'Tax invoice', render: (r) => <Tag value={r.tax_invoice ? 'ok' : 'REJECTED'} /> },
                { key: 'capital_required_inr', label: 'Capital', align: 'right', render: (r) => inr(r.capital_required_inr) },
                { key: 'total_score', label: 'Score', align: 'right', render: (r) => Number(r.total_score).toFixed(1) },
              ]}
              rows={p.options.map((o, i) => ({ ...o, id: i }))}
            />
          </div>
        ))
      ))}
    </Page>
  )
}

export function Opportunities() {
  const [filter, setFilter] = useState('')
  const q = useApi(`/api/opportunities${filter ? `?verdict=${filter}` : ''}`)
  return (
    <Page
      title="Opportunities"
      subtitle="Ranked on worst-case profit discounted by evidence coverage."
      actions={
        <div>
          {['', 'RECOMMENDED', 'BACKUP', 'REJECTED'].map((v) => (
            <button key={v || 'all'} className={filter === v ? '' : 'ghost'}
              style={{ marginLeft: 6 }} onClick={() => setFilter(v)}>{v || 'All'}</button>
          ))}
        </div>
      }
    >
      {view(q, (d) => (
        <DataTable
          columns={[
            { key: 'title', label: 'Product', render: (r) => (
              <Link to={`/products/${r.slug}`}>{r.title}</Link>) },
            { key: 'verdict', label: 'Verdict', render: (r) => <Tag value={r.verdict} /> },
            { key: 'expected_margin_pct', label: 'Expected', align: 'right', render: (r) => pct(r.expected_margin_pct) },
            { key: 'worst_margin_pct', label: 'Worst', align: 'right', render: (r) => pct(r.worst_margin_pct) },
            { key: 'landed_cost_inr', label: 'Landed', align: 'right', render: (r) => inr(r.landed_cost_inr) },
            { key: 'discovery_coverage_pct', label: 'Coverage', align: 'right', render: (r) => pct(r.discovery_coverage_pct) },
            { key: 'selected_supplier', label: 'Supplier' },
          ]}
          rows={d.opportunities.map((o) => ({ ...o, id: o.group_key }))}
          empty="No opportunities match this filter."
        />
      ))}
    </Page>
  )
}

export function ProductDetail() {
  const { key } = useParams()
  const q = useApi(`/api/opportunities/${encodeURIComponent(key)}`)
  return view(q, (d) => {
    const o = d.opportunity
    return (
      <Page title={o.title} subtitle={<>Verdict <Tag value={o.verdict} /> · confidence {o.confidence}</>}>
        <div className="cards">
          <Card label="Expected margin" value={pct(o.expected_margin_pct)} />
          <Card label="Worst case" value={pct(o.worst_margin_pct)} note="what the gate judges" />
          <Card label="Landed cost" value={inr(o.landed_cost_inr)} />
          <Card label="Coverage" value={pct(o.discovery_coverage_pct)} note="evidence completeness" />
        </div>

        {(o.blockers?.length > 0 || o.recoverable?.length > 0) && (
          <>
            <h2>Why this verdict</h2>
            {o.blockers?.length > 0 && <><div className="small muted">Blockers</div>
              <ul className="reasons">{o.blockers.map((b, i) => <li key={i}>{b}</li>)}</ul></>}
            {o.recoverable?.length > 0 && <><div className="small muted" style={{ marginTop: 8 }}>To strengthen</div>
              <ul className="reasons">{o.recoverable.map((b, i) => <li key={i}>{b}</li>)}</ul></>}
          </>
        )}

        <h2>Scenarios</h2>
        <DataTable
          columns={[
            { key: 'scenario', label: 'Scenario' },
            { key: 'landed_cost_inr', label: 'Landed', align: 'right', render: (r) => inr(r.landed_cost_inr) },
            { key: 'net_profit_inr', label: 'Net / order', align: 'right', render: (r) => inr(r.net_profit_inr) },
            { key: 'net_margin_pct', label: 'Margin', align: 'right', render: (r) => pct(r.net_margin_pct) },
            { key: 'min_profitable_price_inr', label: 'Min price', align: 'right', render: (r) => inr(r.min_profitable_price_inr) },
          ]}
          rows={Object.entries(d.scenarios).map(([k, v]) => ({ ...v, scenario: k, id: k }))}
          empty={
            (d.supplier_matches ?? []).length > 0
              ? 'Economics not computed: sourcing is unresolved, so there is no landed cost to price against. Confirm the supplier match below.'
              : 'Economics not computed: no supplier matched, so there is no acquisition cost to work from.'
          }
        />

        <h2>Supplier options</h2>
        {d.suppliers.length > 0 ? (
          <DataTable
            columns={[
              { key: 'supplier_name', label: 'Supplier' },
              { key: 'landed_cost_inr', label: 'Landed', align: 'right', render: (r) => inr(r.landed_cost_inr) },
              { key: 'moq_units', label: 'MOQ', align: 'right' },
              { key: 'tax_invoice', label: 'Invoice', render: (r) => <Tag value={r.tax_invoice ? 'ok' : 'REJECTED'} /> },
            ]}
            rows={d.suppliers.map((s, i) => ({ ...s, id: i }))}
          />
        ) : (d.supplier_matches ?? []).length > 0 ? (
          // A match below the actionable bar is not the same as no match. The
          // table previously reported "No supplier matched this product" while
          // the reason line said the match was only PROBABLE -- two
          // contradictory readings of the same engine state.
          <>
            <DataTable
              columns={[
                { key: 'supplier_name', label: 'Supplier' },
                { key: 'confidence', label: 'Match confidence', render: (r) => <Tag value={r.confidence} /> },
                { key: 'similarity', label: 'Similarity', align: 'right', render: (r) => pct(Number(r.similarity) * 100) },
                { key: 'unit_cost_inr', label: 'Unit', align: 'right', render: (r) => inr(r.unit_cost_inr) },
                { key: 'moq_units', label: 'MOQ', align: 'right' },
                { key: 'evidence', label: 'Matched on', render: (r) => (
                  <span className="small muted">{r.evidence?.[0]?.detail ?? '—'}</span>) },
              ]}
              rows={d.supplier_matches.map((m, i) => ({ ...m, id: i }))}
            />
            <p className="muted small" style={{ marginTop: 8 }}>
              Matched on title similarity only, which is below the bar for
              automatic selection — a shared product name is a hypothesis, not
              proof the supplier stocks this item. Confirm it and the match
              becomes usable; landed cost and economics are not computed until
              then.
            </p>
          </>
        ) : (
          <Empty>
            Nothing in the ingested supplier catalogues matched this product, so
            it cannot be sourced or costed.
          </Empty>
        )}
        <h2>Observations</h2>
        <DataTable
          columns={[
            { key: 'raw_title', label: 'Observed title' },
            { key: 'source', label: 'Source' },
            { key: 'coverage_pct', label: 'Coverage', align: 'right', render: (r) => pct(r.coverage_pct) },
          ]}
          rows={d.observations.map((m) => ({ ...m, id: m.id }))}
        />

        <h2>Decision trail</h2>
        <DataTable
          columns={[
            { key: 'rule_id', label: 'Rule' },
            { key: 'disposition', label: 'Disposition', render: (r) => <Tag value={r.disposition} /> },
            { key: 'reason', label: 'Reason', render: (r) => <span className="small">{r.reason}</span> },
          ]}
          rows={d.decisions.map((x, i) => ({ ...x, id: i }))}
          empty="No decisions recorded."
        />
      </Page>
    )
  })
}

export function Approvals() {
  const q = useApi('/api/approvals')
  const [busy, setBusy] = useState(null)
  const [notice, setNotice] = useState(null)

  async function act(id, kind) {
    setBusy(id)
    try {
      const result = kind === 'approve'
        ? await postJson(`/api/approvals/${id}/approve`, { approver: 'operator' })
        : await postJson(`/api/approvals/${id}/execute`, { dry_run: true })
      setNotice(kind === 'approve'
        ? `Approved. It can now be executed — in dry run.`
        : `Dry run: ${result.result?.detail ?? 'no external state changed'}`)
      q.reload()
    } catch (e) {
      setNotice(e.detail ?? e.message)
    } finally {
      setBusy(null)
    }
  }

  return (
    <Page title="Approval queue" subtitle="Nothing executes until a human approves it.">
      <SafetyBanner />
      {notice && <div className="banner">{notice}</div>}
      {view(q, (d) => (
        <DataTable
          columns={[
            { key: 'action', label: 'Action' },
            { key: 'sku_code', label: 'SKU' },
            { key: 'state', label: 'State', render: (r) => <Tag value={r.state} /> },
            { key: 'reason', label: 'Reason', render: (r) => <span className="small muted">{r.reason}</span> },
            { key: 'act', label: '', render: (r) => (
              r.state === 'RECOMMENDED'
                ? <button disabled={busy === r.id} onClick={() => act(r.id, 'approve')}>Approve</button>
                : r.state === 'APPROVED'
                  ? <button disabled={busy === r.id} onClick={() => act(r.id, 'execute')}>Execute (dry run)</button>
                  : <span className="muted small">—</span>
            ) },
          ]}
          rows={d.all.map((r) => ({ ...r, id: r.id }))}
          empty="Nothing proposed."
        />
      ))}
    </Page>
  )
}

export function Orders() {
  const q = useApi('/api/orders')
  return (
    <Page title="Orders" subtitle="Normalized from marketplace vocabularies; imports are idempotent.">
      {view(q, (d) => (
        <>
          <div className="cards">
            <Card label="Orders" value={d.orders.length} />
            <Card label="Duplicates ignored" value={d.duplicates_ignored} note="replayed imports change nothing" />
          </div>
          <h2>Recent</h2>
          <DataTable
            columns={[
              { key: 'marketplace_order_id', label: 'Order' },
              { key: 'marketplace', label: 'Channel' },
              { key: 'status', label: 'Status', render: (r) => <Tag value={r.status} /> },
              { key: 'units', label: 'Units', align: 'right' },
              { key: 'gross_inr', label: 'Gross', align: 'right', render: (r) => inr(r.gross_inr) },
              { key: 'ordered_at', label: 'Ordered', render: (r) => r.ordered_at.slice(0, 10) },
            ]}
            rows={d.orders.slice(0, 60).map((o) => ({ ...o, id: o.id }))}
          />
        </>
      ))}
    </Page>
  )
}

export function Inventory() {
  const q = useApi('/api/inventory')
  return (
    <Page title="Inventory" subtitle="Event-sourced positions; stock cannot go negative.">
      {view(q, (d) => (
        <DataTable
          columns={[
            { key: 'sku_code', label: 'SKU' },
            { key: 'on_hand', label: 'On hand', align: 'right' },
            { key: 'reserved', label: 'Reserved', align: 'right' },
            { key: 'available', label: 'Available', align: 'right' },
            { key: 'sold_units', label: 'Sold', align: 'right' },
            { key: 'capital_locked_inr', label: 'Capital', align: 'right', render: (r) => inr(r.capital_locked_inr) },
            { key: 'reorder', label: 'Reorder', render: (r) => r.reorder
              ? <Tag value={r.reorder.should_reorder ? 'REQUIRES_REVIEW' : 'ok'} /> : '—' },
            { key: 'qty', label: 'Suggested', align: 'right', render: (r) => r.reorder?.recommended_qty ?? '—' },
            { key: 'velocity', label: 'Velocity', render: (r) => r.reorder
              ? <span className="small muted">{r.reorder.velocity_per_day ?? '—'}/d {r.reorder.velocity_confidence}</span> : '—' },
          ]}
          rows={d.positions.map((p) => ({ ...p, id: p.sku_code }))}
        />
      ))}
    </Page>
  )
}

export function Analytics() {
  const q = useApi('/api/analytics')
  return (
    <Page title="Analytics" subtitle="Realized economics, ranked by contribution per unit of capital.">
      {view(q, (d) => (
        <>
          <h2>Portfolio</h2>
          <DataTable
            columns={[
              { key: 'sku_code', label: 'SKU' },
              { key: 'lifecycle', label: 'Stage', render: (r) => <Tag value={r.lifecycle} /> },
              { key: 'actual_contribution_inr', label: 'Contribution', align: 'right', render: (r) => inr(r.actual_contribution_inr) },
              { key: 'return_rate_pct', label: 'Returns', align: 'right', render: (r) => pct(r.return_rate_pct) },
              { key: 'capital_locked_inr', label: 'Capital', align: 'right', render: (r) => inr(r.capital_locked_inr) },
              { key: 'rank_score', label: 'Score', align: 'right', render: (r) => r.rank_score ? Number(r.rank_score).toFixed(1) : '—' },
              { key: 'evidence_sufficient', label: 'Evidence', render: (r) => <Tag value={r.evidence_sufficient ? 'ok' : 'VALIDATE'} /> },
            ]}
            rows={d.portfolio.map((p) => ({ ...p, id: p.sku_code }))}
          />

          <h2>Settlement reconciliation</h2>
          <div className="cards">
            {Object.entries(d.settlement.by_state).map(([k, v]) => (
              <Card key={k} label={k} value={v} />
            ))}
          </div>

          <h2>Recommendations</h2>
          {d.recommendations.length === 0 ? <Empty>No actions recommended.</Empty> : (
            d.recommendations.map((r, i) => (
              <div className="card" key={i} style={{ marginBottom: 10 }}>
                <div><Tag value={r.action} /> <strong>{r.sku_code}</strong>
                  <span className="muted small"> · confidence {r.confidence}</span></div>
                <div className="small" style={{ marginTop: 6 }}>{r.proposed_change}</div>
                <div className="small muted" style={{ marginTop: 4 }}>Upside: {r.expected_impact}</div>
                <div className="small muted">Downside: {r.downside}</div>
                <div className="small muted">Evidence: {r.evidence.join(' · ')}</div>
              </div>
            ))
          )}
        </>
      ))}
    </Page>
  )
}

export function Ledger() {
  const q = useApi('/api/ledger?limit=100')
  return (
    <Page title="Audit ledger" subtitle="Append-only, hash-chained, tamper-evident.">
      {view(q, (d) => (
        <>
          <div className="cards">
            <Card label="Events" value={d.total} />
            <Card label="Hash chain"
              value={d.integrity.valid === null ? 'not initialised' : d.integrity.valid === 'True' ? 'valid' : 'BROKEN'}
              note={d.integrity.reason ?? 'every event chains to its predecessor'} />
          </div>
          <h2>Recent events</h2>
          <DataTable
            columns={[
              { key: 'seq', label: 'Seq', align: 'right' },
              { key: 'timestamp', label: 'When', render: (r) => r.timestamp.slice(0, 19).replace('T', ' ') },
              { key: 'event_type', label: 'Type' },
              { key: 'actor', label: 'Actor' },
              { key: 'event_hash', label: 'Hash', render: (r) => <span className="small muted">{r.event_hash.slice(0, 16)}…</span> },
            ]}
            rows={d.events.map((e) => ({ ...e, id: e.id }))}
            empty="No ledger events. Run `python main.py demo` to seed some."
          />
        </>
      ))}
    </Page>
  )
}

export function Settings() {
  const q = useApi('/api/settings')
  return (
    <Page title="Settings" subtitle="Configuration, capability and credential status.">
      <SafetyBanner />
      {view(q, (d) => (
        <>
          <h2>Capabilities absent by design</h2>
          <div className="cards">
            {d.capabilities_absent.map((c) => <Card key={c} label={c} value="ABSENT" />)}
          </div>

          <h2>Marketplaces</h2>
          <DataTable
            columns={[
              { key: 'name', label: 'Channel' },
              { key: 'access', label: 'Access' },
              { key: 'credentials_configured', label: 'Credentials', render: (r) => <Tag value={r.credentials_configured ? 'ok' : 'REJECTED'} /> },
              { key: 'live_transport', label: 'Transport', render: (r) => <Tag value={r.live_transport ? 'ok' : 'NOT_SUPPORTED'} /> },
              { key: 'required_env_vars', label: 'Required env', render: (r) => <span className="small muted">{r.required_env_vars.join(', ') || '—'}</span> },
            ]}
            rows={d.marketplaces.map((m) => ({ ...m, id: m.name }))}
          />

          <h2>Reference data</h2>
          <DataTable
            columns={[
              { key: 'name', label: 'Document' },
              { key: 'state', label: 'State', render: (r) => (
                <span className={`tag ${r.state === 'VERIFIED' ? 'ok' : 'REQUIRES_REVIEW'}`}>{r.state}</span>) },
            ]}
            rows={Object.entries(d.reference_data).map(([name, state]) => ({ name, state, id: name }))}
          />
          <p className="muted small" style={{ marginTop: 10 }}>
            UNVERIFIED entries ship as shapes, not figures. The decision engine refuses
            to approve anything pinned to one.
          </p>
        </>
      ))}
    </Page>
  )
}
