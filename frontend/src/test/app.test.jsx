import { describe, expect, it, vi, beforeEach } from 'vitest'
import { act, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import React from 'react'
import { inr, pct } from '../api'
import { Dashboard, Opportunities, Settings } from '../pages'

// The frontend renders what the backend computed. These tests assert it
// renders real values faithfully and surfaces failures rather than showing an
// empty table that looks like "no results".

function mockApi(payloads) {
  global.fetch = vi.fn((url) => {
    const path = String(url)
    const match = Object.keys(payloads).find((k) => path.includes(k))
    if (!match) return Promise.resolve({ ok: false, status: 404, statusText: 'Not Found', json: () => Promise.resolve({ detail: 'nope' }) })
    return Promise.resolve({ ok: true, status: 200, json: () => Promise.resolve(payloads[match]) })
  })
}

beforeEach(() => vi.restoreAllMocks())

describe('money formatting', () => {
  it('formats rupees with Indian digit grouping', () => {
    expect(inr('1234567.89')).toBe('₹12,34,567.89')
    expect(inr('1299.00')).toBe('₹1,299.00')
    expect(inr('99')).toBe('₹99.00')
    expect(inr('-64.46')).toBe('-₹64.46')
  })

  it('never turns a decimal string into a float', () => {
    // The backend refuses floats for money; re-parsing here would undo that.
    const exact = '0.1'
    expect(inr(exact)).toContain('0.10')
    expect(inr(null)).toBe('—')
  })

  it('formats percentages and handles absent values', () => {
    expect(pct(14.83)).toBe('14.8%')
    expect(pct(null)).toBe('—')
  })
})

describe('Dashboard', () => {
  it('renders real engine numbers', async () => {
    mockApi({
      '/api/dashboard': {
        mode: 'PAPER', built_at: '2026-08-27T00:00:00Z',
        discovery: { observations: 7, distinct_products: 6, feeds_available: 2, feeds_total: 18 },
        opportunities: { total: 6, by_verdict: { REJECTED: 5, BACKUP: 1 } },
        operations: { orders: 47, realized_contribution_inr: '19426.00', capital_locked_inr: '14700.00', products: 2, low_stock: 1 },
        approvals_pending: 3, ledger: { events: 14, chain_valid: true },
      },
    })
    render(<MemoryRouter><Dashboard /></MemoryRouter>)
    await waitFor(() => expect(screen.getByText('₹19,426.00')).toBeInTheDocument())
    expect(screen.getByText('2/18')).toBeInTheDocument()
    expect(screen.getByText(/PAPER MODE/)).toBeInTheDocument()
  })

  it('shows an error instead of an empty dashboard when the API fails', async () => {
    global.fetch = vi.fn(() => Promise.resolve({
      ok: false, status: 503, statusText: 'Service Unavailable',
      json: () => Promise.resolve({ detail: 'reference data not found' }),
    }))
    render(<MemoryRouter><Dashboard /></MemoryRouter>)
    await waitFor(() => expect(screen.getByText(/reference data not found/)).toBeInTheDocument())
  })
})

describe('Opportunities', () => {
  it('shows verdict, worst case and coverage for each candidate', async () => {
    mockApi({
      '/api/opportunities': {
        opportunities: [{
          group_key: 'KX880', title: 'Kemei KX-880 Trimmer', verdict: 'BACKUP',
          confidence: 'MEDIUM', expected_margin_pct: '44.5', worst_margin_pct: '37.9',
          landed_cost_inr: '203.64', discovery_coverage_pct: '61',
          selected_supplier: 'Shenzhen Kemei Direct', requires_approval: true,
          reasons: [], blockers: [], recoverable: [],
        }],
      },
    })
    render(<MemoryRouter><Opportunities /></MemoryRouter>)
    await waitFor(() => expect(screen.getByText('Kemei KX-880 Trimmer')).toBeInTheDocument())
    // A filter button also reads BACKUP; assert on the row's tag specifically.
    expect(document.querySelector('td .tag.BACKUP')).not.toBeNull()
    expect(screen.getByText('37.9%')).toBeInTheDocument()
    expect(screen.getByText('₹203.64')).toBeInTheDocument()
  })
})

describe('Settings', () => {
  it('states that money-moving capabilities are absent, not disabled', async () => {
    mockApi({
      '/api/settings': {
        mode: 'PAPER', dry_run_default: true, execution_requires_approval: true,
        capabilities_absent: ['purchasing', 'payment processing', 'money transfer'],
        reference_data: { 'tax_table:template-0': 'UNVERIFIED' },
        marketplaces: [{ name: 'amazon_in', access: 'SELLER_API', capabilities: [],
          credentials_configured: false, required_env_vars: ['AMAZON_SP_CLIENT_ID'],
          live_transport: false, notes: '' }],
        credentials: {}, credential_report: '',
      },
    })
    render(<MemoryRouter><Settings /></MemoryRouter>)
    await waitFor(() => expect(screen.getAllByText('ABSENT').length).toBe(3))
    expect(screen.getByText('purchasing')).toBeInTheDocument()
    expect(screen.getByText('UNVERIFIED')).toBeInTheDocument()
  })
})

describe('Research screen', () => {
  it('labels AI suggestions distinctly from sourced facts', async () => {
    const { default: Research } = await import('../pages/Research')
    global.fetch = vi.fn(() => Promise.resolve({
      ok: true, status: 200,
      json: () => Promise.resolve({
        id: 'run-1', request: 'hair trimmers', status: 'COMPLETE',
        ai_available: true, ai_error: null, warnings: [],
        sources: [{ source: 'manual_research', available: true }, { source: 'meesho', available: false }],
        disclaimer: 'No price here was generated by a model.',
        candidates: [{
          id: 'c1', name: 'Rechargeable Hair Trimmer', category: 'personal_care_electrical',
          rationale: 'steady repeat category', search_terms: ['trimmer'],
          ai_confidence: 'medium', is_ai_suggestion: true, facts_are_ai_generated: false,
          provenance_summary: { sourced: ['observed_price_inr'], unavailable: ['review_count'], evidence_pct: '12.0' },
          claims: [
            { field: 'observed_price_inr', value: '1349', provenance: 'SOURCED',
              reference: 'manual://obs/1', observed_at: '2026-08-27T00:00:00Z', confidence: 'medium', note: null },
            { field: 'review_count', value: null, provenance: 'UNAVAILABLE',
              reference: null, observed_at: null, confidence: 'low', note: 'no legitimate API' },
          ],
        }],
        export_preview: { 'observations.csv': 'title,price\nRechargeable Hair Trimmer,1349\n',
          'suppliers.csv': '# not written: supplier facts come from an ingested catalogue' },
      }),
    }))

    const { default: React2 } = await import('react')
    render(<MemoryRouter><Research /></MemoryRouter>)
    await act(async () => {
      screen.getByText('Research').click()
    })

    await waitFor(() => expect(screen.getByText('Rechargeable Hair Trimmer')).toBeInTheDocument())
    // Origin is always shown as an AI suggestion.
    expect(screen.getAllByText('REQUIRES_REVIEW').length).toBeGreaterThan(0)
    // Sourced vs unavailable counts are both surfaced.
    expect(screen.getByText('12.0%')).toBeInTheDocument()
  })

  it('surfaces degraded mode rather than pretending it worked', async () => {
    const { default: Research } = await import('../pages/Research')
    global.fetch = vi.fn(() => Promise.resolve({
      ok: true, status: 200,
      json: () => Promise.resolve({
        id: 'run-2', request: 'x', status: 'DEGRADED', ai_available: false,
        ai_error: 'no LLM client configured', candidates: [],
        warnings: ['no research model is configured'], sources: [],
        disclaimer: '', export_preview: { 'observations.csv': '', 'suppliers.csv': '' },
      }),
    }))
    render(<MemoryRouter><Research /></MemoryRouter>)
    await act(async () => {
      screen.getByText('Research').click()
    })
    await waitFor(() => expect(screen.getByText(/degraded mode/i)).toBeInTheDocument())
    // The warning appears in both the status banner and the warnings list,
    // which is intentional: the operator should not have to hunt for it.
    expect(screen.getAllByText(/no research model is configured/).length).toBeGreaterThan(0)
  })
})
