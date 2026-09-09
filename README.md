# ecom-autopilot

A decision engine and controlled operating layer for Indian marketplace
selling.

**Part 1** works out whether a product is worth selling: true landed cost,
three-scenario economics, market price reality, compliance, and a human
approval. **Part 2** turns an approved decision into listings, inventory,
orders, settlements and measured profit — then feeds the measurements back so
the next decision rests on evidence instead of assumption.

Verified in a clean virtualenv installing only from `pyproject.toml`:

| Gate | Result |
|---|---|
| `python -m compileall` | pass |
| `pytest` | **1029 passed** |
| `ruff check .` | clean |
| `ruff format --check .` | clean |
| `python tools/mutation_check.py` | 82 mutations, 81 caught, 1 inert control survived |

---

## Running the full application

Two processes: the API and the frontend dev server.

```bash
# 1. Backend engines + HTTP API
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e '.[dev]'
python main.py init-db && python main.py demo    # seeds the audit ledger
python -m uvicorn api.app:app --reload --port 8000

# 2. Frontend, in a second terminal
cd frontend
npm install
npm run dev
```

| Service | URL |
|---|---|
| API | http://127.0.0.1:8000 |
| API docs (OpenAPI) | http://127.0.0.1:8000/docs |
| Frontend | http://127.0.0.1:5173 |

The Vite dev server proxies `/api` to port 8000, so the browser stays on one
origin. For a production-style check: `npm run build && npm run preview`
(http://127.0.0.1:4173) with `VITE_API_BASE=http://127.0.0.1:8000`.

Tests: `pytest` for the engines and API, `cd frontend && npm test` for the UI.

### AI Research Assistant

`research/` adds a research screen at `/research`. The organising rule is a
hard separation between what a model may suggest and what may be treated as
fact:

| Provenance | Meaning |
|---|---|
| `SOURCED` | a legitimate source supplied it, with a reference and a date |
| `AI_INFERENCE` | a model suggested it — never a number, never evidence |
| `UNAVAILABLE` | nothing could supply it, recorded rather than filled |

A language model is good at suggesting *what to investigate*. It cannot know
what a trimmer costs in Delhi this week, and any number it produces is a
fabrication that becomes indistinguishable from a measurement once it lands in
a CSV. So the enforcement runs at three levels:

- **The prompt validator** rejects any response containing a price, cost, MOQ,
  GST, HSN, weight, review count, rank, demand, competitor count or supplier —
  by allowlist, so a new fabrication is refused by default.
- **The `Claim` type** raises if `AI_INFERENCE` is attached to a factual field,
  and if a `SOURCED` claim has no reference.
- **The export path** writes only `SOURCED` values into `observations.csv`.
  Everything else is written blank, which the existing ingester already reads
  as UNKNOWN.

`suppliers.csv` is never written from research at all: an acquisition cost has
exactly one legitimate origin, which is a catalogue you ingested.

**Degradation is normal.** With no LLM configured the assistant falls back to
terms from your own request and says so. With no live sources every factual
field returns `UNAVAILABLE` — the honest answer for most Indian marketplaces,
not a failure. An empty result is explained rather than left to read as "no
promising products exist".

#### Enabling live research with OpenAI

Set one environment variable and restart the API:

```bash
export OPENAI_API_KEY=sk-...          # Windows: set OPENAI_API_KEY=sk-...
python -m uvicorn api.app:app --reload --port 8000
```

Optional overrides: `OPENAI_MODEL` (default `gpt-4o-mini`), `OPENAI_BASE_URL`
(default `https://api.openai.com/v1/chat/completions`).

`llm/openai_client.py` implements the project's existing `LLMClient` protocol
over stdlib `urllib` — no third-party HTTP client was added, matching how the
marketplace transports in `part2/transport.py` are built. The key is read from
the environment inside the request handler on every call, never cached on an
object, never logged, and never present in any API response — a test asserts
this. Without the variable set, `default_researcher()` falls back to
`NullLLMClient`, which is the same degraded-but-working path any other
transport failure takes; there is no special case for "no key" versus "key
rejected" versus "network unreachable".

The anti-fabrication validator in `llm/services.py` applies identically
regardless of transport: a real OpenAI response containing a price is rejected
by the same allowlist that rejects one from the recorded test client, because
the validator has no visibility into which transport produced the text.

**Research decides nothing.** "Send to Analysis" adds candidates to discovery;
the RECOMMENDED / BACKUP / REJECTED verdict comes from the existing matching,
economics, compliance and opportunity engines, and human approval is still
required before anything executes.

### Phase 3 — marketplace adapters and readiness

**Phase 3 performs no live marketplace execution.** It decides whether a
product *could* be listed and says what is missing when it could not. Nothing
publishes a listing, places an order, syncs inventory, contacts a customer or
moves money.

`marketplace/` sits on top of the Part 2 adapters rather than replacing them:
capability declarations and credential detection come from
`part2/marketplace.py`, so there is one list of what each channel can do.

Three "not available" states are kept distinct, because collapsing them makes
a dashboard lie:

| State | Meaning |
|---|---|
| `UNSUPPORTED` | the marketplace cannot do this, ever — a credential changes nothing |
| `NOT_CONFIGURED` | it could, but no credentials exist |
| `NOT_READY` | the channel is fine; this product is not fit to list |

Write capabilities return `NOT_ENABLED_FOR_PHASE_3` **even where an adapter
and valid credentials exist**. `POST /api/listings/{slug}/publish` exists so a
client receives a structured refusal rather than a 404; it always answers
`published: false`.

**Provenance survives the crossing.** A `ListingDraft` carries each factual
field with the provenance it had in Phase 2. `AI_DRAFT` copy is legitimate
draft text and can never satisfy a field that requires evidence — a listing
fact must be `SOURCED`. Compliance state, economics and supplier confirmation
are read from the opportunity, never recomputed.

Readiness is refused when compliance is `FAIL`/`UNKNOWN`/`REQUIRES_HUMAN_REVIEW`,
when economics are unavailable, or when no supplier is confirmed. A probable
supplier match is reported with its confidence and never as a source of supply.

**Adding a marketplace:** declare its `Capability` set and credential
variables on a `BaseAdapter` in `part2/marketplace.py`; the registry, capability
reporting, validation and UI pick it up with no further change.

Endpoints: `GET /api/marketplaces/registry`, `GET /api/listings/{slug}/preview`,
`GET /api/listings/{slug}/validate`, `POST /api/listings/{slug}/publish` (refuses).

### Phase 4 — controlled execution

**No external side effect occurs without explicit human approval plus twelve
gates passing immediately beforehand.** Dry run is the default and the only
mode enabled unless `ECOM_AUTOPILOT_LIVE_EXECUTION` is explicitly set.

**Approval binds to one exact operation** — its type, SKU, marketplace and a
SHA-256 fingerprint of the payload. Changing any of them invalidates it. An
approval is authorisation for an action, not a standing permission for a
product.

**The twelve gates**, all re-evaluated at attempt time: approval exists ·
approval matches this operation · payload unchanged · status is APPROVED ·
capability SUPPORTED · adapter configured · live mode authorised · compliance
cleared · verdict executable · economics available · supplier confirmed ·
inventory valid · idempotency available · not cancelled. Gates are evaluated
in full rather than short-circuiting, so an operator sees every reason at once.

**Idempotency** is content-derived — operation, marketplace, SKU and payload
fingerprint — never a timestamp or random value, so a resubmission is
recognised as the same intended effect.

**Success is never fabricated.** A dry run reports `DRY_RUN` with
`side_effect_occurred: false` and never calls an adapter at all. Live success
is claimed only when the adapter confirms it. A lost response becomes
`UNKNOWN_OUTCOME` and `REQUIRES_REVIEW`, never an automatic retry — the
request may already have taken effect.

**Not implemented, by design:** payment, bank or wallet transfer, supplier
payment, customer refund, order fulfilment. No adapter for any of them exists.

Endpoints: `POST /api/executions`, `/approve`, `/reject`, `/cancel`, `/run`;
`GET /api/executions`, `/{id}`, `/{id}/audit`.

### Phase 5 — durable execution safety

**SQLite arbitrates idempotency, not any process.** Phase 4 kept it in a
per-process dictionary: correct for one worker, worthless for two. A
`UNIQUE` index on `idempotency_key` in `execution_ledger` now makes the claim
atomic — a worker inserts inside `BEGIN IMMEDIATE`, exactly one insert
succeeds, and the loser reads back the winner's row.

**Proved with real subprocesses**, not threads. Six processes, one shared
database, one shared key → exactly one `WON`. Threading would prove nothing
here, since threads share a store and processes do not.

**Crash recovery never re-executes.** A row left `EXECUTING` by a dead process
becomes `REQUIRES_REVIEW` with outcome `UNKNOWN_OUTCOME`. Re-running risks a
duplicate; assuming success risks a phantom. Both are worse than asking a human.

**Only `CONFIRMED_FAILURE` permits a retry.** `REQUIRES_REVIEW`,
`RECONCILING` and `CONFIRMED_SUCCESS` all refuse, because only a confirmed
failure establishes that nothing landed.

### Phase 6 — health, readiness and operations

`GET /api/health` reports per-component status (database, marketplace
adapters, execution) rather than "the process is running".

`GET /api/readiness` keeps **application health separate from marketplace
readiness** — the app can be perfectly healthy with no marketplace configured.
Live execution is reported `NOT_READY` with its blockers enumerated.

### What the web layer is

A thin HTTP surface over the existing engines. `api/service.py` assembles
domain state by calling the same functions the CLI calls; `api/app.py` exposes
it. No price, margin, verdict or state transition is computed in the web layer
— if a number appears in the UI, an engine produced it.

Eleven screens: Dashboard, Discovery, Suppliers, Opportunities, Product
Detail, Approval Queue, Orders, Inventory, Analytics, Audit Ledger, Settings.

**The safety model is unchanged and enforced at the HTTP layer too.** The only
mutating endpoints create an approval proposal or approve one. Executing
requires an APPROVED record — the API returns 409 otherwise, because the gate
lives in `part2.execution` and HTTP cannot route around it. Every execution
defaults to dry run. Tests assert no endpoint path can purchase, pay, publish
or transfer.

---

## What this system will and will not do

**AI is advisory.** A model may suggest product candidates and search terms and
may draft listing copy. It never establishes a supplier fact, a market price, a
demand figure, an economic input or a compliance status. Those come from
ingested evidence or stay UNAVAILABLE.

**Missing evidence never becomes positive evidence.** No HSN means compliance is
UNKNOWN, not PASS. No economics means no profitability claim. No confirmed
supplier means no recommendation. A probable supplier match is never a
confirmed one.

**Human approval is mandatory** for anything that could touch the outside
world, and an approval is bound to one exact operation, SKU, marketplace and
payload fingerprint.

**Dry run is the default**, and the only mode enabled unless the environment
explicitly says otherwise.

**Unsupported marketplace capabilities stay unsupported.** Credentials never
convert an UNSUPPORTED capability into a SUPPORTED one.

**Absent by design:** payment, bank or wallet transfer, supplier payment,
customer refund, order fulfilment, and live marketplace publishing.

---

## Operations

### Backup and restore

```bash
python main.py backup                      # timestamped copy beside the database
python main.py backup path/to/target.db    # explicit destination
```

Uses SQLite's own backup API rather than a file copy, so a backup taken while
the application is running is internally consistent instead of a torn
snapshot. It reads the live database and refuses to overwrite any existing
file.

**Restore is manual by design.** Stop the application, move the backup over
`data/ecom_autopilot.db`, restart. An automatic restore is one mistyped
argument away from destroying the ledger it exists to protect.

### Export

```
GET /api/export/executions      durable execution ledger
GET /api/export/audit           execution lifecycle audit trail
GET /api/export/opportunities   ranked opportunities with verdicts
GET /api/export/ledger          hash-chained event ledger
```

All bounded and paginated. IDs, states and provenance are preserved. None of
these records has a field that could hold a credential, and a test plants a
secret in the environment to prove none appears in the output.

### Health and readiness

`GET /api/health` reports per-component status. `GET /api/readiness` keeps
application health separate from marketplace readiness and enumerates the
blockers preventing live execution.

---

## Release validation

```bash
python tools/release_gate.py                 # A→Z: safety, security, regression
python tools/release_gate.py --skip-frontend # no Node toolchain
python tools/validate.py --fresh             # code correctness in a clean venv
```

The two are complementary. `validate.py` proves the code is correct — tests,
lint, format, every CLI command. `release_gate.py` proves the *release* is
safe: that each phase's safety properties still hold, that no credential can
escape, that dependencies carry no known vulnerability, and that this README
does not claim capabilities the system lacks. A green test suite and a safe
release are different claims.

---

## Getting started

This project is a directory tree, not a set of loose files. If files are
missing, `pytest` stops immediately with a preflight message naming them
rather than an unrelated import error.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e '.[dev]'

python tools/validate.py           # every gate, plus a CLI smoke test
python tools/validate.py --fresh   # same, in a throwaway venv built from the manifest
python tools/validate.py --quick   # skip mutation testing
./validate.sh                      # thin wrapper, same thing, POSIX shells
```

The validator is Python rather than shell so it runs identically on Windows
and macOS. It builds its throwaway environment in the system temp directory,
**outside the project tree**, so no integrity scanner has to be taught to
ignore it — and cleanup clears Windows' read-only bit on venv files, retries,
and downgrades a stubborn leftover to a warning rather than masking the real
validation result with a secondary error.

Each of the 16 CLI commands is exercised the way it is meant to be used.
`ingest` needs a supplier file, so the validator writes a minimal one — aliased
headers, a rupee sign, a quoted thousands separator — into a temporary
directory, ingests it, checks the parse, and removes it. Skipping the command
because it takes an argument would hide exactly the class of defect the step
exists to catch. Every gate is checked on its *output*, not just its exit code:
a command that prints an error and returns 0 still fails.

### Commands

```
Discovery — find what to sell
  discover             collect candidates from every legitimate feed
  trends               search-interest evidence, with its limits stated
  match                how observations grouped; what needs a human
  suppliers            supplier options ranked on true landed cost
  opportunities        every candidate ranked, with rejections
  rank                 the ranked list only
  recommend            the shortlist a human is asked to approve

Part 1 — decide what to sell
  analyze              run the full decision pipeline over a demo catalogue
  ingest --source f    read a supplier CSV / XLSX / PDF, report what was read
  sources              which discovery sources have a legitimate access route
  reference            verification state of all reference data

Part 2 — operate what was approved
  marketplaces         what each channel adapter can legitimately do
  publish [--dry-run]  the publication gate, in dry run by default
  inventory            stock positions and replenishment advice
  reconcile            settlements matched against orders
  portfolio            products ranked by contribution per unit of capital
  optimize             recommendations, with evidence and downside

Foundation
  init-db · demo · status · verify-ledger · rebuild-projections
```

Every external action defaults to `--dry-run`. Nothing in this repository can
reach a marketplace without both credentials and a transport you supply.

---

## The whole pipeline

```
DISCOVERY SIGNALS          discovery/     18 sources, DATA_UNAVAILABLE where honest
        ↓
SUPPLIER INGESTION         sources/       CSV · XLSX · PDF, per-field provenance
        ↓
IDENTITY RESOLUTION        llm/           Claude judges; findings, not decisions
        ↓
SUPPLIER SELECTION         part2/         true landed cost, not sticker price
        ↓
LANDED COST                econ/          + freight + QC + MOQ carry − ITC
        ↓
ECONOMICS  BEST/EXPECTED/WORST            versioned scenario deltas
        ↓
PROFITABILITY GATE                        gates on the downside
        ↓
MARKET PRICE + HEADROOM    signals/       the one gate that tests arithmetic
        ↓                                 against the world
OPPORTUNITY SCORE                         value AND coverage, never blended
        ↓
RISK → COMPLIANCE          compliance/    rules-as-data; UNKNOWN ≠ PASS
        ↓
CLAUDE COMMERCIAL JUDGE    llm/
        ↓
DETERMINISTIC RULES        decisions/     findings → dispositions → state
        ↓
HUMAN APPROVAL  →  LISTING_READY          ── Part 1 ends here ──
        ↓
PUBLICATION GATE           part2/publisher    re-checks approval, compliance,
        ↓                                     and economics at publish time
MARKETPLACE ADAPTERS       part2/marketplace  capabilities declared, dry-run
        ↓
INVENTORY                  part2/inventory    event-sourced, cannot go negative
        ↓
ORDERS → RETURNS / RTO     part2/operations   idempotent imports
        ↓
SETTLEMENT RECONCILIATION                     unmatched money stays visible
        ↓
REALIZED PROFIT                               settled money, not order value
        ↓
PREDICTION ERROR                              predicted vs actual margin
        ↓
PORTFOLIO + LIFECYCLE      part2/intelligence TEST→VALIDATE→SCALE→MAINTAIN
        ↓                                     →REDUCE→KILL
OPTIMISATION RECOMMENDATIONS                  evidence, impact, downside
        ↓
EXECUTION GATE             part2/execution    RECOMMENDED→APPROVED→…→EXECUTED
```

---

## Repository structure

```
ecom-autopilot-part2/
├── pyproject.toml       dependencies + ruff + pytest, single manifest
├── conftest.py          preflight: diagnoses an incomplete checkout
├── validate.sh          all five validation gates
├── main.py              CLI
│
│ ── Part 1: decide ──────────────────────────────────────────────
├── sources/    ingest.py     CSV / XLSX / PDF → rows with provenance
├── discovery/  registry.py   18 source adapters; no scraping
├── reference/  loader.py     versioned data, staleness, version pinning
├── econ/       money.py landed_cost.py calculator.py
├── signals/    market.py     signals, price reference, score, risk
├── compliance/ engine.py     rules-as-data evaluation
├── llm/        services.py   the four Claude call sites
├── decisions/  rules.py pipeline.py
├── catalog/    enums.py schemas.py repository.py
├── database/   types.py tables.py connection.py migrations.py
├── ledger/     events.py projections.py replay.py
│
│ ── Discovery & ranking ─────────────────────────────────────────
├── sourcing/model.py         normalized discovery model, coverage
├── sourcing/matching.py      duplicate detection, supplier matching
├── sourcing/feeds.py         legitimate feeds; DATA_UNAVAILABLE elsewhere
├── sourcing/opportunities.py RECOMMENDED / BACKUP / REJECTED
│
│ ── Part 2: operate ─────────────────────────────────────────────
├── part2/execution.py     approval gate, idempotency, failure taxonomy
├── part2/marketplace.py   adapter contract, 4 channels, dry-run, drift
├── part2/inventory.py     movements, reservations, reorder policy
├── part2/operations.py    orders, returns/RTO, settlement, realized profit
├── part2/intelligence.py  pricing, supplier ranking, portfolio, optimisation
├── part2/publisher.py     publication prerequisites
├── part2/transport.py     HTTP (stdlib), rate limiting, retries, pagination
├── part2/auth.py          credential specs, OAuth token providers
├── part2/normalize.py     marketplace payloads -> domain objects
│
├── config/settings.yaml
├── data/reference/      versioned reference data
├── tests/               1029 tests across 22 modules
└── tools/  mutation_check.py, project_files.py
```

---

## The discovery layer

```
LEGITIMATE FEEDS          manual observations · seller exports · Trends CSV
      ↓                   everything else: DATA_UNAVAILABLE, visibly
NORMALIZE                 titles, model codes, GTINs, brands
      ↓
GROUP DUPLICATES          GTIN > model code > similarity; similarity alone
      ↓                   never merges, it only raises a question
MATCH SUPPLIERS           from ingested catalogues only — never from a
      ↓                   discovery source
RANK SUPPLIERS            existing Part 2 engine: true landed cost, MOQ
      ↓                   capital, input tax credit
PART 1 PIPELINE           existing economics, gates, compliance, scoring
      ↓
CLASSIFY                  RECOMMENDED · BACKUP · REJECTED
      ↓
HUMAN APPROVAL            nothing executes without it
```

No money logic lives in `sourcing/`. Landed cost, scenarios, the profitability
gate, headroom, compliance and scoring are all the existing engines, so there
is one definition of what a product costs.

**The three verdicts mean different things.** `RECOMMENDED` cleared every gate
with enough evidence behind it. `BACKUP` would clear the gates but rests on
thin evidence or something recoverable — missing comparables, an unconfirmed
supplier match. Those are research tasks, not dead ends. `REJECTED` failed a
gate that matters, and the report says which and by how much.

**Ranking is on the worst case**, discounted by evidence coverage. Ranking on
expected profit puts the most optimistic assumptions at the top of the list,
which is the wrong order for deciding where capital goes.

**Identity is decided by evidence, not similarity.** A shared GTIN is
conclusive; a shared manufacturer model code is near-conclusive; token overlap
is a hypothesis that routes to review. Merging two different products silently
averages their prices and produces economics for something that does not
exist. Claude may be consulted on an uncertain match — its answer is appended
as a *finding* and never changes the confidence band.

**Everything runs offline.** Fixtures under `data/fixtures/discovery/` supply
observations, a Trends export and a supplier catalogue, so the full chain works
with no credentials and no network.

---

## Nine ideas the design turns on

### 1. A discovery source is not a supplier

A marketplace can suggest a product is moving. It cannot be where you buy it:
prices sit at or near landed cost and it issues no tax invoice, so a purchase
yields no input tax credit and you pay GST twice.

`discovery/registry.py` records 18 sources with the access route each genuinely
offers. Sources with no legitimate route are registered with `access=NONE` —
visible, and yielding `DATA_UNAVAILABLE` rather than being quietly filled by a
scraper. Sourcing runs manufacturer → authorised wholesaler → distributor →
B2B supplier → importer.

### 2. Landed cost is not supplier cost

```
supplier cost + freight + packaging + QC + defect provision
              + MOQ carrying cost − input tax credit
```

Running the same trimmer both ways:

| | supplier | ITC | landed | net/order | margin |
|---|---|---|---|---|---|
| retail source, no tax invoice | ₹399 | ₹0 | ₹447 | **−₹64** | −8.1% |
| wholesaler with tax invoice | ₹280 | ₹43 | ₹275 | **+₹118** | 14.8% |

₹182 per order from sourcing alone, on identical software.

### 3. The gate considers the downside

Three scenarios as versioned deltas. Policy requires the worst case to hold,
and by default it does: a bad SKU costs inventory, capital, shelf-time and
account metrics, while a good one only makes money. Every failure names its
gate and its gap — *"reduce landed cost by ₹163 or raise price to ₹1,493"* is
a sourcing instruction.

### 4. Missing data stays missing

Roughly half the demand signals for Indian marketplaces have no legitimate
source. A zero is a claim, an average is a claim, `DATA_UNAVAILABLE` is the
truth. The score therefore carries its own coverage, and below the floor no
ranking value is emitted at all — `78/100 (85% coverage)`, never a bare number.

### 5. Claude produces findings; code produces decisions

Four call sites — Extractor, Judge Identity, Judge Commercial, Writer. Each
validates against a schema and records a full invocation row whether it
succeeded or not. A malformed response is treated as missing evidence; there
is no path from bad JSON to an approval. Transport is injected and no HTTP
client is bundled.

### 6. Nothing invents a rate

`data/reference/` ships **shapes, not figures**. Rate cards, tax tables and the
compliance ruleset report `UNVERIFIED`, and the decision engine refuses to
approve anything pinned to unverified or stale reference data. Compliance
additionally clears nothing against a stale rulebook: a book that has not been
checked may be *missing* requirements.

### 7. UNKNOWN is never success

Part 2 talks to systems outside this process, which adds three failure modes
Part 1 never had. A timeout after the request was sent means the marketplace
may or may not have acted. Recording that as failure invites a retry that
duplicates; recording it as success invites drift. It is recorded as `UNKNOWN`
and routed to a human.

Outcomes are `OK` · `NOT_SUPPORTED` · `RETRYABLE_FAILURE` · `PERMANENT_FAILURE`
· `UNKNOWN` · `DRY_RUN`. Only `OK` counts as success.

### 8. Every external write is idempotent

Each carries a key derived from the *intent* — SKU, marketplace, target price
— never from the attempt. A replay of the same intent returns the first result
without touching the transport. A non-idempotent write that fails retryably is
never retried, because the first attempt may have landed; the outcome becomes
`UNKNOWN` instead.

### 9. Recommendation is not execution

`RECOMMENDED → APPROVED → QUEUED → EXECUTING → EXECUTED / FAILED /
REQUIRES_REVIEW`. `QUEUED` is unreachable from `RECOMMENDED`; the only route
runs through a human approval. Nothing reaches a marketplace because a ranking
function liked it.

---

## Part 2 operating behaviour

**Inventory** is event-sourced. Stock cannot go negative and you cannot
reserve what you do not have — both refused at the point of recording, not
detected later. A new product gets a small controlled test (default 12 orders)
rather than a forecast-sized order, and every cap that bit is reported.

**Orders** normalise marketplace vocabularies onto internal statuses. An
unmapped status raises rather than defaulting: a marketplace changing its
vocabulary is a human's problem, not a guess. Imports are keyed on the
marketplace's own identifier, so replayed webhooks change nothing.

**Settlement** reconciles to `MATCHED` / `PARTIAL` / `UNMATCHED` /
`REQUIRES_REVIEW`. Money that ties to no order is never absorbed into profit —
the discrepancy is usually where the real problem is.

**Realized economics** are built from *settled* money, not order value. The gap
between them is fees, taxes and adjustments, and it is where predicted margin
usually goes wrong. Prediction error against Part 1's forecast is the system's
most valuable learning signal.

**Portfolio ranking** is by risk-adjusted contribution per unit of capital, not
revenue. Revenue ranks the product turning over the most money, which in a
capital-constrained operation is frequently the one that should be killed.
Lifecycle calls require a minimum sample: 20 orders for a decision, 30 before a
kill.

---

## Marketplace adapters — what is and is not implemented

| Channel | Access model | Transport | Status |
|---|---|---|---|
| Amazon India | Selling Partner API | implemented | needs credentials; **never run live** |
| Flipkart | Seller API | implemented | needs credentials; **never run live** |
| ONDC | Network participant | partial | needs an injected Ed25519 signer |
| Meesho | Panel export only | n/a | listing/price writes `NOT_SUPPORTED` by design |

**What is implemented.** Real HTTP over the standard library — no third-party
client, so the repository still declares none. OAuth2 token acquisition and
caching (Login with Amazon's refresh-token grant, Flipkart's client-credentials
grant), refreshed under a lock so a burst triggers one refresh rather than a
hundred. Per-operation token-bucket rate limiting seeded from published limits
and tightened by response headers. Bounded retries with exponential backoff
honouring `Retry-After`. Cursor pagination with loop and ceiling detection.
Response normalization into the existing `Order` and `SettlementLine` types.

**What has not happened.** None of it has run against a live endpoint. I have
no credentials for these marketplaces and could not obtain any. Every path is
tested against a fake transport that records what *would* have been sent, so
the request construction matches each published specification — but a
specification is not a running service. **Treat your first live call as an
experiment: one request, check the response, do not batch.**

**ONDC is deliberately incomplete.** The network requires an Ed25519 signature
over a BLAKE-512 digest. Neither primitive is in the standard library, and
adding a cryptography dependency that could not be exercised against the live
network would be worse than saying so. `build_signing_string` and
`build_authorization_header` produce exactly what the specification calls for
and are tested; the signer is a `Signer` protocol you supply.

**Offline is the default and cannot be lost by accident.** An adapter built
without an explicit `transport=` is offline by construction — credentials in
the environment do not enable a network call. A mutation asserts this.

### Credentials

```bash
python main.py credentials      # what is configured; values never shown
```

| Marketplace | Required environment variables |
|---|---|
| `amazon_in` | `AMAZON_SP_REFRESH_TOKEN`, `AMAZON_SP_CLIENT_ID`, `AMAZON_SP_CLIENT_SECRET` |
| `flipkart` | `FLIPKART_CLIENT_ID`, `FLIPKART_CLIENT_SECRET` |
| `ondc` | `ONDC_SUBSCRIBER_ID`, `ONDC_SIGNING_PRIVATE_KEY` |
| AI research (`research/`) | `OPENAI_API_KEY` |

Optional overrides: `AMAZON_SP_ENDPOINT`, `AMAZON_SP_MARKETPLACE_ID`,
`FLIPKART_ENDPOINT`, `ONDC_ENDPOINT`, `ONDC_UNIQUE_KEY_ID`, `OPENAI_MODEL`,
`OPENAI_BASE_URL`.

Read from the environment when needed and never stored, logged, returned in a
result, or written to the ledger. Authorization headers are redacted at every
boundary, and a test asserts a token cannot appear in a serialised result.

### Going live

```python
from part2.transport import UrllibTransport, RateLimiter
from part2.auth import lwa_provider
from part2.marketplace import get_adapter

limiter = RateLimiter()
transport = UrllibTransport(limiter=limiter)
adapter = get_adapter("amazon_in", transport=transport, tokens=lwa_provider(transport))
adapter.configure_limits(limiter)
result = adapter.get_orders(since=yesterday)  # OK / RETRYABLE / UNKNOWN / ...
```

---

## Safety boundary

Part 1 contains no operational capability — no purchasing, publication,
fulfilment or pricing automation. Part 2 operates the business but **never
moves money**: there is no payment gateway, no card capture, no payout, no bank
transfer. Ordering stock and paying for it are different actions with different
blast radii; Part 2 can recommend and record a purchase, and has no code that
can execute a payment.

Both boundaries are test-enforced by scanning project source only — never
installed dependencies.

---

## Testing

**1029 tests across twenty-two modules.**

| Module | Tests | Covers |
|---|---|---|
| `test_part2.py` | 103 | Execution safety, adapters, inventory, orders, settlement, pricing, portfolio, concurrency |
| `test_repository_integrity.py` | 115 | Checkout completeness, manifest honesty, README accuracy, boundaries, credentials |
| `test_research.py` | 54 | Hallucination prevention, provenance, unavailable data, AI degradation |
| `test_openai_client.py` | 21 | OpenAI transport: key handling, degradation, fabrication rejection |
| `test_api.py` | 89 | HTTP API endpoints, approval gate, safety posture |
| `test_transport.py` | 70 | HTTP transport, auth, rate limiting, pagination, normalization |
| `test_supplier_ingest.py` | 62 | Supplier CSV ingestion: validation, provenance, dedup, no invented facts |
| `test_sourcing.py` | 58 | Discovery, normalization, matching, feeds, opportunity ranking |
| `test_pipeline.py` | 57 | End-to-end decision pipeline, ingestion, LLM degradation |
| `test_phase5_persistence.py` | 21 | Phase 5: durable idempotency, multi-process claim, crash recovery |
| `test_execution_phase4.py` | 46 | Phase 4: approval binding, gates, idempotency, dry-run |
| `test_marketplace_phase3.py` | 35 | Phase 3: adapter capability, listing readiness, dry-run boundary |
| `test_opportunity_evidence.py` | 24 | 2E: missing evidence never reaches RECOMMENDED |
| `test_compliance_evidence.py` | 20 | 2D: missing compliance evidence never becomes PASS |
| `test_economics_evidence.py` | 23 | 2C: missing economics inputs never become confirmed zero |
| `test_economics.py` | 56 | Landed cost, economics, scenarios, gates, scoring, risk, compliance |
| `test_append_only.py` | 51 | UPDATE/DELETE rejection, offer history, transition immutability |
| `test_schemas.py` | 44 | Enums, bounds, timezone rules, cross-field business rules |
| `test_replay.py` | 22 | Replay determinism, point-in-time reconstruction |
| `test_audit_remediation.py` | 21 | Ledger concurrency, terminal-state guard, backfill ordering |
| `test_ledger.py` | 19 | Append, ordering, hash chain, tamper detection |
| `test_database.py` | 18 | Init, idempotency, foreign keys, monetary round-trips |

**Property tests** hold for every input: higher landed cost cannot improve
margin; `worst ≤ expected ≤ best`; missing ITC cannot improve economics;
inventory never goes negative; `reserved + available == on_hand`; duplicate
order imports are always idempotent; negative contribution cannot rank
positive; no price below the floor is ever allowed.

**Failure injection** covers API timeout, 401, 429, 500, malformed JSON,
partial responses, duplicate webhooks, duplicate settlements, listing
rejection, price and inventory drift, supplier price increases, return spikes,
missing demand data, Claude timeouts and malformed Claude output.

**Concurrency** tests cover concurrent ledger writers (verified to 64-way),
racing workers on one execution intent, concurrent reservations against the
last unit, and duplicate concurrent order imports.

### Mutation testing

Eighty-two mutations: 81 guarded, plus one inert control that must survive —
proving the harness can tell the difference. Run a slice with
`--slice 1/2` when the full set is too slow for a CI step.

```
ok   UNKNOWN counts as success                  CAUGHT
ok   approval gate removed                      CAUGHT
ok   idempotency store ignores prior results    CAUGHT
ok   non-idempotent write is retried            CAUGHT
ok   inventory may go negative                  CAUGHT
ok   reservation ignores availability           CAUGHT
ok   unmatched settlement treated as matched    CAUGHT
ok   realized profit ignores return costs       CAUGHT
ok   worst-case gate bypassed by pricing        CAUGHT
ok   publication skips prerequisite checks      CAUGHT
ok   dry run performs the real call             CAUGHT
ok   (control) comment-only edit                SURVIVED
```

---

## Cross-platform notes

**Source-tree boundary.** Everything that walks the repository goes through
`tools/project_files.py`. Source roots are derived from the manifest, and
generated directories are detected structurally: any directory containing
`pyvenv.cfg` is a virtualenv whatever it is called, so `.venv`, `.check-venv`
and `.validation-venv` are all excluded without the code knowing their names.

**UTF-8.** Prices are quoted in rupees, and `\u20b9` has no cp1252 code point.
`main.py` and the root `conftest.py` reconfigure stdout and stderr at startup;
every file read and write passes an explicit encoding; every subprocess is
launched with `encoding="utf-8"` plus `PYTHONUTF8=1`. Tests enforce all three,
including one that runs the CLI under `PYTHONIOENCODING=cp1252`.

---

## Known limitations

These are real and not worked around.

- **No marketplace transport is bundled.** See the adapter table above.
- **Most demand signals are unobtainable.** Expect coverage of 60–70% and read
  any score with its coverage attached.
- **Market price reference is manual** — ten to twenty minutes per product, and
  required.
- **Velocity is a guess** until a SKU has traded, labelled `GUESSED` in every
  calculation that uses it.
- **PDF ingestion is candidates-only**, every row flagged `REQUIRES_REVIEW`.
- **Bundled reference data is unverified.** Enter the real figures before
  acting on any result.
- **Initial-exposure targets are planning figures, not forecasts.** A 12-order
  test target says what a wrong bet costs, not what will sell.
- **Verified on Python 3.12.** `requires-python = ">=3.12"`; run
  `./validate.sh --fresh` to confirm on your interpreter.

---

## What is deliberately still absent

Payment processing, autonomous purchasing, money movement of any kind, and
carrier integrations. Fulfilment beyond recording is
`MANUAL_ACTION_REQUIRED` — where no integration exists, the system says so
rather than fabricating success.
