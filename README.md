# d2p — Demo to Product

[![CI](https://github.com/Hosico02/d2p/actions/workflows/ci.yml/badge.svg)](https://github.com/Hosico02/d2p/actions/workflows/ci.yml)

A multi-agent loop that turns a **demo repo** into a **product repo**. No
hardcoded "demo type" detectors — the agents read the project, search the
web for mature competitor products in the same domain, and figure the rest
out themselves. Adding a new demo type (audio, blockchain, robotics, …)
requires **zero code changes**.

---

## How it works

```
                    ┌─────────────────────────────────────────┐
                    │  Analyzer                               │
                    │   • read demo (listing + key files)     │
                    │   • web-search 3–5 mature competitors   │
                    │   • extract features + UI elements      │
                    │   • preserve the demo's "essence"       │
                    └────────────────┬────────────────────────┘
                                     │
                                     ▼
                    ┌─────────────────────────────────────────┐
                    │  Planner                                │
                    │   • diff competitor features vs repo    │
                    │   • emit small, file-level Tasks        │
                    └────────────────┬────────────────────────┘
                                     │
                          ┌──────────┴──────────┐
                          ▼                     ▼
                ┌─────────────────┐   ┌─────────────────┐
                │  Executor #1    │   │  Executor #N    │  ← parallel
                │  read affected  │   │  read affected  │
                │  files → write  │   │  files → write  │
                └────────┬────────┘   └────────┬────────┘
                         └──────────┬──────────┘
                                    ▼
                    ┌─────────────────────────────────────────┐
                    │  QA Agent                               │
                    │   • run accumulated regression tests    │
                    │   • probe for new bugs (checklist +     │
                    │     domain-informed scenarios)          │
                    │   • emit FAILING TESTS as bug reports   │
                    │   • dispatch fix-Tasks (test path is    │
                    │     forbidden to the fix-Executor)      │
                    └────────────────┬────────────────────────┘
                                     │
                                     ▼
                              [next iteration]
```

**Key design properties**

- **No hardcoded detectors.** The Analyzer's prompt does the work. New
  domains need no Python changes.
- **Essence preservation.** The Analyzer separately extracts the demo's
  `essence` (e.g. "agent-vs-agent simulation harness") and `audience`
  (e.g. "LLM agents, humans only observe"). The Planner is forbidden from
  proposing changes that drift the project into a different kind of product.
- **Failing tests are bug reports.** QA emits unittest files that fail
  *now* and stay in `tests/d2p_qa/` as permanent regression guardrails.
  Each iteration grows the corpus.
- **Health + baseline rollback.** Before each Executor write, the
  orchestrator snapshots the modules. If a module that *was* importable
  becomes broken, or a demo-author test that *was* passing flips to
  failing, the change is rolled back automatically.
- **Forbidden-file enforcement.** The fix-Executor sent by QA receives the
  bug-test path in `forbidden_files=[...]` so it cannot weaken or delete
  the test that documents the bug.

---

## Setup

```bash
git clone <this repo>
cd d2p

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Fill in at least one provider key (MINIMAX_API_KEY by default).
```

`MINIMAX_API_KEY` accepts MiniMax Token-Plan keys (prefix `sk-cp-…`). The
default base URL is `https://api.minimaxi.com/anthropic`, default model is
`MiniMax-M2.7-highspeed`.

To use Claude or OpenAI instead:

```bash
# Claude (Anthropic API) — Haiku for executor, Opus for analyzer/planner/QA
D2P_PROVIDER=claude python run.py …

# Codex / OpenAI — gpt-4o-mini for executor, gpt-4o for analyzer/planner/QA
D2P_PROVIDER=codex python run.py …

# Claude via local CLI binary
D2P_PROVIDER=claude-cli python run.py …
```

Per-role model overrides:

```bash
D2P_ROLE_EXECUTOR_MODEL=claude-haiku-4-5
D2P_ROLE_PLANNER_MODEL=claude-opus-4-7
```

---

## Run

```bash
python run.py <path/to/your/demo> --iter 2 --parallel 2
```

Flags:

| flag | default | what it does |
|---|---|---|
| `--iter N` | 2 | max iterations (Analyze → Plan → Execute → QA = 1 iter) |
| `--parallel N` | 4 | concurrent Executors per iteration |
| `--no-qa` | off | skip the QA stage (faster, but no failing-test guardrails added) |
| `--reanalyze-every N` | 0 | re-run Analyzer every N iters (essence/audience pinned); 0 disables |
| `--qa-wontfix-after N` | 3 | retire a QA bug after N failed fix attempts; 0 disables |
| `--max-concurrent-fixes N` | 0 | cap fix tasks per iter (0=no cap); lowest-attempt bugs go first |
| `--no-cache-analysis` | off | force a fresh Analyzer call (ignore `.d2p/analysis_cache.json`) |
| `-v` | off | verbose logs |

Example:

```bash
python run.py ../werewolf-demo --iter 3 --parallel 2 \
    --reanalyze-every 4 --qa-wontfix-after 3
```

Artifacts land in `<target>/.d2p/run-<timestamp>/`:

```
.d2p/run-2026-05-18T12-00-00/
├── analysis.json            # competitor research + capability matrix
├── analysis_iter<N>.json    # re-analysis snapshots (if --reanalyze-every > 0)
├── plan_iter<N>.json        # Task list with file-level intent
├── exec_iter<N>.json        # per-Executor outputs
├── qa_iter<N>.json          # failing tests + fix dispatches
├── qa_fix_iter<N>.json      # per-fix-Executor outputs
├── qa_rerun_iter<N>.json    # full corpus re-run after fixes (regression sweep)
├── qa_regressions_iter<N>.json
├── iter<N>_changes.md       # human-readable digest of what moved this iter
└── summary.json             # final analysis + open bugs + cumulative LLM usage
```

The demo's own `tests/d2p_qa/` directory grows with each run — that's the
permanent regression corpus. Each entry in its `_meta.json` carries
`attempts`, `first_seen_iter`, and `status` (`open` / `fixed` / `wontfix`),
so retired bugs stay visible without re-burning fix budget on them.

### Iteration digests

After every iteration the orchestrator writes `iter<N>_changes.md` — a
skim-friendly view of what actually moved that round:

- the Planner's rationale
- feature tasks (done/failed) with files touched
- QA fixes (done/failed) with the test path each one targeted
- bug lifecycle counters (see below)
- a "Files touched" rollup
- the cumulative LLM-usage line (calls, cost, cache hit ratio)

The Bugs section labels each count by its place in the lifecycle so the
numbers are interpretable in isolation:

```
## Bugs
- carried in (open from prior iters): 2
- new this iter: 4
- incidentally fixed (passed before fix sweep): 1
- fix tasks: 3 ok, 0 failed
- retired (wontfix) this iter: 0
- **still open going forward: 3**
```

That last line is the number that gets carried into the next iter. The
final line of the iter md is the cumulative cost — you don't have to
spelunk JSON to know what a run cost.

---

## Architecture map

| File | Lines | Role |
|---|---:|---|
| `run.py` | 50 | CLI entry |
| `d2p/orchestrator.py` | ~730 | Closed-loop driver, sandbox, rollback, parallelism, iter digests, escalation |
| `d2p/agents.py` | ~960 | Analyzer (cache+fingerprint) / Planner (cap-aware) / Executor LLM agents |
| `d2p/qa.py` | ~720 | QA agent (probe → failing test → fix dispatch, retirement, meta locks) |
| `d2p/providers/` | ~750 | minimax / claude / claude-cli / codex backends, role router, usage ledger, fallback wiring |
| `d2p/lang/` | ~470 | Language adapters (Python + JS health probes, 3.10+ picker) |
| `d2p/fs.py` | 91 | Sandbox file ops + snapshot/restore |
| `d2p/health.py` | 25 | Module-level import probe |
| `d2p/symbols.py` | 60 | Symbol map for the analyzer's project view |
| `d2p/models.py` | 78 | Dataclasses for Task / Feature / ExecutionResult etc. |
| **Total core** | **~4k LOC** | |

### Correctness invariants

- **Sandbox**: every write resolves under the target root; `..` paths raise.
- **Per-task snapshot/restore**: snapshot taken under a per-file lock right
  before the executor writes; rolled back if the post-task probe regresses.
- **Health probe**: module-level `import X` for every top-level `.py` and
  every `tests/*.py`. Catches symbol-deletion regressions that
  syntax-checking misses.
- **Baseline test gate**: demo-author tests that were passing must still
  pass after each task, else rollback.
- **QA test corpus regression sweep**: after each fix round, every corpus
  test is re-run; any previously-green test that regressed triggers a
  conservative rollback of ALL fixes in that round.
- **Meta-file locking**: `_meta.json` mutations (`flip_meta_status`,
  `bump_attempts`, `mark_wontfix`, `_save_meta`) take a single lock —
  required because `flip_meta_status` is invoked from parallel post_check
  callbacks. Without it the read-modify-write loses updates.
- **Test execution serialised**: all corpus test subprocesses share the
  demo's cwd; concurrent runs would collide on shared on-disk state
  (sqlite files, lock files). A second lock serialises the subprocess
  window only — orchestrator parallelism is preserved elsewhere.
- **Forbidden files**: fix tasks include the bug-test path in
  `forbidden_files`; the executor refuses writes to it.

For comparison: the older sibling project [MatrixOmnix](https://github.com/) (`demo2project`) hard-codes
60+ gap detectors + per-detector planner cases + per-task executor handlers,
totalling ~20k LOC. This project is the deliberately-minimal alternative
where the agents themselves do that work.

---

## Provider details

`d2p/providers/__init__.py` ships a `RoleRouter` that picks a model per role
(**executor / fix / analyzer / planner / qa**) so you can run cheap fast
models on the hot path and reasoning-grade models on Analyzer / Planner /
QA. The `fix` role is split from `executor` because empirically QA-bug
fixes need a stronger model than feature edits — see "Sweet spot" below.

| Provider | executor / fix default | analyzer / planner / qa default |
|---|---|---|
| `minimax` | `MiniMax-M2.7-highspeed` | same (single-model) |
| `claude` | `claude-haiku-4-5` | `claude-opus-4-7` |
| `claude-cli` | `haiku` | `opus` |
| `codex` | `gpt-4o-mini` | `gpt-4o` |

Override any role via env:

```bash
D2P_ROLE_EXECUTOR_MODEL=gpt-4o-mini  D2P_ROLE_FIX_MODEL=gpt-4o \
D2P_ROLE_ANALYZER_MODEL=gpt-4o       python run.py ...
```

### Sweet spot (empirical)

```bash
D2P_PROVIDER=claude-cli D2P_ROLE_FIX_MODEL=sonnet python run.py <demo> --iter 2
```

Haiku is fine for feature edits (10× cheaper than Opus and fast enough);
fixes need Sonnet to climb past ~0% success on harder bugs. Opus is
reserved for Analyzer / Planner / QA where reasoning quality dominates.

### Auto-escalation on task failure

Any role can have a fallback model wired via env. When the primary
provider fails a task with a *retryable* error (regression rollback,
syntax fail, SEARCH miss, post-check failure), the orchestrator retries
once with the fallback before giving up:

```bash
# Primary haiku, escalate to sonnet only when haiku fails:
D2P_PROVIDER=claude-cli \
D2P_ROLE_EXECUTOR_FALLBACK_MODEL=sonnet \
D2P_ROLE_FIX_FALLBACK_MODEL=sonnet \
python run.py <demo> --iter 2
```

Structural failures (forbidden test-file edits, sandbox escapes, empty
plan output) are NOT escalated — a stronger model would hit the same
wall. Saves the sonnet/opus tokens for cases where they can plausibly
help.

Fallback usage is tagged as `<role>-fallback` in `summary.json`, so the
cost of escalations is separately visible from the cheap-model baseline:

```
"per_role": {
  "fix:haiku":           { "calls": 5, "cost_usd": 0.20 },
  "fix-fallback:sonnet": { "calls": 1, "cost_usd": 0.14 },
  ...
}
```

### Cost & cache visibility

Every provider call routes through a shared `UsageAccumulator`. At the
end of a run, `summary.json` carries a `usage` block with totals plus a
per-role breakdown:

```json
"usage": {
  "total_calls": 29,
  "total_cost_usd": 1.3612,
  "cache_hit_ratio": 0.683,
  "per_role": {
    "executor:haiku": { "calls": 12, "input": 120, "output": 28536,
                        "cache_creation": 119353, "cache_read": 321363,
                        "cost_usd": 0.3241 },
    "fix:haiku":      { "calls": 12, "cost_usd": 0.4606, ... },
    ...
  }
}
```

`cache_hit_ratio = cache_read / (cache_read + cache_creation)` — higher is
cheaper and faster. The `claude-cli` prompt layout is intentionally
stable-prefix (`=== System ===` → `=== Role ===` → `=== User ===` →
`=== Call options ===`) so per-call variables sink to the trailing block
and the SDK's prompt cache can read the prefix instead of re-encoding it.

---

## What gets generated in your demo

After a run, the **target demo** (not this repo) typically gains:

- Files the LLM decided are missing relative to mature competitors (README
  sections, contract docs, deployment scaffold, runtime guards, …).
- A growing `tests/d2p_qa/` directory of unittest files — each one captures
  a real bug found by the QA agent. The test names document the bug.
- A `.d2p/` directory with the run audit trail (analyzer report, plan,
  per-executor outputs, QA logs, **per-iter markdown digests**, and a final
  `summary.json` with cumulative cost/cache stats).

If a write would break a module that previously imported clean, or break a
demo-author test that previously passed, **the orchestrator rolls back
automatically** — your demo can never be left in a regressed state.

### QA bug lifecycle

Each entry in `tests/d2p_qa/_meta.json` tracks:

| field | meaning |
|---|---|
| `status` | `open` (failing, will retry) / `fixed` (passing) / `wontfix` (retired) |
| `attempts` | how many fix tasks were dispatched against this bug |
| `first_seen_iter` | iteration that first discovered the bug |

After `--qa-wontfix-after N` failed attempts the bug flips to `wontfix`:
the test stays in the corpus (so future runs notice if it accidentally
turns green — the system will flip it back to `fixed`), but the
orchestrator stops dispatching the same broken fix forever.

### Analyzer caching

The Analyzer call is slow (web search + JSON-mode reasoning over the
whole codebase) and largely deterministic for a stable codebase. d2p
fingerprints the analyzer's input (`listing + key file contents + model
identity + system prompt`) and stores results in
`<target>/.d2p/analysis_cache.json`. Subsequent runs against the same
demo are an instant cache hit. Pass `--no-cache-analysis` to force a
fresh run (e.g. when you've changed competitors or want fresh web
findings).

---

## Limitations (honest)

- **LLM stability is the floor.** The whole loop is agent-driven; if the
  model misdiagnoses, the run misdiagnoses. Health-rollback and
  forbidden-test guards prevent silent damage but cannot turn a bad
  generation into a good one. Multi-iteration helps.
- **No score**, no "production_ready" badge. The product is ready when
  *its own tests pass and you accept the result* — d2p doesn't pretend to
  know your bar.
- **Web search via the provider.** Analyzer uses the model's built-in web
  capability where available. If you wire in a provider without web access,
  the Analyzer falls back to repo-only inference (still useful but weaker).

---

## License

MIT. Use, fork, modify freely.
