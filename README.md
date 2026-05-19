# d2p — Demo to Product

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
| `-v` | off | verbose logs |

Example:

```bash
python run.py ../werewolf-demo --iter 3 --parallel 2
```

Artifacts land in `<target>/.d2p/run-<timestamp>/`:

```
.d2p/run-2026-05-18T12-00-00/
├── analysis.json          # competitor research + capability matrix
├── plan.json              # Task list with file-level intent
├── execution-iter-1.json  # per-Executor outputs
├── qa-iter-1.json         # failing tests + fix dispatches
└── ...
```

The demo's own `tests/d2p_qa/` directory grows with each run — that's the
permanent regression corpus.

---

## Architecture map

| File | Lines | Role |
|---|---:|---|
| `run.py` | 30 | CLI entry |
| `d2p/orchestrator.py` | 388 | Closed-loop driver, sandbox, rollback, parallelism |
| `d2p/agents.py` | 875 | Analyzer / Planner / Executor LLM agents |
| `d2p/qa.py` | 594 | QA agent (probe → failing test → fix dispatch) |
| `d2p/providers/` | ~600 | minimax / claude / claude-cli / codex backends + role router |
| `d2p/lang/` | ~200 | Language adapters (Python + JS health probes) |
| `d2p/fs.py` | 91 | Sandbox file ops + snapshot/restore |
| `d2p/health.py` | 25 | Module-level import probe |
| `d2p/symbols.py` | 60 | Symbol map for the analyzer's project view |
| `d2p/models.py` | 78 | Dataclasses for Task / Feature / ExecutionResult etc. |
| **Total core** | **~2.5k LOC** | |

For comparison: the older sibling project [MatrixOmnix](https://github.com/) (`demo2project`) hard-codes
60+ gap detectors + per-detector planner cases + per-task executor handlers,
totalling ~20k LOC. This project is the deliberately-minimal alternative
where the agents themselves do that work.

---

## Provider details

`d2p/providers/__init__.py` ships a `RoleRouter` that picks a model per role
(executor / analyzer / planner / qa) so you can run cheap fast models on the
hot path and reasoning-grade models on Analyzer / Planner / QA:

| Provider | executor default | analyzer / planner / qa default |
|---|---|---|
| `minimax` | `MiniMax-M2.7-highspeed` | same (single-model) |
| `claude` | `claude-haiku-4-5` | `claude-opus-4-7` |
| `claude-cli` | `haiku` | `opus` |
| `codex` | `gpt-4o-mini` | `gpt-4o` |

Override any role via env:

```bash
D2P_ROLE_EXECUTOR_MODEL=gpt-4o-mini   D2P_ROLE_ANALYZER_MODEL=gpt-4o   python run.py ...
```

---

## What gets generated in your demo

After a run, the **target demo** (not this repo) typically gains:

- Files the LLM decided are missing relative to mature competitors (README
  sections, contract docs, deployment scaffold, runtime guards, …).
- A growing `tests/d2p_qa/` directory of unittest files — each one captures
  a real bug found by the QA agent. The test names document the bug.
- A `.d2p/` directory with the run audit trail (analyzer report, plan,
  per-executor outputs, qa logs).

If a write would break a module that previously imported clean, or break a
demo-author test that previously passed, **the orchestrator rolls back
automatically** — your demo can never be left in a regressed state.

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
