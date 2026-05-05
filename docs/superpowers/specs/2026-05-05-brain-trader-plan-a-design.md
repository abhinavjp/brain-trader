---
title: Brain Trader — Plan A Design Spec
date: 2026-05-05
status: approved
---

# Brain Trader — Plan A: Modular Monolith

## Overview

Brain Trader is a multi-agent LLM framework for financial market analysis and trading decisions. It mirrors the structure of a real trading firm: specialist analysts gather data, bull/bear researchers debate the findings, a trader synthesizes a proposal, and a risk panel with a portfolio manager issues the final decision.

This spec describes a ground-up reimplementation of the TradingAgents reference project with clean architecture, typed config, a proper data abstraction layer, SQLite-backed memory, structured observability, and a complete test suite. The external behavior is equivalent; the internals are rebuilt.

**Tech stack:** Python 3.12+, LangGraph, LangChain, Pydantic v2, Typer, Rich, SQLite (via stdlib `sqlite3`)

---

## Package Structure

```
brain_trader/
├── brain_trader/
│   ├── __init__.py                  # public API: BrainTrader, BrainTraderConfig
│   ├── config/
│   │   ├── schema.py                # Pydantic BrainTraderConfig + sub-models
│   │   └── defaults.py              # default values, env var mappings
│   ├── agents/
│   │   ├── state.py                 # AgentState, InvestDebateState, RiskDebateState
│   │   ├── analysts/
│   │   │   ├── market.py
│   │   │   ├── news.py
│   │   │   ├── social.py
│   │   │   └── fundamentals.py
│   │   ├── researchers/
│   │   │   ├── bull.py
│   │   │   └── bear.py
│   │   ├── risk/
│   │   │   ├── aggressive.py
│   │   │   ├── conservative.py
│   │   │   └── neutral.py
│   │   ├── managers/
│   │   │   ├── research.py
│   │   │   └── portfolio.py
│   │   └── trader.py
│   ├── data/
│   │   ├── registry.py              # DataVendorRegistry
│   │   ├── tools.py                 # LangChain @tool wrappers (call registry)
│   │   └── vendors/
│   │       ├── base.py              # BaseDataVendor ABC
│   │       ├── yfinance.py
│   │       └── alpha_vantage.py
│   ├── llm/
│   │   ├── factory.py               # create_llm_client()
│   │   ├── catalog.py               # model → provider mapping
│   │   └── providers/
│   │       ├── base.py              # BaseLLMClient
│   │       ├── openai.py
│   │       ├── anthropic.py
│   │       ├── google.py
│   │       ├── azure.py
│   │       └── openai_compat.py     # deepseek, qwen, glm, ollama, openrouter
│   ├── graph/
│   │   ├── builder.py               # GraphBuilder: constructs StateGraph from config
│   │   ├── runner.py                # GraphRunner: executes graph, handles checkpoints
│   │   └── conditions.py            # conditional edge logic
│   ├── memory/
│   │   ├── log.py                   # DecisionLog: SQLite-backed
│   │   └── reflection.py            # Reflector: LLM-generated post-trade reflection
│   ├── observability/
│   │   ├── logging.py               # structured JSON logging setup
│   │   ├── callbacks.py             # CostTracker LangChain callback
│   │   └── traces.py                # per-run trace writer
│   └── cli/
│       ├── main.py                  # Typer app, entry point
│       ├── wizard.py                # interactive config wizard (Rich prompts)
│       └── display.py               # Rich panels, progress, tables
├── tests/
│   ├── conftest.py                  # mock_llm, sample_config, seeded_memory_db
│   ├── unit/
│   │   ├── test_config.py
│   │   ├── test_data_registry.py
│   │   ├── test_memory_log.py
│   │   └── test_signal_processing.py
│   └── integration/
│       ├── test_graph_flows.py
│       └── test_checkpoint_resume.py
├── docs/
│   ├── hooks/
│   │   └── pre-push                 # graphify hook (copy to .git/hooks/)
│   └── superpowers/specs/
├── CLAUDE.md
├── RULES.md
├── CONTRIBUTING.md
├── pyproject.toml
├── .env.example
├── .gitignore
└── Dockerfile
```

---

## Components

### Config (`brain_trader/config/`)

All configuration flows through a single validated Pydantic model. No plain dicts anywhere in library code.

```python
class DataVendorConfig(BaseModel):
    core_stock: Literal["yfinance", "alpha_vantage"] = "yfinance"
    technical_indicators: Literal["yfinance", "alpha_vantage"] = "yfinance"
    fundamental_data: Literal["yfinance", "alpha_vantage"] = "yfinance"
    news_data: Literal["yfinance", "alpha_vantage"] = "yfinance"
    tool_overrides: dict[str, str] = {}   # per-tool vendor override

class LLMConfig(BaseModel):
    provider: str = "openai"
    deep_model: str = "gpt-5.4"
    quick_model: str = "gpt-5.4-mini"
    backend_url: str | None = None
    reasoning_effort: str | None = None
    thinking_level: str | None = None
    effort: str | None = None

class ObservabilityConfig(BaseModel):
    log_level: str = "INFO"
    log_format: Literal["json", "text"] = "json"
    trace_enabled: bool = True
    cost_tracking: bool = True

class BrainTraderConfig(BaseSettings):
    llm: LLMConfig = LLMConfig()
    data: DataVendorConfig = DataVendorConfig()
    analysts: list[str] = ["market", "social", "news", "fundamentals"]
    max_debate_rounds: int = Field(default=1, ge=1, le=10)
    max_risk_rounds: int = Field(default=1, ge=1, le=10)
    output_language: str = "English"
    checkpoint_enabled: bool = False
    results_dir: Path = Path.home() / ".brain_trader" / "logs"
    cache_dir: Path = Path.home() / ".brain_trader" / "cache"
    memory_log_path: Path = Path.home() / ".brain_trader" / "memory" / "decisions.db"
    memory_max_entries: int | None = None
    observability: ObservabilityConfig = ObservabilityConfig()

    model_config = SettingsConfigDict(
        env_prefix="BRAIN_TRADER_",
        env_nested_delimiter="__",
        env_file=".env",
    )
```

Config can be loaded three ways (last wins):
1. Defaults from `BrainTraderConfig()`
2. `.env` file
3. Environment variables (`BRAIN_TRADER_LLM__PROVIDER=anthropic`)

### Data Layer (`brain_trader/data/`)

**`BaseDataVendor` ABC** defines the interface every vendor must implement:
- `get_stock_data(ticker, start, end) -> str`
- `get_indicators(ticker, indicators, start, end) -> str`
- `get_fundamentals(ticker) -> str`
- `get_balance_sheet(ticker) -> str`
- `get_cashflow(ticker) -> str`
- `get_income_statement(ticker) -> str`
- `get_news(ticker, date) -> str`
- `get_global_news(date) -> str`
- `get_insider_transactions(ticker) -> str`

All methods return a formatted string (for LLM consumption). On data unavailability they return a structured error string rather than raising, so analyst agents can note the gap in their report.

Vendors implement `@with_retry(attempts=3, base_delay=2.0)` for transient failures.

**`DataVendorRegistry`** is initialized once from config and resolves the correct vendor per tool at startup. Agent tool code calls `registry.get_stock_data(ticker, ...)` — no vendor conditionals in agent files.

**`tools.py`** wraps registry methods as LangChain `@tool` functions. These are the objects passed to `ToolNode`.

### Graph (`brain_trader/graph/`)

**`GraphBuilder`** constructs the `StateGraph` from config. Replaces `GraphSetup`. Accepts `selected_analysts`, `tool_nodes`, `conditional_logic` and returns a compiled `StateGraph`. Contains zero execution logic.

**`GraphRunner`** executes a compiled graph. Responsibilities:
- Generate `run_id` (UUID4)
- Inject `past_context` into initial state
- Handle checkpoint setup/teardown
- Stream or invoke graph
- Write trace JSON via `TraceWriter`
- Call `DecisionLog.store()` on completion
- Clear checkpoint on success
- Return `(final_state, signal)`

**`conditions.py`** contains all conditional edge functions. Unchanged from reference in behavior; extracted to a single module (not a class method) for testability.

### Memory (`brain_trader/memory/`)

**`DecisionLog`** uses SQLite instead of the markdown append-only log. Schema:

```sql
CREATE TABLE decisions (
    id          INTEGER PRIMARY KEY,
    ticker      TEXT NOT NULL,
    trade_date  TEXT NOT NULL,
    rating      TEXT NOT NULL,
    decision    TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'pending',  -- 'pending' | 'resolved'
    raw_return  REAL,
    alpha_return REAL,
    holding_days INTEGER,
    reflection  TEXT,
    created_at  TEXT NOT NULL
);
```

All writes are transactions. `batch_update_with_outcomes` is a single `UPDATE` per row. `get_context()` is a SQL query with `ORDER BY trade_date DESC LIMIT ?`. The markdown export method (`export_markdown(path)`) produces the same human-readable format as the reference for compatibility.

**`Reflector`** is unchanged in behavior: takes `final_decision`, `raw_return`, `alpha_return` and calls the quick LLM to generate a one-paragraph reflection.

### Observability (`brain_trader/observability/`)

**`logging.py`** configures Python's `logging` module with a JSON formatter. Every log record includes `run_id`, `ticker`, `node`, `timestamp`. Initialized once at `BrainTrader.__init__`.

**`CostTracker`** is a `BaseCallbackHandler` that accumulates token counts and estimated cost per run from `on_llm_end` events. Attached to LLM clients at construction. `summary()` returns a dict written into the trace file and printed in the CLI summary panel.

**`TraceWriter`** writes `results/{ticker}/traces/{run_id}.json` containing: config snapshot, node timings, token counts, final signal.

### Core Orchestrator (`BrainTrader`)

The single public-facing class. Kept thin:

```python
class BrainTrader:
    def __init__(self, config: BrainTraderConfig | None = None): ...
    def analyze(self, ticker: str, date: str) -> tuple[dict, str]: ...
```

`analyze()` orchestrates: resolve pending memory → build graph → run → return `(state, signal)`. All real work delegated to `GraphBuilder`, `GraphRunner`, `DecisionLog`, `Reflector`.

### CLI (`brain_trader/cli/`)

**Entry point:** `brain-trader` command (configured in `pyproject.toml`).

**Subcommands:**
- `brain-trader analyze [TICKER] [DATE]` — run analysis
- `brain-trader analyze --checkpoint` — with checkpoint resume
- `brain-trader memory export [FILE]` — export decision log to markdown
- `brain-trader memory clear` — clear decision log

**`wizard.py`** walks through provider, model, analyst selection interactively (Rich prompts). Saves to `~/.brain_trader/config.toml`.

**`display.py`** renders analyst reports as Rich panels, debate rounds as progress, and final decision as a styled summary table with signal, token count, cost estimate.

---

## Data Flow

```
BrainTrader.analyze(ticker, date)
  ├── resolve pending memory entries (same ticker, fetch returns + reflect)
  └── GraphRunner.run(ticker, date, past_context)
        ├── build initial AgentState
        ├── compile graph (± checkpointer)
        └── stream graph nodes:
              [Analyst Phase]  sequential, each analyst calls DataVendorRegistry
              [Research Phase] Bull ↔ Bear debate → Research Manager
              [Trader]         investment plan → trade proposal
              [Risk Phase]     Aggressive ↔ Conservative ↔ Neutral → Portfolio Manager
        ├── write trace JSON
        ├── store_decision() → SQLite
        ├── clear checkpoint (if enabled)
        └── return (final_state, signal)
```

---

## Error Handling

| Error type | Handling |
|---|---|
| Vendor data unavailable | Return structured error string; analyst notes gap in report |
| Vendor network failure | `@with_retry(3, 2s)` → `DataVendorError` propagates to caller |
| LLM transient timeout | LangChain retry (provider-specific) |
| Structured output parse failure | Fall back to `SignalProcessor` free-text extraction |
| Graph node exception | Checkpoint NOT cleared (preserves resume); `AnalysisError` raised |
| Checkpoint schema mismatch | Discard checkpoint, start fresh, log warning |

---

## Testing Strategy

| Layer | Scope | Key tools |
|---|---|---|
| Unit | Config validation, registry routing, memory CRUD, signal parsing, conditional logic | `pytest`, no network |
| Integration | Full graph flows (all analyst combos), checkpoint resume | `unittest.mock` for LLM, real SQLite |
| Smoke | CLI subcommands, `analyze()` public API | Subprocess + mock LLM fixtures |

**`conftest.py` fixtures:**
- `mock_llm` — `MagicMock` returning deterministic `AIMessage` sequences
- `sample_config` — `BrainTraderConfig` with temp dirs
- `seeded_memory_db` — pre-populated SQLite with known pending and resolved entries

---

## Repo Initialization

- `.gitignore` includes `reference/` (inspiration-only, never committed)
- `CLAUDE.md` documents package layout, how to run tests, coding conventions, env var requirements, how to add a vendor/provider
- `RULES.md` enforces: typed config only, no vendor imports in agent code, no `print()` in library code, tests must pass before commit
- `docs/hooks/pre-push` contains the graphify hook script; CLAUDE.md explains how to install it
- `pyproject.toml` defines `[project.scripts] brain-trader = "brain_trader.cli.main:app"`
