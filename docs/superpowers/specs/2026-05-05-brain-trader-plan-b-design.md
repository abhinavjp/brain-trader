---
title: Brain Trader — Plan B Design Spec (Generic Framework)
date: 2026-05-05
status: approved
note: To be implemented in a separate repository. Brain Trader (Plan A) becomes the first domain pack.
---

# Brain Trader — Plan B: Generic Multi-Agent Framework (Domain Pack System)

## Overview

Plan B extracts the orchestration layer from Brain Trader into a domain-agnostic core framework. Any problem that can be structured as "gather data → debate findings → synthesize → assess risk → decide" can be solved by writing a domain pack, with zero changes to core orchestration.

Brain Trader (Plan A) becomes the first domain pack (`financial`). Future packs could cover legal research, competitive intelligence, medical triage, policy analysis, etc.

**Working name:** `agentcore` (rename before publish)
**Tech stack:** Python 3.12+, LangGraph, LangChain, Pydantic v2, Typer, Rich

---

## Architecture Principle

Core provides:
- Graph orchestration (LangGraph StateGraph builder + runner)
- LLM routing (multi-provider factory)
- Memory (base protocol + default SQLite implementation)
- Observability (structured logging, cost tracking, traces)
- CLI shell (generic entry point + hookable display)
- Pack discovery (Python entry points)

Packs provide:
- Agent definitions (analysts, debaters, risk panel, synthesizer, decider)
- Tool implementations (data fetching, API calls)
- State schema (TypedDict fields specific to the domain)
- Config schema (Pydantic model for domain-specific settings)
- Display formatter (how CLI renders domain results)
- Graph topology override (optional; default pipeline used if omitted)

---

## Package Structure

```
agentcore/
├── agentcore/
│   ├── __init__.py                  # public API: DomainOrchestrator, load_pack()
│   ├── core/
│   │   ├── orchestrator.py          # DomainOrchestrator: top-level runner
│   │   ├── state.py                 # BaseState, BaseDebateState TypedDicts
│   │   └── topology.py              # GraphTopology, default_pipeline()
│   ├── packs/
│   │   ├── base.py                  # DomainPack Protocol
│   │   └── registry.py              # pack discovery via entry points
│   ├── graph/
│   │   ├── builder.py               # domain-agnostic GraphBuilder
│   │   ├── runner.py                # GraphRunner (identical to Plan A)
│   │   └── conditions.py            # default conditional edge helpers
│   ├── llm/                         # identical to Plan A (shared)
│   ├── memory/
│   │   ├── base.py                  # BaseMemory Protocol
│   │   └── log.py                   # default SQLite implementation
│   ├── observability/               # identical to Plan A (shared)
│   └── cli/
│       ├── shell.py                 # generic Typer shell
│       └── display.py               # hookable display interface
├── packs/
│   └── financial/                   # Brain Trader as a domain pack
│       ├── __init__.py
│       ├── pack.py                  # FinancialPack implements DomainPack
│       ├── agents/                  # all Brain Trader agents
│       ├── data/                    # DataVendorRegistry + vendors
│       ├── state.py                 # financial-specific state fields
│       └── config.py               # FinancialConfig(BaseModel)
├── tests/
│   ├── core/                        # core orchestration tests
│   ├── packs/
│   │   └── test_pack_contract.py    # generic contract tests any pack must pass
│   └── financial/                   # financial pack-specific tests
├── docs/superpowers/specs/
├── CLAUDE.md
├── RULES.md
└── pyproject.toml
```

---

## The DomainPack Contract

Any domain pack is a Python object (typically a class instance) that satisfies this Protocol:

```python
class AgentRole(TypedDict):
    name: str
    system_prompt: str
    tools: list[BaseTool]
    llm_tier: Literal["deep", "quick"]

class GraphTopology(TypedDict):
    analysts: list[str]               # ordered list of analyst role names
    debate_pair: tuple[str, str]      # (bull_role, bear_role) equivalents
    risk_panel: list[str]             # risk debater role names
    synthesizer: str                  # research manager equivalent
    trader: str                       # trader role name
    decider: str                      # portfolio manager equivalent

class DomainPack(Protocol):
    name: str                         # e.g. "financial", "legal"
    version: str

    def state_schema(self) -> type[TypedDict]: ...
    def config_schema(self) -> type[BaseModel]: ...
    def agent_roles(self) -> dict[str, AgentRole]: ...
    def topology(self) -> GraphTopology: ...      # or None for default pipeline
    def initial_state(self, subject: str, context: str, config: BaseModel) -> dict: ...
    def format_result(self, final_state: dict) -> str: ...
```

Packs that follow the standard pipeline (analysts → debate → risk → decision) return `None` from `topology()` and the core uses `default_pipeline()`. Packs with non-standard flows return a custom `GraphTopology`.

---

## Core Orchestrator

```python
class DomainOrchestrator:
    def __init__(self, pack: DomainPack, config: BaseModel | None = None): ...
    def run(self, subject: str, context: str = "") -> tuple[dict, str]: ...
```

`subject` is domain-specific: a ticker symbol for financial, a case description for legal, a company name for competitive intelligence.

`run()` mirrors `BrainTrader.analyze()` exactly, but delegates all domain-specific behavior to the pack:

```
DomainOrchestrator.run(subject, context)
  ├── pack.initial_state(subject, context, config) → AgentState seed
  ├── GraphBuilder.build(pack.agent_roles(), pack.topology())
  ├── GraphRunner.run(initial_state)
  ├── memory.store(subject, result)
  └── return (final_state, pack.format_result(final_state))
```

---

## Pack Discovery

Packs are discovered via Python entry points. A pack package declares:

```toml
# in pack's pyproject.toml
[project.entry-points."agentcore.packs"]
financial = "agentcore_financial:FinancialPack"
```

Core discovers installed packs at startup:

```python
# agentcore/packs/registry.py
def discover_packs() -> dict[str, DomainPack]:
    return {
        ep.name: ep.load()()
        for ep in importlib.metadata.entry_points(group="agentcore.packs")
    }
```

This means adding a new domain = `pip install agentcore-legal` → it appears in `agentcore list-packs`.

---

## Memory

**`BaseMemory` Protocol:**

```python
class BaseMemory(Protocol):
    def store(self, subject: str, run_date: str, result: str) -> None: ...
    def resolve(self, subject: str, run_date: str, outcome: dict) -> None: ...
    def get_context(self, subject: str) -> str: ...
```

The default SQLite implementation from Plan A is the bundled default. Packs can override with a custom `BaseMemory` implementation — e.g. a legal pack might store case citations in a vector store.

---

## CLI Shell

The generic CLI has two layers:

1. **Core commands** (always available):
   - `agentcore packs list` — show installed packs
   - `agentcore run [PACK] [SUBJECT]` — run any pack
   - `agentcore memory export [PACK] [FILE]` — export memory log

2. **Pack-contributed commands** — packs can register additional subcommands via `pack.cli_commands() -> list[typer.Typer]`. The financial pack contributes `brain-trader` as its own entry point.

---

## Pack Contract Tests

Any pack implementation must pass a standard test suite in `tests/packs/test_pack_contract.py`:

```python
@pytest.mark.parametrize("pack", discover_packs().values())
def test_pack_has_required_fields(pack): ...
def test_pack_state_schema_is_typeddict(pack): ...
def test_pack_config_schema_is_pydantic(pack): ...
def test_pack_agent_roles_have_required_keys(pack): ...
def test_pack_topology_is_valid_or_none(pack): ...
def test_pack_initial_state_returns_dict(pack): ...
def test_pack_format_result_returns_string(pack): ...
```

This ensures new packs satisfy the protocol before they ship.

---

## Relationship to Plan A

| Plan A (Brain Trader) | Plan B (agentcore) |
|---|---|
| `BrainTrader` class | `DomainOrchestrator` class |
| Hard-coded financial agents | `FinancialPack` implements `DomainPack` |
| `DataVendorRegistry` in `data/` | Moved into `financial/data/` |
| `BrainTraderConfig` | `FinancialConfig` extends `BaseModel` |
| `brain-trader` CLI | `agentcore run financial` + `brain-trader` alias |
| SQLite `DecisionLog` | Becomes default `BaseMemory` implementation in core |
| Observability, LLM layer | Copied verbatim into core (unchanged) |

Migration path: Brain Trader (Plan A) can be refactored into a `FinancialPack` without changing its external behavior. The refactor is additive — wrap existing classes into the pack protocol, move the data layer into `packs/financial/data/`.

---

## Implementation Order (for when Plan B is built)

1. Extract `llm/`, `observability/`, `graph/` from Brain Trader into `agentcore` core
2. Define `DomainPack` Protocol and `BaseMemory` Protocol
3. Implement `DomainOrchestrator` wrapping `GraphBuilder` + `GraphRunner`
4. Implement pack discovery via entry points
5. Wrap Brain Trader's agents/data/config into `FinancialPack`
6. Write pack contract test suite
7. Wire up generic CLI shell + financial pack commands
8. Publish `agentcore` and `agentcore-financial` as separate packages
