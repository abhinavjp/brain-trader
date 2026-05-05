# Brain Trader — Claude Code Instructions

## Project Overview

Brain Trader is a multi-agent LLM framework for financial market analysis. Specialist analyst agents gather market data, bull/bear researchers debate the findings, a trader synthesizes a proposal, and a risk panel with a portfolio manager issues a final BUY/HOLD/SELL decision.

## Package Layout

```
brain_trader/
├── config/          Pydantic config (BrainTraderConfig + sub-models)
├── agents/          All agent node functions + AgentState
│   ├── analysts/    market, news, social, fundamentals
│   ├── researchers/ bull, bear
│   ├── risk/        aggressive, conservative, neutral
│   ├── managers/    research, portfolio
│   └── trader.py
├── data/            DataVendorRegistry + BaseDataVendor ABC + yfinance/alpha_vantage vendors
├── llm/             LLM provider factory + model catalog
├── graph/           GraphBuilder (StateGraph construction) + GraphRunner (execution)
├── memory/          SQLite-backed DecisionLog + Reflector
├── observability/   Structured JSON logging + CostTracker + TraceWriter
└── cli/             Typer entry point + Rich display + config wizard
```

## How to Run Tests

```bash
pip install -e ".[dev]"
pytest                          # all tests
pytest tests/unit/              # unit tests only (no network, no LLM)
pytest tests/integration/       # integration tests (mock LLM)
```

## Coding Conventions

- **Config:** All config flows through `BrainTraderConfig`. Never create plain `dict` config objects.
- **Data vendors:** Agent code never imports `yfinance` or `alpha_vantage` directly. All data access goes through `DataVendorRegistry`.
- **Logging:** Never use `print()` in library code. Use the structured logger from `brain_trader.observability.logging`.
- **Comments:** Only when the WHY is non-obvious. No docstrings explaining what the code does.
- **Types:** All function signatures are fully typed. Run `mypy brain_trader/` before committing.

## Environment Setup

Copy `.env.example` to `.env` and fill in your API keys. At minimum you need one LLM provider key and optionally `ALPHA_VANTAGE_API_KEY` for the Alpha Vantage data vendor.

## Adding a New Data Vendor

1. Create `brain_trader/data/vendors/yourvendor.py`
2. Implement `BaseDataVendor` (all abstract methods)
3. Register in `DataVendorRegistry._register_defaults()` with a string key
4. Add the key as a `Literal` option in `DataVendorConfig`

## Adding a New LLM Provider

1. Create `brain_trader/llm/providers/yourprovider.py`
2. Extend `BaseLLMClient`
3. Add a branch in `brain_trader/llm/factory.py`
4. Add provider models to `brain_trader/llm/catalog.py`

## Graphify Hook

The `docs/hooks/pre-push` script runs graphify on every push to keep the knowledge graph current. Install it once:

```bash
cp docs/hooks/pre-push .git/hooks/pre-push
chmod +x .git/hooks/pre-push
```

## Reference Folder

`reference/TradingAgents/` is inspiration-only. It is gitignored and must never be imported, referenced in code, or committed. Use it only for understanding the original design.

## Design Specs

Full design documents are in `docs/superpowers/specs/`:
- `2026-05-05-brain-trader-plan-a-design.md` — this project (Brain Trader)
- `2026-05-05-brain-trader-plan-b-design.md` — future generic framework (separate repo)
