# Plan 09 — Integration Tests & End-to-End Smoke

**Subsystem:** `tests/integration/`
**Depends on:** All plans 01–08
**TDD:** write failing test → confirm red → implement → confirm green → commit

---

## Overview

These tests exercise the full agent pipeline end-to-end **without network calls** — all LLM and data vendor I/O is mocked. They live in `tests/integration/` and are excluded from the default `pytest` run unless `--integration` is passed (configured via `conftest.py`).

Three test suites:

| File | Tests |
|---|---|
| `test_full_pipeline.py` | Complete BrainTrader analysis run using mock LLM |
| `test_graph_flow.py` | Graph state transitions + debate termination |
| `test_cli_smoke.py` | CLI subprocess smoke tests |

---

## Task 1 — pytest Configuration

### File: `tests/conftest.py` (additions)

```python
# Add to existing conftest.py

def pytest_addoption(parser):
    parser.addoption(
        "--integration",
        action="store_true",
        default=False,
        help="Run integration tests (slower, mock LLM required)",
    )


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--integration"):
        skip_integration = pytest.mark.skip(reason="Pass --integration to run")
        for item in items:
            if "integration" in str(item.fspath):
                item.add_marker(skip_integration)
```

### File: `tests/integration/__init__.py`

Empty — marks the directory as a package.

---

## Task 2 — Deterministic LLM Fixture

### File: `tests/integration/conftest.py`

```python
from __future__ import annotations

from collections import deque
from typing import Any, Iterator
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult


class _ScriptedLLM:
    """Returns pre-scripted AIMessage responses in order.

    When the queue is exhausted it loops back to the last response.
    """

    def __init__(self, responses: list[str]) -> None:
        self._queue: deque[str] = deque(responses)
        self._last: str = responses[-1]

    def invoke(self, *args: Any, **kwargs: Any) -> AIMessage:
        text = self._queue.popleft() if self._queue else self._last
        return AIMessage(content=text)

    def bind_tools(self, tools: Any) -> "_ScriptedLLM":
        return self

    def __or__(self, other: Any) -> "_ScriptedLLM":
        """Support prompt | llm chain syntax."""
        return self

    def __ror__(self, other: Any) -> "_ScriptedLLM":
        return self


# Minimal scripted responses for each agent node
_BUY_SCRIPT = [
    # market analyst
    "Market conditions are bullish. Strong momentum and increasing volume.",
    # news analyst
    "Positive news flow. Earnings beat expectations. No negative headlines.",
    # social analyst
    "Social sentiment is positive. High retail interest.",
    # fundamentals analyst
    "Strong P/E ratio relative to peers. Good free cash flow.",
    # bull researcher
    "BULLISH: Strong fundamentals + positive momentum.",
    # bear researcher
    "BEARISH: Valuation stretched vs historical averages.",
    # research manager → summarises debate
    "Research Summary: Bull case outweighs bear case on fundamentals.",
    # trader proposal
    "Proposal: BUY — risk/reward favourable at current price.",
    # risk panel × 3 (aggressive, conservative, neutral)
    "RISK APPROVE: Acceptable risk profile.",
    "RISK NEUTRAL: Proceed with caution.",
    "RISK APPROVE: Agrees with trade proposal.",
    # portfolio manager
    "FINAL DECISION: BUY. Conviction: HIGH.",
]

_HOLD_SCRIPT = [
    "Market is range-bound with no clear direction.",
    "News flow is neutral. No catalysts.",
    "Social sentiment mixed.",
    "Fundamentals stable but no growth.",
    "BULLISH: Some upside potential.",
    "BEARISH: Macro headwinds limit upside.",
    "Research Summary: Neutral conviction.",
    "Proposal: HOLD — insufficient signal.",
    "RISK NEUTRAL: Low conviction, hold.",
    "RISK NEUTRAL: Agree with hold.",
    "RISK NEUTRAL: Agree.",
    "FINAL DECISION: HOLD. Conviction: LOW.",
]


@pytest.fixture
def scripted_buy_llm() -> _ScriptedLLM:
    return _ScriptedLLM(_BUY_SCRIPT)


@pytest.fixture
def scripted_hold_llm() -> _ScriptedLLM:
    return _ScriptedLLM(_HOLD_SCRIPT)


@pytest.fixture
def integration_config(tmp_path):
    """Minimal BrainTraderConfig pointing at tmp directories."""
    from brain_trader.config.models import BrainTraderConfig
    return BrainTraderConfig(
        trace_dir=tmp_path / "traces",
        memory=MagicMock(db_path=tmp_path / "test.db"),
    )
```

---

## Task 3 — Full Pipeline Test

### File: `tests/integration/test_full_pipeline.py`

```python
"""End-to-end pipeline tests with scripted LLM (no network)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from tests.integration.conftest import _ScriptedLLM, _BUY_SCRIPT, _HOLD_SCRIPT


@pytest.mark.integration
class TestFullPipeline:
    """Exercises BrainTrader.analyze() through the entire agent pipeline."""

    def _make_bt(self, llm_responses: list[str], tmp_path, integration_config):
        """Build a BrainTrader with a scripted LLM and mocked data vendors."""
        from brain_trader._core import BrainTrader
        from tests.integration.conftest import _ScriptedLLM

        scripted_llm = _ScriptedLLM(llm_responses)

        with (
            patch("brain_trader._core.LLMFactory") as mock_factory,
            patch("brain_trader._core.DataVendorRegistry") as mock_registry,
            patch("brain_trader._core.DecisionLog") as mock_log,
        ):
            mock_factory.create.return_value = scripted_llm
            mock_registry.return_value.get_tools.return_value = []
            mock_log.return_value = MagicMock()
            bt = BrainTrader.from_config(integration_config)
            return bt, mock_log.return_value

    def test_buy_decision_end_to_end(self, tmp_path, integration_config):
        bt, mock_memory = self._make_bt(_BUY_SCRIPT, tmp_path, integration_config)
        with (
            patch("brain_trader._core.DataVendorRegistry"),
            patch("brain_trader._core.DecisionLog", return_value=mock_memory),
            patch("brain_trader._core.LLMFactory"),
        ):
            # GraphBuilder creates the graph using the scripted LLM
            result = bt.analyze("AAPL", "2024-01-15")

        assert result.decision in {"BUY", "HOLD", "SELL"}
        assert result.ticker == "AAPL"
        assert result.trade_date == "2024-01-15"
        assert result.run_id is not None

    def test_memory_store_called_after_analysis(self, tmp_path, integration_config):
        bt, mock_memory = self._make_bt(_BUY_SCRIPT, tmp_path, integration_config)
        bt.analyze("TSLA", "2024-01-16")
        mock_memory.store.assert_called_once()
        call_kwargs = mock_memory.store.call_args.kwargs
        assert call_kwargs["ticker"] == "TSLA"
        assert call_kwargs["trade_date"] == "2024-01-16"

    def test_trace_file_created(self, tmp_path, integration_config):
        integration_config.trace_dir = tmp_path / "traces"
        bt, _ = self._make_bt(_HOLD_SCRIPT, tmp_path, integration_config)
        bt.analyze("GOOG", "2024-01-17")
        trace_files = list((tmp_path / "traces" / "GOOG").glob("*.jsonl"))
        assert len(trace_files) == 1

    def test_usage_summary_populated(self, tmp_path, integration_config):
        bt, _ = self._make_bt(_BUY_SCRIPT, tmp_path, integration_config)
        result = bt.analyze("MSFT", "2024-01-18")
        # Usage may be 0 if scripted LLM doesn't emit token metadata — just check field exists
        assert hasattr(result.usage, "calls")
        assert hasattr(result.usage, "total_tokens")
```

---

## Task 4 — Graph Flow Tests

### File: `tests/integration/test_graph_flow.py`

```python
"""Test graph conditional edges and debate termination without a real LLM."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from brain_trader.graph.conditions import (
    should_continue_investment_debate,
    should_continue_risk_debate,
    route_after_trade_proposal,
)


@pytest.mark.integration
class TestConditionalEdges:
    """Unit-level tests for graph edge conditions (no LLM needed)."""

    # ------------------------------------------------------------------
    # Investment debate
    # ------------------------------------------------------------------

    def test_investment_debate_continues_below_max(self):
        state = {
            "investment_debate_state": {"messages": [], "round": 0},
            "bull_argument": "bullish",
            "bear_argument": "bearish",
            "_max_debate_rounds": 2,
        }
        assert should_continue_investment_debate(state) == "continue_debate"

    def test_investment_debate_ends_at_max(self):
        state = {
            "investment_debate_state": {"messages": [], "round": 2},
            "_max_debate_rounds": 2,
        }
        assert should_continue_investment_debate(state) == "end_debate"

    def test_investment_debate_ends_when_no_bull_or_bear(self):
        state = {
            "investment_debate_state": {"messages": [], "round": 1},
            "bull_argument": "",
            "bear_argument": "",
            "_max_debate_rounds": 3,
        }
        assert should_continue_investment_debate(state) == "end_debate"

    # ------------------------------------------------------------------
    # Risk debate
    # ------------------------------------------------------------------

    def test_risk_debate_continues_below_max(self):
        state = {
            "risk_debate_state": {"messages": [], "round": 0},
            "_max_risk_rounds": 1,
        }
        assert should_continue_risk_debate(state) == "continue_risk_debate"

    def test_risk_debate_ends_at_max(self):
        state = {
            "risk_debate_state": {"messages": [], "round": 1},
            "_max_risk_rounds": 1,
        }
        assert should_continue_risk_debate(state) == "end_risk_debate"

    # ------------------------------------------------------------------
    # Signal extractor
    # ------------------------------------------------------------------

    def test_route_after_proposal_buy(self):
        from brain_trader.graph.signals import extract_signal
        assert extract_signal("I recommend a BUY here.") == "BUY"

    def test_route_after_proposal_sell(self):
        from brain_trader.graph.signals import extract_signal
        assert extract_signal("This looks like a SELL.") == "SELL"

    def test_route_after_proposal_hold_default(self):
        from brain_trader.graph.signals import extract_signal
        assert extract_signal("No clear direction.") == "HOLD"

    def test_route_after_proposal_case_insensitive(self):
        from brain_trader.graph.signals import extract_signal
        assert extract_signal("Recommend buy at current levels.") == "BUY"


@pytest.mark.integration
class TestGraphBuilder:
    """Smoke test: GraphBuilder produces a compilable StateGraph."""

    def test_build_returns_compiled_graph(self, integration_config):
        from brain_trader.graph.builder import GraphBuilder
        from brain_trader.data.registry import DataVendorRegistry

        scripted_llm = MagicMock()
        scripted_llm.bind_tools.return_value = scripted_llm

        with patch("brain_trader.graph.builder.DataVendorRegistry") as mock_reg:
            mock_reg.return_value.get_tools.return_value = []
            builder = GraphBuilder(
                config=integration_config,
                registry=mock_reg.return_value,
                deep_llm=scripted_llm,
                quick_llm=scripted_llm,
                memory=MagicMock(),
            )
            graph = builder.build()
        # LangGraph compiled graphs have an `invoke` method
        assert hasattr(graph, "invoke")
```

---

## Task 5 — CLI Smoke Tests

### File: `tests/integration/test_cli_smoke.py`

```python
"""CLI smoke tests via Typer test runner (BrainTrader mocked)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from brain_trader.cli.main import app
from brain_trader.graph.runner import RunResult
from brain_trader.observability.callbacks import UsageSummary


runner = CliRunner(mix_stderr=False)


def _make_result(decision: str = "BUY") -> RunResult:
    return RunResult(
        ticker="AAPL",
        trade_date="2024-01-15",
        decision=decision,
        state={
            "research_report": "ok",
            "final_decision": decision,
            "market_report": "ok",
            "trade_proposal": "ok",
        },
        usage=UsageSummary(100, 50, 150, 0.001, 3),
        run_id="smoke-001",
    )


@pytest.mark.integration
class TestCLISmoke:
    @patch("brain_trader.cli.main.BrainTrader.from_env")
    def test_analyze_buy(self, mock_from_env):
        mock_from_env.return_value.analyze.return_value = _make_result("BUY")
        result = runner.invoke(app, ["analyze", "AAPL", "2024-01-15"])
        assert result.exit_code == 0
        assert "BUY" in result.output

    @patch("brain_trader.cli.main.BrainTrader.from_env")
    def test_analyze_sell(self, mock_from_env):
        mock_from_env.return_value.analyze.return_value = _make_result("SELL")
        result = runner.invoke(app, ["analyze", "AAPL", "2024-01-15"])
        assert result.exit_code == 0
        assert "SELL" in result.output

    @patch("brain_trader.cli.main.BrainTrader.from_env")
    def test_analyze_quiet_flag(self, mock_from_env):
        mock_from_env.return_value.analyze.return_value = _make_result("HOLD")
        result = runner.invoke(app, ["analyze", "AAPL", "2024-01-15", "--quiet"])
        assert result.exit_code == 0
        assert result.output.strip() == "HOLD"

    @patch("brain_trader.cli.main.BrainTrader.from_env")
    def test_analyze_error_exits_1(self, mock_from_env):
        mock_from_env.side_effect = RuntimeError("no API key")
        result = runner.invoke(app, ["analyze", "AAPL", "2024-01-15"])
        assert result.exit_code == 1

    @patch("brain_trader.cli.main.BrainTrader.from_env")
    def test_memory_pending_no_rows(self, mock_from_env):
        bt = MagicMock()
        bt.pending.return_value = []
        mock_from_env.return_value = bt
        result = runner.invoke(app, ["memory", "pending", "AAPL"])
        assert result.exit_code == 0
        assert "No pending" in result.output

    @patch("brain_trader.cli.main.BrainTrader.from_env")
    def test_memory_resolve_ok(self, mock_from_env):
        mock_from_env.return_value = MagicMock()
        result = runner.invoke(
            app,
            ["memory", "resolve", "AAPL", "2024-01-15", "--raw", "3.5"],
        )
        assert result.exit_code == 0

    @patch("brain_trader.cli.main.BrainTraderConfig")
    def test_config_show_no_crash(self, mock_config):
        mock_config.return_value.model_dump.return_value = {
            "llm": {"provider": "openai", "api_key": "sk-secret"}
        }
        result = runner.invoke(app, ["config", "show"])
        assert result.exit_code == 0
        # API key must be redacted
        assert "sk-secret" not in result.output

    def test_help_text(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "analyze" in result.output
        assert "memory" in result.output
        assert "config" in result.output
```

---

## Task 6 — pyproject.toml Test Configuration

Add to `pyproject.toml`:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v --tb=short"

[tool.mypy]
python_version = "3.11"
strict = true
ignore_missing_imports = true
```

And ensure `tests/integration/` is listed under `testpaths` but gated behind the `--integration` flag from Task 1.

---

## Task 7 — GitHub Actions CI (Optional)

### File: `.github/workflows/ci.yml`

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.11", "3.12"]

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}

      - name: Install dependencies
        run: pip install -e ".[dev]"

      - name: Lint
        run: ruff check brain_trader/ tests/

      - name: Type check
        run: mypy brain_trader/

      - name: Unit tests
        run: pytest tests/unit/ -v

      - name: Integration tests (mock LLM)
        run: pytest tests/integration/ --integration -v
```

---

## Commit Plan

```
git add tests/integration/ tests/conftest.py .github/
git commit -m "test(integration): full pipeline, graph flow, and CLI smoke tests"
git push
```

---

## Summary: Complete Plan Series

With Plan 09, the full implementation blueprint is complete:

| Plan | Subsystem | Key Types |
|---|---|---|
| 01 | Config | `BrainTraderConfig`, `LLMConfig`, `DataVendorConfig` |
| 02 | Data layer | `BaseDataVendor`, `DataVendorRegistry`, `YFinanceVendor` |
| 03 | LLM layer | `BaseLLMClient`, `LLMFactory`, `ModelCatalog` |
| 04 | Agents | `AgentState`, 13 agent factory functions |
| 05 | Graph layer | `GraphBuilder`, `GraphRunner`, `RunResult`, signal extractor |
| 06 | Memory layer | `DecisionLog`, `Reflector` |
| 07 | Observability | `get_logger`, `CostTracker`, `TraceWriter` |
| 08 | Core + CLI | `BrainTrader`, Typer CLI, Rich output |
| 09 | Integration tests | Full pipeline, graph flow, CLI smoke |

**Implementation order:** 01 → 02 → 03 → 04 → 05 → 06 → 07 → 08 → 09 (each plan's tests pass before moving to the next).
