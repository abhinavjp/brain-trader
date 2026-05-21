# Plan 07 — Observability Layer

**Subsystem:** `brain_trader/observability/`
**Depends on:** Plan 01 (config), Plan 03 (LLM)
**TDD:** write failing test → confirm red → implement → confirm green → commit

---

## Overview

Three components:

| Module | Class/Function | Purpose |
|---|---|---|
| `logging.py` | `get_logger(name)` | Structured JSON logger, honours `BRAIN_TRADER_LOG_LEVEL` |
| `callbacks.py` | `CostTracker` | LangChain callback that accumulates token usage + USD cost |
| `traces.py` | `TraceWriter` | Writes one JSON-Lines file per run; called by `GraphRunner` |

All three are **pure utilities** with no circular imports. `GraphRunner` (Plan 05) will be updated in Task 5 to wire `CostTracker` and `TraceWriter` in.

---

## Task 1 — Structured JSON Logger

### File: `brain_trader/observability/logging.py`

```python
from __future__ import annotations

import logging
import sys
from typing import Any

try:
    import orjson as _json  # fast path
    def _dumps(obj: Any) -> str:
        return _json.dumps(obj).decode()
except ImportError:  # pragma: no cover
    import json as _json  # type: ignore[no-redef]
    def _dumps(obj: Any) -> str:
        return _json.dumps(obj, default=str)


class _JsonFormatter(logging.Formatter):
    """Emit one JSON object per log record."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, datefmt="%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        # Any extra key=value passed via extra={} lands on the record
        for key, val in record.__dict__.items():
            if key not in _STDLIB_ATTRS:
                payload[key] = val
        return _dumps(payload)


_STDLIB_ATTRS = frozenset(logging.LogRecord(
    "", 0, "", 0, "", (), None
).__dict__.keys()) | {"message", "asctime"}

_LOG_LEVEL_ENV = "BRAIN_TRADER_LOG_LEVEL"
_DEFAULT_LEVEL = "INFO"

_registry: dict[str, logging.Logger] = {}


def get_logger(name: str) -> logging.Logger:
    """Return a structured JSON logger.  Idempotent — same object on repeat calls."""
    if name in _registry:
        return _registry[name]

    import os
    level_str = os.getenv(_LOG_LEVEL_ENV, _DEFAULT_LEVEL).upper()
    level = getattr(logging, level_str, logging.INFO)

    logger = logging.getLogger(f"brain_trader.{name}")
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(_JsonFormatter())
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False

    _registry[name] = logger
    return logger
```

**Rules:**
- Never call `print()` — this module is the single logging surface
- `get_logger` is idempotent: calling it twice with the same name returns the same `Logger` object
- Level is read from env at first call; tests override via `monkeypatch.setenv`
- `extra={"ticker": "AAPL"}` pattern propagates arbitrary fields into JSON

---

## Task 2 — CostTracker Callback

### File: `brain_trader/observability/callbacks.py`

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult


# USD per 1 000 tokens (input / output) — approximate public pricing
_COST_TABLE: dict[str, tuple[float, float]] = {
    # (input_per_1k, output_per_1k)
    "gpt-4o":              (0.005,  0.015),
    "gpt-4o-mini":         (0.00015, 0.0006),
    "gpt-4-turbo":         (0.01,   0.03),
    "claude-3-5-sonnet":   (0.003,  0.015),
    "claude-3-haiku":      (0.00025, 0.00125),
    "gemini-2.0-flash":    (0.00010, 0.00040),
    "gemini-1.5-pro":      (0.00125, 0.005),
    "deepseek-chat":       (0.00014, 0.00028),
    "grok-2":              (0.002,  0.010),
    "qwen-turbo":          (0.00005, 0.00015),
}
_DEFAULT_COST = (0.001, 0.002)  # unknown model fallback


@dataclass
class UsageSummary:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated_usd: float = 0.0
    calls: int = 0


class CostTracker(BaseCallbackHandler):
    """Accumulate token usage and estimated cost across all LLM calls in a run."""

    def __init__(self) -> None:
        super().__init__()
        self._usage = UsageSummary()

    # ------------------------------------------------------------------
    # LangChain callback hook
    # ------------------------------------------------------------------

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        for generations in response.generations:
            for gen in generations:
                token_usage = getattr(gen.message, "usage_metadata", None)
                if token_usage is None:
                    # Older LangChain: response.llm_output["token_usage"]
                    token_usage = (response.llm_output or {}).get("token_usage", {})

                prompt_tokens: int = (
                    token_usage.get("input_tokens", 0)
                    or token_usage.get("prompt_tokens", 0)
                )
                completion_tokens: int = (
                    token_usage.get("output_tokens", 0)
                    or token_usage.get("completion_tokens", 0)
                )
                model_name: str = (
                    (response.llm_output or {}).get("model_name", "")
                    or kwargs.get("invocation_params", {}).get("model_name", "")
                    or ""
                )
                cost = self._estimate(model_name, prompt_tokens, completion_tokens)
                self._usage.prompt_tokens += prompt_tokens
                self._usage.completion_tokens += completion_tokens
                self._usage.total_tokens += prompt_tokens + completion_tokens
                self._usage.estimated_usd += cost
                self._usage.calls += 1

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def summary(self) -> UsageSummary:
        return self._usage

    def reset(self) -> None:
        self._usage = UsageSummary()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _estimate(model_name: str, prompt: int, completion: int) -> float:
        key = next(
            (k for k in _COST_TABLE if k in model_name.lower()),
            None,
        )
        in_rate, out_rate = _COST_TABLE.get(key or "", _DEFAULT_COST)
        return (prompt / 1000) * in_rate + (completion / 1000) * out_rate
```

**Rules:**
- `CostTracker` is instantiated once per `GraphRunner.run()` call and passed in `callbacks=[tracker]`
- `UsageSummary` is a plain `dataclass` — no Pydantic — to keep it import-light
- Cost table is approximate; errors in estimation are acceptable (it's for logging/awareness only)
- `reset()` allows reuse across multiple tickers in a batch

---

## Task 3 — TraceWriter

### File: `brain_trader/observability/traces.py`

```python
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from brain_trader.observability.callbacks import UsageSummary
from brain_trader.observability.logging import get_logger

_log = get_logger("traces")

_TRACE_VERSION = 1


class TraceWriter:
    """Write one JSON-Lines trace file per analysis run.

    Each line is a self-contained JSON object (event or summary).
    Files land in ``<trace_dir>/<ticker>/<YYYY-MM-DDTHH-MM-SS>.jsonl``.
    """

    def __init__(self, trace_dir: Path) -> None:
        self._trace_dir = trace_dir
        self._handle: Any = None
        self._ticker: str = ""
        self._run_id: str = ""

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def begin(self, ticker: str, run_id: str) -> None:
        """Open a new trace file.  Must be called before ``write_event``."""
        self._ticker = ticker
        self._run_id = run_id
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
        target = self._trace_dir / ticker / f"{ts}.jsonl"
        target.parent.mkdir(parents=True, exist_ok=True)
        self._handle = target.open("w", encoding="utf-8")
        self._write({"event": "run_start", "ticker": ticker, "run_id": run_id})
        _log.info("trace started", extra={"ticker": ticker, "run_id": run_id, "path": str(target)})

    def write_event(self, event_type: str, **fields: Any) -> None:
        """Append a single event to the open trace."""
        if self._handle is None:
            raise RuntimeError("TraceWriter.begin() must be called first")
        self._write({"event": event_type, **fields})

    def finish(self, decision: str, usage: UsageSummary | None = None) -> None:
        """Write the run_end summary and close the file."""
        if self._handle is None:
            return
        payload: dict[str, Any] = {
            "event": "run_end",
            "ticker": self._ticker,
            "run_id": self._run_id,
            "decision": decision,
        }
        if usage is not None:
            payload["usage"] = {
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens,
                "estimated_usd": round(usage.estimated_usd, 6),
                "calls": usage.calls,
            }
        self._write(payload)
        self._handle.close()
        self._handle = None
        _log.info("trace finished", extra={"ticker": self._ticker, "decision": decision})

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _write(self, payload: dict[str, Any]) -> None:
        line = json.dumps({
            "_v": _TRACE_VERSION,
            "ts": datetime.now(timezone.utc).isoformat(),
            **payload,
        }, default=str)
        self._handle.write(line + "\n")
        self._handle.flush()
```

**Rules:**
- `TraceWriter` is NOT a context manager — `GraphRunner` calls `begin/finish` manually so it can pass `UsageSummary` at the end
- `finish()` is idempotent if `_handle` is already `None`
- All writes are line-buffered (`flush()` after each line) — safe for crash recovery
- The `_v` field enables future schema migration
- `trace_dir` comes from `BrainTraderConfig.trace_dir` (Path, default `.brain_trader/traces`)

---

## Task 4 — Unit Tests

### File: `tests/unit/test_observability.py`

```python
import json
import logging
import os
from pathlib import Path
from unittest.mock import MagicMock, UUID

import pytest
from langchain_core.outputs import ChatGeneration, LLMResult

from brain_trader.observability.callbacks import CostTracker, UsageSummary
from brain_trader.observability.logging import _registry, get_logger
from brain_trader.observability.traces import TraceWriter


# ---------------------------------------------------------------------------
# Logger tests
# ---------------------------------------------------------------------------


class TestGetLogger:
    def test_returns_logger(self):
        logger = get_logger("test_module")
        assert isinstance(logger, logging.Logger)

    def test_idempotent(self):
        a = get_logger("idempotent_test")
        b = get_logger("idempotent_test")
        assert a is b

    def test_json_output(self, capsys):
        # Clear registry entry to get fresh handler
        name = "json_output_test"
        _registry.pop(f"brain_trader.{name}", None)
        if logging.getLogger(f"brain_trader.{name}").handlers:
            logging.getLogger(f"brain_trader.{name}").handlers.clear()
        _registry.pop(name, None)

        logger = get_logger(name)
        logger.info("hello world")
        out = capsys.readouterr().out
        data = json.loads(out.strip())
        assert data["msg"] == "hello world"
        assert data["level"] == "INFO"
        assert "ts" in data

    def test_extra_fields_in_json(self, capsys):
        name = "extra_test"
        _registry.pop(name, None)
        _registry.pop(f"brain_trader.{name}", None)
        if logging.getLogger(f"brain_trader.{name}").handlers:
            logging.getLogger(f"brain_trader.{name}").handlers.clear()

        logger = get_logger(name)
        logger.info("with extra", extra={"ticker": "AAPL"})
        out = capsys.readouterr().out
        data = json.loads(out.strip())
        assert data["ticker"] == "AAPL"

    def test_respects_env_level(self, monkeypatch):
        name = "level_test"
        _registry.pop(name, None)
        if logging.getLogger(f"brain_trader.{name}").handlers:
            logging.getLogger(f"brain_trader.{name}").handlers.clear()

        monkeypatch.setenv("BRAIN_TRADER_LOG_LEVEL", "DEBUG")
        logger = get_logger(name)
        assert logger.level == logging.DEBUG


# ---------------------------------------------------------------------------
# CostTracker tests
# ---------------------------------------------------------------------------


def _make_llm_result(
    prompt_tokens: int,
    completion_tokens: int,
    model_name: str = "gpt-4o-mini",
) -> LLMResult:
    """Build a minimal LLMResult for testing."""
    mock_msg = MagicMock()
    mock_msg.usage_metadata = {
        "input_tokens": prompt_tokens,
        "output_tokens": completion_tokens,
    }
    gen = ChatGeneration(message=mock_msg, text="test")
    return LLMResult(
        generations=[[gen]],
        llm_output={"model_name": model_name},
    )


class TestCostTracker:
    def test_initial_summary_zero(self):
        tracker = CostTracker()
        s = tracker.summary
        assert s.prompt_tokens == 0
        assert s.completion_tokens == 0
        assert s.estimated_usd == 0.0
        assert s.calls == 0

    def test_accumulates_single_call(self):
        tracker = CostTracker()
        result = _make_llm_result(100, 50, "gpt-4o-mini")
        tracker.on_llm_end(result, run_id=UUID(int=0))
        s = tracker.summary
        assert s.prompt_tokens == 100
        assert s.completion_tokens == 50
        assert s.total_tokens == 150
        assert s.calls == 1
        assert s.estimated_usd > 0

    def test_accumulates_multiple_calls(self):
        tracker = CostTracker()
        for _ in range(3):
            result = _make_llm_result(100, 50, "gpt-4o-mini")
            tracker.on_llm_end(result, run_id=UUID(int=0))
        assert tracker.summary.calls == 3
        assert tracker.summary.prompt_tokens == 300

    def test_reset_clears_state(self):
        tracker = CostTracker()
        result = _make_llm_result(100, 50)
        tracker.on_llm_end(result, run_id=UUID(int=0))
        tracker.reset()
        s = tracker.summary
        assert s.prompt_tokens == 0
        assert s.calls == 0

    def test_known_model_cost_calculation(self):
        tracker = CostTracker()
        # gpt-4o: input=0.005/1k, output=0.015/1k
        # 1000 prompt + 1000 completion = $0.005 + $0.015 = $0.020
        result = _make_llm_result(1000, 1000, "gpt-4o")
        tracker.on_llm_end(result, run_id=UUID(int=0))
        assert abs(tracker.summary.estimated_usd - 0.020) < 0.0001

    def test_unknown_model_uses_default(self):
        tracker = CostTracker()
        result = _make_llm_result(1000, 1000, "unknown-model-xyz")
        tracker.on_llm_end(result, run_id=UUID(int=0))
        # default: (0.001, 0.002) → 0.001 + 0.002 = 0.003
        assert abs(tracker.summary.estimated_usd - 0.003) < 0.0001


# ---------------------------------------------------------------------------
# TraceWriter tests
# ---------------------------------------------------------------------------


class TestTraceWriter:
    def test_creates_file_on_begin(self, tmp_path):
        writer = TraceWriter(tmp_path)
        writer.begin("AAPL", "run-001")
        files = list(tmp_path.glob("AAPL/*.jsonl"))
        assert len(files) == 1
        writer.finish("BUY")

    def test_run_start_event_written(self, tmp_path):
        writer = TraceWriter(tmp_path)
        writer.begin("TSLA", "run-002")
        writer.finish("HOLD")
        lines = (tmp_path / "TSLA").glob("*.jsonl")
        events = [json.loads(l) for l in next(lines).read_text().splitlines()]
        assert events[0]["event"] == "run_start"
        assert events[0]["ticker"] == "TSLA"

    def test_run_end_event_has_decision(self, tmp_path):
        writer = TraceWriter(tmp_path)
        writer.begin("GOOG", "run-003")
        writer.finish("SELL")
        lines = list((tmp_path / "GOOG").glob("*.jsonl"))
        events = [json.loads(l) for l in lines[0].read_text().splitlines()]
        end = next(e for e in events if e["event"] == "run_end")
        assert end["decision"] == "SELL"

    def test_run_end_event_has_usage(self, tmp_path):
        writer = TraceWriter(tmp_path)
        writer.begin("MSFT", "run-004")
        usage = UsageSummary(
            prompt_tokens=100, completion_tokens=50, total_tokens=150,
            estimated_usd=0.005, calls=2
        )
        writer.finish("BUY", usage=usage)
        lines = list((tmp_path / "MSFT").glob("*.jsonl"))
        events = [json.loads(l) for l in lines[0].read_text().splitlines()]
        end = next(e for e in events if e["event"] == "run_end")
        assert end["usage"]["total_tokens"] == 150
        assert end["usage"]["calls"] == 2

    def test_write_event_appends_line(self, tmp_path):
        writer = TraceWriter(tmp_path)
        writer.begin("NVDA", "run-005")
        writer.write_event("agent_start", agent="market_analyst")
        writer.finish("BUY")
        lines = list((tmp_path / "NVDA").glob("*.jsonl"))
        events = [json.loads(l) for l in lines[0].read_text().splitlines()]
        assert any(e.get("agent") == "market_analyst" for e in events)

    def test_write_event_before_begin_raises(self, tmp_path):
        writer = TraceWriter(tmp_path)
        with pytest.raises(RuntimeError, match="begin"):
            writer.write_event("agent_start", agent="market_analyst")

    def test_finish_idempotent(self, tmp_path):
        writer = TraceWriter(tmp_path)
        writer.begin("AMD", "run-006")
        writer.finish("HOLD")
        writer.finish("HOLD")  # should not raise

    def test_version_field_present(self, tmp_path):
        writer = TraceWriter(tmp_path)
        writer.begin("META", "run-007")
        writer.finish("BUY")
        lines = list((tmp_path / "META").glob("*.jsonl"))
        events = [json.loads(l) for l in lines[0].read_text().splitlines()]
        assert all(e["_v"] == 1 for e in events)
```

---

## Task 5 — Wire Into GraphRunner

**File:** `brain_trader/graph/runner.py` (from Plan 05) — add `CostTracker` + `TraceWriter`

The following diff shows the changes to `GraphRunner.run()`:

```python
# brain_trader/graph/runner.py  (additions only — full file in Plan 05)

from brain_trader.observability.callbacks import CostTracker
from brain_trader.observability.traces import TraceWriter
from brain_trader.observability.logging import get_logger

_log = get_logger("runner")


class GraphRunner:
    def __init__(self, graph, config: BrainTraderConfig) -> None:
        self._graph = graph
        self._config = config
        self._writer = TraceWriter(config.trace_dir)

    def run(self, ticker: str, trade_date: str, run_id: str | None = None) -> RunResult:
        import uuid
        run_id = run_id or str(uuid.uuid4())
        tracker = CostTracker()
        self._writer.begin(ticker, run_id)
        _log.info("run started", extra={"ticker": ticker, "run_id": run_id})

        initial_state = {
            "messages": [],
            "ticker": ticker,
            "trade_date": trade_date,
            "market_report": "",
            "news_report": "",
            "social_report": "",
            "fundamentals_report": "",
            "bull_argument": "",
            "bear_argument": "",
            "research_report": "",
            "trade_proposal": "",
            "risk_debate_state": {"messages": [], "round": 0},
            "investment_debate_state": {"messages": [], "round": 0},
            "final_decision": "",
            "_max_debate_rounds": self._config.max_debate_rounds,
            "_max_risk_rounds": self._config.max_risk_rounds,
        }

        try:
            # Pass tracker as callback to all LLM calls via RunnableConfig
            result_state = self._graph.invoke(
                initial_state,
                config={"callbacks": [tracker]},
            )
        except Exception as exc:
            _log.error("run failed", extra={"ticker": ticker, "error": str(exc)})
            self._writer.finish("ERROR", usage=tracker.summary)
            raise

        decision = result_state.get("final_decision", "HOLD")
        self._writer.finish(decision, usage=tracker.summary)
        _log.info(
            "run complete",
            extra={
                "ticker": ticker,
                "decision": decision,
                "tokens": tracker.summary.total_tokens,
                "usd": round(tracker.summary.estimated_usd, 4),
            },
        )
        return RunResult(
            ticker=ticker,
            trade_date=trade_date,
            decision=decision,
            state=result_state,
            usage=tracker.summary,
            run_id=run_id,
        )
```

`RunResult` gains the `usage` and `run_id` fields:

```python
@dataclass
class RunResult:
    ticker: str
    trade_date: str
    decision: str
    state: dict
    usage: UsageSummary
    run_id: str
```

---

## Task 6 — Add `trace_dir` and `max_risk_rounds` to Config

**File:** `brain_trader/config/models.py` (from Plan 01) — two additions:

```python
from pathlib import Path

class BrainTraderConfig(BaseSettings):
    llm: LLMConfig = Field(default_factory=LLMConfig)
    data: DataVendorConfig = Field(default_factory=DataVendorConfig)
    max_debate_rounds: int = Field(default=1, ge=1, le=10)
    max_risk_rounds: int = Field(default=1, ge=1, le=5)          # NEW
    trace_dir: Path = Field(default=Path(".brain_trader/traces")) # NEW

    model_config = SettingsConfigDict(
        env_prefix="BRAIN_TRADER_",
        env_nested_delimiter="__",
    )
```

`BRAIN_TRADER_TRACE_DIR=/var/log/brain_trader/traces` is the env override.

---

## Task 7 — Integration Test (optional, no LLM)

### File: `tests/unit/test_observability_integration.py`

```python
"""Verify CostTracker + TraceWriter wire-up without a live LLM."""
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from brain_trader.observability.callbacks import CostTracker, UsageSummary
from brain_trader.observability.traces import TraceWriter


def test_cost_tracker_and_trace_writer_round_trip(tmp_path):
    tracker = CostTracker()
    writer = TraceWriter(tmp_path)

    writer.begin("INTC", "run-int-001")
    writer.write_event("agent_start", agent="market_analyst")

    # Simulate token accumulation
    from unittest.mock import UUID
    from langchain_core.outputs import ChatGeneration, LLMResult
    mock_msg = MagicMock()
    mock_msg.usage_metadata = {"input_tokens": 500, "output_tokens": 250}
    gen = ChatGeneration(message=mock_msg, text="analysis")
    result = LLMResult(
        generations=[[gen]],
        llm_output={"model_name": "gpt-4o-mini"},
    )
    tracker.on_llm_end(result, run_id=UUID(int=0))

    writer.write_event("agent_end", agent="market_analyst")
    writer.finish("BUY", usage=tracker.summary)

    # Verify trace file
    trace_files = list((tmp_path / "INTC").glob("*.jsonl"))
    assert len(trace_files) == 1
    events = [json.loads(l) for l in trace_files[0].read_text().splitlines()]

    assert events[0]["event"] == "run_start"
    end = next(e for e in events if e["event"] == "run_end")
    assert end["decision"] == "BUY"
    assert end["usage"]["prompt_tokens"] == 500
    assert end["usage"]["calls"] == 1
```

---

## Commit Plan

```
git add brain_trader/observability/ tests/unit/test_observability.py \
        tests/unit/test_observability_integration.py \
        brain_trader/config/models.py brain_trader/graph/runner.py
git commit -m "feat(observability): structured JSON logging, CostTracker, TraceWriter"
```

(Config and GraphRunner changes are coordinated; committed together to keep the repo building.)

---

## Module Init

### File: `brain_trader/observability/__init__.py`

```python
from brain_trader.observability.logging import get_logger
from brain_trader.observability.callbacks import CostTracker, UsageSummary
from brain_trader.observability.traces import TraceWriter

__all__ = ["get_logger", "CostTracker", "UsageSummary", "TraceWriter"]
```
