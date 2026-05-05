import pytest
from pathlib import Path
from unittest.mock import MagicMock
from langchain_core.messages import AIMessage


@pytest.fixture
def tmp_dirs(tmp_path: Path):
    (tmp_path / "logs").mkdir()
    (tmp_path / "cache").mkdir()
    (tmp_path / "memory").mkdir()
    return tmp_path


@pytest.fixture
def sample_config(tmp_dirs: Path):
    from brain_trader.config.schema import BrainTraderConfig, LLMConfig
    return BrainTraderConfig(
        llm=LLMConfig(provider="openai", deep_model="gpt-5.4", quick_model="gpt-5.4-mini"),
        results_dir=tmp_dirs / "logs",
        cache_dir=tmp_dirs / "cache",
        memory_log_path=tmp_dirs / "memory" / "decisions.db",
    )


@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.bind_tools.return_value = llm
    llm.invoke.return_value = AIMessage(content="Mock analyst report.", tool_calls=[])
    return llm


@pytest.fixture
def seeded_memory_db(tmp_dirs: Path):
    from brain_trader.memory.log import DecisionLog
    db_path = tmp_dirs / "memory" / "decisions.db"
    log = DecisionLog(db_path)
    log.store("NVDA", "2026-01-10", "BUY", "Strong momentum. FINAL: BUY 100 shares.")
    log.store("AAPL", "2026-01-11", "HOLD", "Mixed signals. FINAL: HOLD.")
    return log
