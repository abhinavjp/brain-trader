# Brain Trader — Coding Rules

These rules are enforced by pre-commit hooks and CI. Violations block merge.

## Hard Rules

### Config
- All configuration must flow through `BrainTraderConfig` (Pydantic)
- Never instantiate a plain `dict` as a config object
- Never read env vars directly in agent/data/graph code — use config fields

### Data Access
- Agent code must never import `yfinance`, `alpha_vantage`, or any data library directly
- All data access goes through `DataVendorRegistry`
- Tool functions in `data/tools.py` are the only place that calls registry methods

### Logging
- Never use `print()` in library code (`brain_trader/` package)
- Use `logger = logging.getLogger(__name__)` at module level
- The structured logger is initialized by `BrainTrader.__init__` — do not configure logging elsewhere

### Public API
Only these are part of the public API surface:
- `brain_trader.BrainTrader`
- `brain_trader.BrainTraderConfig`
- `brain_trader.cli.main:app` (the CLI entry point)

Everything else is internal. Do not import from sub-modules in external code.

### Reference Folder
- `reference/` is listed in `.gitignore` and must never be committed
- Do not `import` from `reference/` in any source file
- Do not copy code from `reference/` without consciously redesigning it

## Code Quality

### Types
- All function signatures in `brain_trader/` must be fully typed
- No `Any` without an explicit comment explaining why
- Run `mypy brain_trader/` before committing

### Tests
- All tests must pass before committing (`pytest`)
- Unit tests must not make network calls or LLM API calls
- Integration tests use `mock_llm` fixture from `conftest.py`

### Comments
- Only add comments when the WHY is non-obvious (a constraint, a workaround, a subtle invariant)
- Never comment what the code does — names should do that
- No multi-line comment blocks or docstrings describing parameters

### File Size
- If a file exceeds ~300 lines, it is probably doing too much — split it
- Each module has one clear purpose

## Git

- Commit messages describe WHY, not what
- Never force-push to `main`
- Pre-commit hook runs tests; never use `--no-verify` to skip
