# Contributing to Brain Trader

## Getting Started

```bash
git clone https://github.com/abhinavjp/brain-trader.git
cd brain-trader
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
cp .env.example .env        # fill in your API keys
```

Install the graphify hook:
```bash
cp docs/hooks/pre-push .git/hooks/pre-push
chmod +x .git/hooks/pre-push
```

## Running Tests

```bash
pytest                  # full suite
pytest tests/unit/      # fast, no network
pytest -x               # stop on first failure
```

## Before Submitting a PR

- [ ] `pytest` passes
- [ ] `mypy brain_trader/` has no errors
- [ ] `ruff check brain_trader/ tests/` has no errors
- [ ] New vendor/provider has a unit test
- [ ] No `reference/` imports anywhere

## Adding a Data Vendor

See `CLAUDE.md` → "Adding a New Data Vendor".

## Adding an LLM Provider

See `CLAUDE.md` → "Adding a New LLM Provider".

## Reporting Bugs

Open an issue with:
- Brain Trader version
- LLM provider + model
- Data vendor
- Minimal reproduction steps
