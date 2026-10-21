# judgelab: evidence-sensitivity experiments on LLM legal judgments

A research instrument. It measures how LLM judgments of fictional Indian criminal cases change when a single evidence item is changed. It measures **observable response differences** only. It is not a decision tool, and it does not measure how a model weighs evidence internally.

- Method: [docs/01-methodology.md](docs/01-methodology.md)
- Protocol and schema: [docs/02-protocol.md](docs/02-protocol.md)

## Setup

```bash
python -m venv .venv
```
```bash
.venv/Scripts/python -m pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in `ANTHROPIC_API_KEY`, `OPENAI_API_KEY` and `GEMINI_API_KEY`. `.env` is gitignored and never committed; real environment variables override it.

## Use

| Step | Command |
|---|---|
| Validate cases and specs | `python -m judgelab validate cases experiments` |
| Offline rehearsal | `python -m judgelab --dry-run plan experiments/pilot-01.yaml`, then `python -m judgelab --dry-run run pilot-01` |
| Live run | `python -m judgelab plan experiments/smoke-01.yaml`, then `python -m judgelab run smoke-01 --max-usd 2` |
| Export | `python -m judgelab export smoke-01 --format csv` |
| Analyse | `python -m judgelab analyze smoke-01` |
| Dashboard | `streamlit run app.py` |
| Tests | `python -m pytest tests` |
