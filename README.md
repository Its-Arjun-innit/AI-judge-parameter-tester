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

**Local AI, free and private (recommended).** Install [Ollama](https://ollama.com), start it, and install one model, for example:

```bash
ollama pull qwen3:4b-instruct
```

The app never downloads models by itself and never falls back to a cloud service.

**Cloud AI (optional).** Copy `.env.example` to `.env` and fill in `ANTHROPIC_API_KEY`, `OPENAI_API_KEY` and `GEMINI_API_KEY`. `.env` is gitignored and never committed; real environment variables override it.

## Use

**For researchers:** start the app and follow the screens: Overview → Cases → Experiments → Results.

```bash
.venv/Scripts/python -m streamlit run app.py
```

It opens at http://localhost:8501 and is reachable from this computer only. To learn the workflow without a model, choose **Practice data** in Models & Settings.

**From the command line** (the app uses the same commands):

| Step | Command |
|---|---|
| Validate cases and specs | `python -m judgelab validate cases experiments` |
| Offline rehearsal | `python -m judgelab --dry-run plan experiments/pilot-01.yaml`, then `python -m judgelab --dry-run run pilot-01` |
| Local run (Ollama) | `python -m judgelab plan experiments/ollama-smoke.yaml`, then `python -m judgelab run ollama-smoke` |
| Cloud run | `python -m judgelab plan experiments/smoke-01.yaml`, then `python -m judgelab run smoke-01 --max-usd 2` |
| Export | `python -m judgelab export ollama-smoke --format csv` |
| Analyse | `python -m judgelab analyze ollama-smoke` |
| Tests | `python -m pytest tests` |

Press Ctrl+C (or use **Stop** in the app) to stop after the current analysis; running the same command again continues where it stopped.
