# OpenAI Evals — Qwen3.5-2B hallucination study

Local OpenAI-Evals project for the Qwen3.5-2B hallucination benchmark (ai-helm
Epic #1062, Ticket 2 #1064). Runs the three studied phenomena against our
self-hosted model through the AI gateway.

## What it does

```
datasets/ (JSONL, ground truth) → OpenAI Evals → Qwen3.5-2B (gateway) → responses
→ graders (looping / factual / overconfidence) → metrics → results/
```

## Prerequisites

- Python 3.10+
- Access to the AI gateway (`https://api.ai.camer.digital`), model `qwen3-5-2b-local`
- An API key for the gateway (set as `OPENAI_API_KEY`)

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Configure the endpoint

Copy `config/evals.env.example` to `config/evals.env` and fill in the API key:

```bash
export OPENAI_BASE_URL=https://api.ai.camer.digital
export OPENAI_API_KEY=<your-gateway-key>
export EVALS_MODEL=qwen3-5-2b-local
```

## Run

```bash
bash runner/run_evals.sh
```

Outputs land in `results/` (gitignored).

## Structure

| Path | Purpose |
|---|---|
| `datasets/` | JSONL prompt datasets (looping / factual / overconfidence), with ground truth |
| `evals/` | OpenAI-Evals eval definitions (YAML) |
| `graders/` | Custom graders (deterministic looping, factual, overconfidence) |
| `runner/` | Run scripts |
| `config/` | Endpoint / model configuration |
| `results/` | Output (gitignored) |

## Design

See the design doc in inference-ops:
`docs/reference/hallucination-benchmark-design.md` (PR #27).
