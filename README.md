# Qwen3.5-2B hallucination benchmark — custom runner

> **Naming note.** This repository is historically named `OpenAI-Evals-qwen3-5-2b`,
> but it is **not** built on the OpenAI Evals framework (`openai-evals`). It is a
> **custom Python pipeline** that follows the Evals *pattern* (dataset → runner →
> grader → metrics) using the raw `openai` client and hand-written graders. The
> framework is **not imported anywhere** in the code.

Custom runner for the Qwen3.5-2B hallucination benchmark (ai-helm Epic #1062,
Ticket 2 #1064). Runs the three studied phenomena against our self-hosted model
through the AI gateway.

## What it does

```
datasets/ (JSONL, ground truth) → runner (openai client) → Qwen3.5-2B (gateway)
→ graders (custom: looping / factual / overconfidence) → metrics → results/
```

Implementation detail:
- **Runner** (`runner/run_evals.py`) uses the raw `openai` client — **no**
  `evals` imports, no registry, no `CompletionFn`, no YAML eval definitions.
- **Graders** (`graders/`) are hand-written: a deterministic n-gram loop
  detector, and hybrid factual/overconfidence graders with an LLM-as-a-judge
  (`gemini-3p1-flash-lite`) applying our own rubric.
- The design doc explains why we keep the custom runner instead of migrating to
  the formal framework: [`hallucination-benchmark-design.md`].

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
| `graders/` | Custom graders (deterministic looping, factual, overconfidence) + LLM-as-a-judge |
| `runner/` | Run scripts (raw `openai` client) |
| `analysis/` | Ticket 3 offline baseline analyzer |
| `config/` | Endpoint / model configuration |
| `results/` | Output (gitignored) |

> There is **no** `evals/` YAML directory — the OpenAI Evals framework is not used.

## Design

See the design doc in inference-ops:
`docs/reference/hallucination-benchmark-design.md` (PR #27). It documents the
"Evals = tooling, not the judge" principle and why the runner is a custom
implementation of that pattern.
