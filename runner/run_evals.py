"""Qwen3.5-2B hallucination benchmark runner.

Reads a dataset (JSONL), calls the model through the OpenAI-compatible gateway,
applies the study's graders, aggregates metrics, and writes results.

Usage:
    python3 runner/run_evals.py --dataset datasets/looping.jsonl [--out results/...]

Env (see config/evals.env):
    OPENAI_BASE_URL, OPENAI_API_KEY, EVALS_MODEL,
    EVALS_TEMPERATURE, EVALS_MAX_TOKENS, EVALS_SEED
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from openai import OpenAI  # noqa: E402

from graders import loop_grader, factual_grader, overconfidence_grader  # noqa: E402

GRADERS = {
    "repetition_looping": loop_grader.grade,
    "false_factual_assertion": factual_grader.grade,
    "overconfidence": overconfidence_grader.grade,
}


def load_dataset(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def call_model(client: OpenAI, model: str, case: dict, temperature: float, max_tokens: int, seed: int) -> dict:
    messages = []
    if case.get("context"):
        messages.append({"role": "system", "content": case["context"]})
    messages.append({"role": "user", "content": case["prompt"]})
    t0 = time.time()
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        seed=seed,
    )
    latency_ms = (time.time() - t0) * 1000
    choice = resp.choices[0]
    return {
        "response": choice.message.content or "",
        "finish_reason": choice.finish_reason,
        "usage": {
            "prompt_tokens": resp.usage.prompt_tokens if resp.usage else None,
            "completion_tokens": resp.usage.completion_tokens if resp.usage else None,
            "total_tokens": resp.usage.total_tokens if resp.usage else None,
        },
        "latency_ms": round(latency_ms, 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", default="results")
    ap.add_argument("--runs", type=int, default=int(os.getenv("EVALS_RUNS", "1")))
    args = ap.parse_args()

    base_url = os.getenv("OPENAI_BASE_URL")
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("EVALS_MODEL", "qwen3-5-2b-local")
    temperature = float(os.getenv("EVALS_TEMPERATURE", "1.0"))
    max_tokens = int(os.getenv("EVALS_MAX_TOKENS", "512"))
    seed = int(os.getenv("EVALS_SEED", "42"))

    if not api_key or "REPLACE" in api_key:
        print("ERROR: set OPENAI_API_KEY (see config/evals.env.example)", file=sys.stderr)
        return 2

    client = OpenAI(base_url=base_url, api_key=api_key)
    cases = load_dataset(args.dataset)
    os.makedirs(args.out, exist_ok=True)

    all_rows = []
    for case in cases:
        phenomenon = case["phenomenon"]
        grader = GRADERS.get(phenomenon)
        if grader is None:
            print(f"WARN: no grader for phenomenon {phenomenon!r}, skipping {case['id']}")
            continue
        for run in range(args.runs):
            try:
                out = call_model(client, model, case, temperature, max_tokens, seed + run)
                verdict = grader(out["response"], case.get("ground_truth", ""), case.get("context", ""))
            except Exception as e:  # noqa: BLE001
                verdict = {"label": "error", "error": str(e)}
                out = {"response": "", "finish_reason": None, "usage": None, "latency_ms": None}
            row = {
                "case_id": case["id"],
                "category": case["category"],
                "phenomenon": phenomenon,
                "run": run,
                "prompt": case["prompt"],
                "context": case.get("context", ""),
                "ground_truth": case.get("ground_truth", ""),
                "response": out["response"],
                "finish_reason": out["finish_reason"],
                "usage": out["usage"],
                "latency_ms": out["latency_ms"],
                "temperature": temperature,
                "max_tokens": max_tokens,
                "seed": seed + run,
                "verdict": verdict,
            }
            all_rows.append(row)

    # aggregate
    evaluated = [r for r in all_rows if r["verdict"].get("label") != "error"]
    hallucinated = [
        r for r in evaluated
        if r["verdict"].get("label") in ("repetition_loop", "false_factual_assertion", "overconfidence")
    ]
    hr = len(hallucinated) / len(evaluated) if evaluated else 0.0

    summary = {
        "dataset": args.dataset,
        "model": model,
        "base_url": base_url,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "seed": seed,
        "runs": args.runs,
        "total_cases": len(cases),
        "evaluated": len(evaluated),
        "hallucinated": len(hallucinated),
        "hallucination_rate": round(hr, 4),
        "by_label": {},
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    for r in evaluated:
        lbl = r["verdict"].get("label")
        summary["by_label"][lbl] = summary["by_label"].get(lbl, 0) + 1

    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    base = os.path.basename(args.dataset).replace(".jsonl", "")
    with open(os.path.join(args.out, f"{base}-{ts}.jsonl"), "w") as f:
        for r in all_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(args.out, f"{base}-{ts}-summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
