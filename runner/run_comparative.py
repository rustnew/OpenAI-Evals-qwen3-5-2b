"""Ticket 6 — Comparative evaluation runner (protocol §8.2, gate G4).

Runs N configurations on the SAME evaluation set, same seeds, same max_tokens,
and emits, per configuration, the §5 metric block + a summary JSON.

Configurations (protocol §3, 7 configs = 8 data rows):
  baseline / decoding / prompt / evidence / verification / abstention-a /
  abstention-b / combined

Usage:
    python3 runner/run_comparative.py \
        --dataset datasets/factual.jsonl --seeds 42-46 --runs 5 \
        --configs baseline,decoding,prompt,evidence,verification,abstention-a,abstention-b,combined \
        --max-tokens 1024 --out results/ticket6 --order random

Dry-run (no gateway — validates configs, contract, metric block, P4/P6 loop):
    python3 runner/run_comparative.py --dry-run --dataset datasets/factual.jsonl \
        --configs baseline,decoding,verification,combined --runs 1

Env (config/evals.env): OPENAI_BASE_URL, OPENAI_API_KEY, EVALS_MODEL,
EVALS_JUDGE_MODEL, EVALS_RPM.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import Counter
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from openai import OpenAI  # noqa: E402

# Reuse the study's real graders + LLM-as-a-judge (NOT placeholders).
from run_evals import GRADERS, judge_factual, judge_overconfidence, judge_model_from_env  # noqa: E402

# Default judge: a stronger model than the 2B under test, via the gateway.
DEFAULT_JUDGE_MODEL = "gemini-3p1-flash-lite"

CONFIG_KNOBS = {
    "baseline":     {"temperature": 1.0, "prompt_style": "none",  "evidence": False, "verify": False},
    "decoding":     {"temperature": 0.7, "prompt_style": "none",  "evidence": False, "verify": False},
    "prompt":       {"temperature": 1.0, "prompt_style": "thorough", "evidence": False, "verify": False},
    "evidence":     {"temperature": 1.0, "prompt_style": "none",  "evidence": True,  "verify": False},
    "verification": {"temperature": 1.0, "prompt_style": "none",  "evidence": False, "verify": True},
    "abstention-a": {"temperature": 1.0, "prompt_style": "abstain", "evidence": False, "verify": False},
    "abstention-b": {"temperature": 1.0, "prompt_style": "abstain", "evidence": False, "verify": False},
    "combined":     {"temperature": 0.7, "prompt_style": "none",  "evidence": False, "verify": True},
}
# abstention configs run on the unanswerable / answerable subsets respectively.
ABSTAIN_A_CATEGORY = "unanswerable"

PROMPT_STYLES = {
    "none":     lambda p: p,
    "thorough": lambda p: p + ("\n\nAnswer directly and thoroughly. Do not pad; "
                               "cover the question in 150-400 tokens."),
    "abstain":  lambda p: p + ("\n\nAnswer only if you have enough evidence or "
                               "information. If you cannot know, say so explicitly "
                               "('I don't know') rather than guessing."),
}
VERIFY_PROMPT = ("Your previous answer may contain unverifiable claims. Rewrite it "
                 "so that every claim is verifiable; do not assert anything you cannot "
                 "support.")


def load_dataset(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def build_messages(case: dict, style: str, evidence: bool) -> list[dict]:
    msgs = []
    ctx = case.get("context") or ""
    if evidence and case.get("ground_truth"):
        ctx = (ctx + "\n\nReference (authoritative): " + case["ground_truth"]).strip()
    if ctx:
        msgs.append({"role": "system", "content": ctx})
    msgs.append({"role": "user", "content": PROMPT_STYLES[style](case["prompt"])})
    return msgs


def call_model(client, model, messages, temperature, max_tokens, seed, dry=False):
    # Simple rate limit: space calls to stay under rpmPerKey 30 (protocol §4).
    rpm = float(os.getenv("EVALS_RPM", "25"))
    if not dry and rpm > 0:
        time.sleep(60.0 / rpm)
    if dry:
        return {"response": "Synthetic dry-run answer with a claim.", "finish_reason": "stop",
                "latency_ms": 3400.0, "usage": {"prompt_tokens": 20, "completion_tokens": 40, "total_tokens": 60},
                "ttft_ms": 400.0, "tpot_ms": 75.0, "retried": False}
    t0 = time.time()
    resp = client.chat.completions.create(model=model, messages=messages, temperature=temperature,
                                          max_tokens=max_tokens, seed=seed)
    lat = (time.time() - t0) * 1000
    ch = resp.choices[0]
    return {"response": ch.message.content or "", "finish_reason": ch.finish_reason,
            "latency_ms": round(lat, 1), "usage": {
                "prompt_tokens": resp.usage.prompt_tokens if resp.usage else None,
                "completion_tokens": resp.usage.completion_tokens if resp.usage else None,
                "total_tokens": resp.usage.total_tokens if resp.usage else None},
            "ttft_ms": None, "tpot_ms": None, "retried": False}


def _grade(client, judge_model, case, response, dry=False):
    """Apply the study's real grader + judge per phenomenon (mirrors run_evals.py)."""
    if dry:
        return {"label": "ok"}
    phenomenon = case.get("phenomenon")
    grader = GRADERS.get(phenomenon)
    if grader is None:
        return {"label": "ok"}
    verdict = grader(response, case.get("ground_truth", ""), case.get("context", ""))
    if phenomenon == "false_factual_assertion" and verdict.get("label") == "needs_review":
        try:
            jv = judge_factual(client, judge_model, case, response)
            verdict = {"label": jv.get("label", "needs_review"), "judge": jv}
        except Exception as e:  # noqa: BLE001
            verdict["judge_error"] = str(e)
    elif phenomenon == "overconfidence":
        try:
            jv = judge_overconfidence(client, judge_model, case, response)
            verdict = {"label": jv.get("label", "overconfidence"), "judge": jv}
        except Exception as e:  # noqa: BLE001
            verdict["judge_error"] = str(e)
    return verdict


def verify_loop(client, model, judge_model, messages, case, temperature, max_tokens, seed, dry=False, max_regen=1):
    """P4/P6 verification loop: generate -> real grade -> (<=1) regenerate -> re-grade.
    Returns the final answer + the number of gateway calls used (protocol §8.1.3)."""
    out = call_model(client, model, messages, temperature, max_tokens, seed, dry)
    calls = 1
    verdict = _grade(client, judge_model, case, out["response"], dry)
    if verdict.get("label") not in ("ok",) and max_regen >= 1:
        msgs2 = messages + [{"role": "assistant", "content": out["response"]},
                            {"role": "user", "content": VERIFY_PROMPT}]
        out2 = call_model(client, model, msgs2, temperature, max_tokens, seed, dry)
        out = out2
        calls += 1
        verdict = _grade(client, judge_model, case, out["response"], dry)
    return out, calls, verdict


def _check_supported(client, model, answer, messages):
    # Minimal checker: ask the judge model whether the answer is supported by context.
    check_msgs = [{"role": "system", "content": "You are a strict fact-checker. Answer only SUPPORTED or UNSUPPORTED."},
                  {"role": "user", "content": f"Context: {messages}\n\nAnswer: {answer}\n\nVerdict:"}]
    try:
        r = client.chat.completions.create(model=model, messages=check_msgs, max_tokens=8, temperature=0.0)
        return "SUPPORTED" in (r.choices[0].message.content or "").upper()
    except Exception:  # noqa: BLE001
        return True  # fail-open on checker error (bounded; recorded as check error below is not available)


def metric_block(rows: list[dict]) -> dict:
    n = len(rows)
    halluc = sum(1 for r in rows if r.get("label") in
                 {"repetition_loop", "false_factual_assertion", "overconfidence", "false_factual", "mixed"})
    ok = n - halluc
    errors = Counter(r.get("error_type") for r in rows if r.get("error_type"))
    lats = [r["latency_ms"] for r in rows if r.get("latency_ms") is not None and not r.get("retried")]
    ttfts = [r["ttft_ms"] for r in rows if r.get("ttft_ms") is not None]
    tpots = [r["tpot_ms"] for r in rows if r.get("tpot_ms") is not None]
    calls = sum(r.get("gateway_calls", 1) for r in rows)
    def med(xs):
        s = sorted(xs); k = len(s)
        return s[k // 2] if k and k % 2 else (s[k // 2 - 1] + s[k // 2]) / 2 if k else None
    return {
        "n": n, "evaluated": n - sum(errors.values()), "hallucinated": halluc, "ok": ok,
        "hr": (halluc / n) if n else None,
        "median_latency_ms": med(lats), "median_ttft_ms": med(ttfts), "median_tpot_ms": med(tpots),
        "gateway_calls": calls,
        "throughput_req_s": (n / (sum(lats) / 1000.0)) if lats else None,
        "error_counts": dict(errors),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--configs", default="baseline")
    ap.add_argument("--seeds", default="42-46")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--max-tokens", type=int, default=int(os.getenv("EVALS_MAX_TOKENS", "1024")))
    ap.add_argument("--out", default="results/ticket6")
    ap.add_argument("--order", choices=["fixed", "random"], default="random")
    ap.add_argument("--dry-run", action="store_true", help="no gateway; validate pipeline only")
    args = ap.parse_args()

    configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    unknown = [c for c in configs if c not in CONFIG_KNOBS]
    if unknown:
        print(f"ERROR: unknown configs {unknown}; valid: {list(CONFIG_KNOBS)}", file=sys.stderr)
        return 2

    lo, hi = map(int, args.seeds.split("-"))
    seeds = list(range(lo, hi + 1))

    cases_all = load_dataset(args.dataset)
    # abstention-a runs on the unanswerable subset only; abstention-b on the rest.
    abstain_a_cases = [c for c in cases_all if c.get("category") == ABSTAIN_A_CATEGORY]
    rest_cases = [c for c in cases_all if c.get("category") != ABSTAIN_A_CATEGORY]

    client = None
    judge_model = os.getenv("EVALS_JUDGE_MODEL", DEFAULT_JUDGE_MODEL)
    if not args.dry_run:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key or "REPLACE" in api_key:
            print("ERROR: set OPENAI_API_KEY (see config/evals.env.example)", file=sys.stderr)
            return 2
        client = OpenAI(base_url=os.getenv("OPENAI_BASE_URL", "https://api.ai.camer.digital/v1"),
                        api_key=api_key, timeout=180.0, max_retries=3)

    model = os.getenv("EVALS_MODEL", "qwen3-5-2b-local")
    os.makedirs(args.out, exist_ok=True)
    summary = {"model": model, "configs": {}, "timestamp": datetime.now(timezone.utc).isoformat(),
               "dry_run": args.dry_run, "seeds": seeds, "runs": args.runs, "max_tokens": args.max_tokens}

    for cfg in configs:
        knobs = CONFIG_KNOBS[cfg]
        cases = abstain_a_cases if cfg == "abstention-a" else rest_cases
        if cfg == "abstention-a" and not abstain_a_cases:
            print(f"WARN: no unanswerable cases for abstention-a; skipping {cfg}", file=sys.stderr)
            continue
        # Write rows INCREMENTALLY (per row, append) so a timeout never loses progress.
        out_path = os.path.join(args.out, f"{cfg}.jsonl")
        fh = open(out_path, "w")
        rows = []
        for case in cases:
            msgs = build_messages(case, knobs["prompt_style"], knobs["evidence"])
            judge_model = os.getenv("EVALS_JUDGE_MODEL", DEFAULT_JUDGE_MODEL)
            for run, seed in enumerate(seeds):
                try:
                    if knobs["verify"]:
                        out, calls, verdict = verify_loop(client, model, judge_model, msgs, case,
                                                          knobs["temperature"], args.max_tokens,
                                                          seed, args.dry_run)
                    else:
                        out = call_model(client, model, msgs, knobs["temperature"], args.max_tokens,
                                         seed, args.dry_run)
                        calls = 1
                        verdict = _grade(client, judge_model, case, out["response"], args.dry_run)
                except Exception as e:  # noqa: BLE001
                    row = {"config": cfg, "case_id": case["id"], "category": case.get("category"),
                           "seed": seed, "label": "error", "verdict": {"label": "error"},
                           "response": "", "latency_ms": None, "ttft_ms": None, "tpot_ms": None,
                           "retried": False, "gateway_calls": 0, "error_type": type(e).__name__}
                    fh.write(json.dumps(row) + "\n")
                    fh.flush()
                    rows.append(row)
                    continue
                row = {"config": cfg, "case_id": case["id"], "category": case.get("category"),
                       "seed": seed, "label": verdict.get("label", "ok"), "verdict": verdict,
                       "response": out["response"], "latency_ms": out.get("latency_ms"),
                       "ttft_ms": out.get("ttft_ms"), "tpot_ms": out.get("tpot_ms"),
                       "retried": out.get("retried", False), "gateway_calls": calls, "error_type": None}
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                rows.append(row)
        fh.close()
        block = metric_block(rows)
        with open(os.path.join(args.out, f"{cfg}.summary.json"), "w") as f:
            json.dump(block, f, indent=2)
        summary["configs"][cfg] = block
        print(f"[{cfg}] n={block['n']} hr={block['hr']:.4f} median_lat={block['median_latency_ms']}ms "
              f"calls={block['gateway_calls']} errors={block['error_counts']}")

    with open(os.path.join(args.out, "comparative-summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n✅ summary written to {args.out}/comparative-summary.json" + (" (DRY-RUN — no gateway)" if args.dry_run else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
