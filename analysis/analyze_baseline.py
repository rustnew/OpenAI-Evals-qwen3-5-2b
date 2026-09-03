"""Ticket 3 — Offline baseline analysis for Qwen3.5-2B hallucinations.

Reads the committed Ticket 2 evidence (raw N=5 judge runs) and computes the
six baseline metrics + severity (provisional) + per-category HR +
systematic-vs-stochastic + a dual-annotation sample. **No gateway / live access
required** — it reads the raw `*.jsonl` evidence only.

Usage:
    python3 analysis/analyze_baseline.py \
        --evidence /path/to/evidence/hallucination-benchmark \
        --out results-baseline

Evidence layout (from inference-ops, PR #27):
    <evidence>/factual-20260901-124234.jsonl
    <evidence>/overconfidence-20260901-125001.jsonl
    <evidence>/looping-20260902-130724.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone

HALLUCINATED_LABELS = {"repetition_loop", "false_factual_assertion", "overconfidence"}
EXPECTED_ABSTENTION_CATEGORIES = {"unanswerable"}

# Abstention phrases per the annotation rules / graders.
_ABSTAIN_PHRASES = [
    "i don't know", "i do not know", "i don't have that information",
    "i cannot determine", "not enough information", "i'm not sure",
    "i am not sure", "i cannot access", "i can't access", "i don't have access",
    "i can't know", "no information",
]


def _abstains(text: str) -> bool:
    t = text.lower()
    return any(p in t for p in _ABSTAIN_PHRASES)


def load_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _claims_of(row: dict) -> list[dict]:
    judge = (row.get("verdict") or {}).get("judge")
    if isinstance(judge, dict):
        return judge.get("claims") or []
    return []


def systematic_or_stochastic(rows: list[dict]) -> dict:
    """Per-case: are the 5 runs' verdict labels identical?"""
    by_case: dict[str, list[str]] = defaultdict(list)
    for r in rows:
        by_case[r["case_id"]].append(r["verdict"].get("label", "error"))
    systematic, stochastic = 0, 0
    details = []
    for case, labels in by_case.items():
        uniq = set(labels)
        if len(uniq) == 1:
            systematic += 1
            status = "systematic"
        else:
            stochastic += 1
            status = "stochastic"
        details.append({"case_id": case, "status": status, "labels": sorted(labels)})
    return {"systematic_cases": systematic, "stochastic_cases": stochastic, "details": details}


def derive_severity(row: dict, label: str) -> str:
    """PROVISIONAL severity per annotation rules §4 (pending human annotation).

    Heuristic only — real severity must be assigned by the judge/annotators.
      critical: fabricated/running or destructive command, security-relevant
      major:    fabricated code, API/library, or debugging claim (breaks build)
      minor:    imprecise prose / version number
    """
    if label not in HALLUCINATED_LABELS:
        return "none"
    text = (row.get("response") or "").lower()
    cat = row.get("category", "")
    # conservative, transparent rule:
    if any(k in text for k in ["rm -rf", "drop table", "delete from", "password", "secret", "api key", "chmod 777", "curl | sh", "sudo "]):
        return "critical"
    if cat in ("coding", "apis_libraries", "debugging") or any(k in text for k in ["function", "method", "library", "import ", "api", ".py", ".js", "config"]):
        return "major"
    return "minor"


def compute_metrics(rows: list[dict], datasets: dict) -> dict:
    evaluated = [r for r in rows if r["verdict"].get("label") != "error"]
    N = len(evaluated)
    label_counter = Counter(r["verdict"].get("label") for r in evaluated)
    hallucinated = [r for r in evaluated if r["verdict"].get("label") in HALLUCINATED_LABELS]

    # Fabrication / unsupported from claim-level data (factual has claims).
    fabricated_rows = [r for r in evaluated if any(c.get("verdict") == "FABRICATED" for c in _claims_of(r))]
    unsupported_rows = [r for r in evaluated if any(c.get("verdict") == "UNSUPPORTED" for c in _claims_of(r))]

    # False confidence: overconfidence + high/medium certainty + not correct.
    false_conf = []
    for r in evaluated:
        j = (r.get("verdict") or {}).get("judge")
        if (r["verdict"].get("label") == "overconfidence"
                and isinstance(j, dict)
                and j.get("correct") is False
                and j.get("confidence_strength") in ("high", "medium")):
            false_conf.append(r)

    # Abstention accuracy over the unanswerable subset (expected abstentions).
    # A "correct abstention" for an unanswerable case = the judge's "ok" label:
    # there is no determinable answer, so "ok" means the model correctly
    # abstained. (A raw regex over the response is a weaker signal — it missed
    # "cannot provide" abstentions and counted an "I don't know…" that asserts
    # afterward — so the judge label is authoritative here, not the regex.)
    unanswerable = [r for r in evaluated if r.get("category") in EXPECTED_ABSTENTION_CATEGORIES]
    correct_abstentions = [r for r in unanswerable if r["verdict"].get("label") == "ok"]
    abstain_by_text = [r for r in unanswerable if _abstains(r.get("response") or "")]
    exp_abs = len(unanswerable)
    abst_ok = len(correct_abstentions)

    return {
        "evaluated": N,
        "hallucinated": len(hallucinated),
        "hallucination_rate": round(len(hallucinated) / N, 4) if N else None,
        "factual_accuracy": round(label_counter.get("ok", 0) / N, 4) if N else None,
        "fabrication_rate": round(len(fabricated_rows) / N, 4) if N else None,
        "fabricated_rows": len(fabricated_rows),
        "unsupported_claim_rate": round(len(unsupported_rows) / N, 4) if N else None,
        "unsupported_rows": len(unsupported_rows),
        "abstention_accuracy": round(abst_ok / exp_abs, 4) if exp_abs else None,
        "expected_abstentions": exp_abs,
        "correct_abstentions": abst_ok,
        "abstain_detected_by_text": len(abstain_by_text),
        "false_confidence_rate": round(len(false_conf) / N, 4) if N else None,
        "false_confidence_rows": len(false_conf),
        "by_label": dict(label_counter),
    }


def per_category_hr(rows: list[dict]) -> dict:
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r["verdict"].get("label") == "error":
            continue
        by_cat[r.get("category", "?")].append(r)
    out = {}
    for cat, rs in sorted(by_cat.items()):
        h = [r for r in rs if r["verdict"].get("label") in HALLUCINATED_LABELS]
        out[cat] = {"evaluated": len(rs), "hallucinated": len(h),
                    "hr": round(len(h) / len(rs), 4) if rs else None}
    return out


def severity_distribution(rows: list[dict]) -> dict:
    sev = Counter()
    per = []
    for r in rows:
        if r["verdict"].get("label") == "error":
            continue
        s = derive_severity(r, r["verdict"].get("label"))
        sev[s] += 1
        per.append({"case_id": r["case_id"], "run": r["run"], "category": r["category"],
                    "label": r["verdict"].get("label"), "severity": s})
    return {"distribution": dict(sev), "per_row": per}


def build_annotation_sample(rows: list[dict], size: int = 50) -> list[dict]:
    """Fixed-size labeled template for two human annotators (Ticket 3 AC2)."""
    # one representative response per case, spread across the three phenomena
    seen = set()
    picked = []
    for r in rows:
        if r["case_id"] in seen:
            continue
        seen.add(r["case_id"])
        picked.append(r)
    # fill remaining with the rest of the pool up to `size`
    pool = [r for r in rows if r not in picked]
    for r in pool:
        if len(picked) >= size:
            break
        picked.append(r)
    sample = []
    for r in picked[:size]:
        sample.append({
            "case_id": r["case_id"],
            "category": r["category"],
            "phenomenon": r["phenomenon"],
            "run": r["run"],
            "prompt": r["prompt"],
            "context": r.get("context", ""),
            "ground_truth": r.get("ground_truth", ""),
            "response": r["response"],
            "judge_label": r["verdict"].get("label"),
            "annotator_a": "",   # human label
            "annotator_b": "",   # human label
            "agreed": None,
        })
    return sample


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--evidence", required=True, help="evidence/hallucination-benchmark dir")
    ap.add_argument("--out", default="results-baseline")
    args = ap.parse_args()

    ev = args.evidence
    files = {
        "factual": "factual-20260901-124234.jsonl",
        "overconfidence": "overconfidence-20260901-125001.jsonl",
        "looping": "looping-20260902-130724.jsonl",
    }
    all_rows = []
    for key, fn in files.items():
        p = os.path.join(ev, fn)
        if not os.path.exists(p):
            print(f"WARN: missing {p}")
            continue
        all_rows.extend(load_jsonl(p))

    datasets = {}
    for d in ("factual", "overconfidence", "looping"):
        dp = os.path.join(ev, "datasets", f"{d}.jsonl")
        if os.path.exists(dp):
            datasets[d] = load_jsonl(dp)

    metrics = compute_metrics(all_rows, datasets)
    cats = per_category_hr(all_rows)
    s2 = systematic_or_stochastic(all_rows)
    sev = severity_distribution(all_rows)
    sample = build_annotation_sample(all_rows, 50)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "evidence_dir": ev,
        "total_rows": len(all_rows),
        "metrics": metrics,
        "per_category_hr": cats,
        "systematic_vs_stochastic": {k: v for k, v in s2.items() if k != "details"},
        "severity": sev["distribution"],
        "notes": [
            "severity is PROVISIONAL (heuristic) — real severity requires human/judge annotation",
            "fabrication/unsupported rates are partial: claim-level data only exists for factual",
            "abstention accuracy computed over the 'unanswerable' category only, using the judge 'ok' label as the correct-abstention signal (regex was too narrow — missed 'cannot provide' abstentions and counted an 'I don't know' that asserts afterward)",
            "metrics from AI-judge verdicts are preliminary until dual human annotation (AC2)",
        ],
    }

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "baseline-report.json"), "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    with open(os.path.join(args.out, "annotation-sample-50.jsonl"), "w") as f:
        for row in sample:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(os.path.join(args.out, "severity-per-row.jsonl"), "w") as f:
        for row in sev["per_row"]:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(os.path.join(args.out, "systematic-details.jsonl"), "w") as f:
        for row in s2["details"]:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nWrote outputs to {args.out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
