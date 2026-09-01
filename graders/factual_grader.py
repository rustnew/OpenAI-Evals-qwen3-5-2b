"""Factual grader for the Qwen3.5-2B hallucination study.

Checks a response's atomic claims against the case's ground truth, per the
annotation rules. For simple cases this is a deterministic containment/equality
check; open responses are flagged for model-graded review (Ticket 3).

The `ground_truth` is the load-bearing field from the dataset.
"""

from __future__ import annotations

import re
from typing import Optional


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def grade(response: str, ground_truth: str, context: str = "") -> dict:
    """Return a factual verdict for a response against the ground truth.

    Returns: {label, supported, fabricated, unsupported, reason}
    """
    r = _norm(response)
    gt = _norm(ground_truth)

    # A response that abstains (says it doesn't know) is NOT a false assertion.
    if _is_abstention(r):
        return {
            "label": "ok",
            "supported": False,
            "fabricated": False,
            "unsupported": False,
            "reason": "abstention",
        }

    # Deterministic check: does the response contain the ground truth?
    # (Simple cases only. Open responses -> model-graded in Ticket 3.)
    if gt and gt in r:
        return {
            "label": "ok",
            "supported": True,
            "fabricated": False,
            "unsupported": False,
            "reason": "ground_truth_present",
        }

    # Ground truth explicitly says the thing does NOT exist / is false.
    # Only flag false-factual if the response POSITIVELY asserts the denied
    # thing (e.g. "dict.clone() exists" / "use dict.clone()"), not merely
    # because it contains common words like "is"/"does".
    if _is_negation(gt) and _asserts_denied_thing(r, gt):
        return {
            "label": "false_factual_assertion",
            "supported": False,
            "fabricated": True,
            "unsupported": True,
            "reason": "asserts_what_ground_truth_denies",
        }

    # Otherwise: not deterministically verifiable -> flag for review.
    return {
        "label": "needs_review",
        "supported": False,
        "fabricated": None,
        "unsupported": None,
        "reason": "not_deterministically_verifiable",
    }


def _is_abstention(r: str) -> bool:
    return any(
        p in r
        for p in [
            "i don't know",
            "i do not know",
            "i don't have that information",
            "i cannot determine",
            "not enough information",
            "i'm not sure",
            "i am not sure",
        ]
    )


def _is_negation(gt: str) -> bool:
    return any(p in gt for p in ["does not exist", "no —", "no,", "does not have", "not determinable"])


def _asserts_denied_thing(r: str, gt: str) -> bool:
    """True if the response positively asserts the thing the ground truth denies.

    Extracts the denied subject from the ground truth (e.g. 'dict.clone()' from
    'dict.clone() does not exist') and checks whether the response asserts it
    exists / recommends using it.
    """
    denied = None
    for marker in [" does not exist", " does not have", " is not a", " does not"]:
        idx = gt.find(marker)
        if idx != -1:
            # the denied subject is the clause immediately before the marker
            clause_start = max(
                gt.rfind(". ", 0, idx), gt.rfind("; ", 0, idx), gt.rfind(": ", 0, idx)
            ) + 1
            denied = gt[clause_start:idx].strip()
            break
    if not denied:
        return False
    # normalize: strip quotes/backticks/punctuation
    denied_norm = denied.strip("`'\".,;:()[]{}")
    rn = r
    # does the response claim the denied thing exists or recommends it?
    return (
        denied_norm in rn
        and any(p in rn for p in ["exists", " is a", " use ", " has ", " provides ", " supports "])
    )


if __name__ == "__main__":
    print(grade("Use copy.deepcopy() from the copy module.", "Use copy.deepcopy() from the copy module."))
    print(grade("Use dict.clone() to copy it.", "Use copy.deepcopy() from the copy module."))
    print(grade("I don't know.", "Guido van Rossum."))
