"""Deterministic repetition-loop detector for the Qwen3.5-2B hallucination study.

Detects abnormal repeated spans (n-gram loops) in a generated response, per the
annotation rules (inference-ops `hallucination-annotation-rules.md`).

A loop is flagged when a contiguous n-gram (n >= 2) repeats consecutively at
least `min_repeats` times (default 3), i.e. the same span appears 3+ times in a
row, replacing continuation/termination.
"""

from __future__ import annotations

import re
from typing import Optional

# Markdown table separator rows (e.g. "| :--- | :--- |") and pure-formatting
# tokens are NOT evidence of a repetition loop — they legitimately repeat in
# tables. Strip them before n-gram analysis to avoid false positives.
_TABLE_SEP_RE = re.compile(r"^\s*\|?[\s:|-]+\|?\s*$")
_FORMAT_TOKENS = {"|", "---", ":---", "---:", ":---:", "===", "```", "**", "__"}


def tokenize(text: str) -> list[str]:
    """Split into whitespace-delimited tokens, dropping pure-formatting tokens
    and markdown table-separator rows (deterministic)."""
    tokens = []
    for line in text.splitlines():
        if _TABLE_SEP_RE.match(line):
            continue
        for t in re.split(r"\s+", line.strip()):
            if not t:
                continue
            # drop a token if it is pure formatting (possibly wrapped in pipes)
            core = t.strip("|")
            if core in _FORMAT_TOKENS or core == "":
                continue
            tokens.append(t)
    return tokens


def find_loop(
    text: str,
    min_span: int = 2,
    max_span: int = 12,
    min_repeats: int = 3,
) -> Optional[dict]:
    """Return loop info if an abnormal repeated span is found, else None.

    Returns a dict with: loop_detected, sequence_length, repetitions,
    start_token, repeated_span.
    """
    tokens = tokenize(text)
    n = len(tokens)
    for span_len in range(min_span, max_span + 1):
        # slide a window; check if the same span repeats consecutively
        i = 0
        while i + span_len <= n:
            span = tuple(tokens[i : i + span_len])
            # count consecutive repeats starting at i
            reps = 1
            j = i + span_len
            while j + span_len <= n and tuple(tokens[j : j + span_len]) == span:
                reps += 1
                j += span_len
            if reps >= min_repeats:
                return {
                    "loop_detected": True,
                    "sequence_length": span_len,
                    "repetitions": reps,
                    "start_token": i,
                    "repeated_span": " ".join(span),
                }
            i += 1
    return None


def grade(text: str, ground_truth: str = "", context: str = "") -> dict:
    """Public grading entrypoint used by the eval.

    `ground_truth` and `context` are accepted for a uniform signature with the
    other graders but are not used by the deterministic loop detector.
    """
    loop = find_loop(text)
    if loop is None:
        return {"loop_detected": False, "label": "ok"}
    return {**loop, "label": "repetition_loop"}


if __name__ == "__main__":
    # quick self-test
    ok = "The Ingress controller routes external traffic to Services."
    loop = "A B C A B C A B C A B C"
    print("ok   ->", grade(ok))
    print("loop ->", grade(loop))
