"""Local test for the Ticket 4 sampling-params extension.

Verifies that EVALS_TOP_P / EVALS_TOP_K / EVALS_THINKING are (a) passed to the
OpenAI client and (b) recorded per row + summary — using a FAKE OpenAI client,
so NO network call and NO gateway key are needed. Run:  python3 tests/test_sampling_params.py
"""

import json
import os
import sys
import tempfile
import types

# ── Inject a fake `openai` module BEFORE importing the runner ───────────────
class _FakeCompletions:
    def __init__(self, captured):
        self._captured = captured
    def create(self, **kwargs):
        self._captured.append(kwargs)
        class _Usage: pass
        usage = _Usage()
        usage.prompt_tokens = 10
        usage.completion_tokens = 5
        usage.total_tokens = 15
        class _Msg: pass
        m = _Msg(); m.content = "ok"
        class _Ch: pass
        c = _Ch(); c.message = m; c.finish_reason = "stop"
        class _Resp: pass
        r = _Resp(); r.choices = [c]; r.usage = usage
        return r

class _FakeChat:
    def __init__(self, captured):
        self.completions = _FakeCompletions(captured)

class _FakeOpenAI:
    def __init__(self, **kwargs):
        self.captured = []
        self.chat = _FakeChat(self.captured)

fake = types.ModuleType("openai")
fake.OpenAI = _FakeOpenAI
sys.modules["openai"] = fake

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "runner"))
import run_evals  # noqa: E402


def _case_file():
    return os.path.join(tempfile.mkdtemp(), "factual.jsonl"), \
        {"id": "FACT-T", "category": "factual", "phenomenon": "false_factual_assertion",
         "prompt": "Q?", "context": "", "ground_truth": "A"}


def main():
    fails = 0

    # 1) call_model passes the knobs and records them
    client = _FakeOpenAI()
    case = {"id": "C1", "prompt": "p", "context": "ctx", "ground_truth": "gt"}
    out = run_evals.call_model(client, "m", case, 0.7, 512, 42, top_p=0.9, top_k=20, enable_thinking=True)
    kwargs = client.captured[-1]
    assert kwargs.get("top_p") == 0.9, f"top_p not passed: {kwargs}"
    assert kwargs.get("extra_body") == {"top_k": 20, "enable_thinking": True}, f"extra_body: {kwargs.get('extra_body')}"
    assert out.get("response") == "ok"
    print("PASS call_model passes top_p + extra_body(top_k, enable_thinking) and returns response")

    # 2) unset knobs are NOT passed (baseline = server default)
    client.captured.clear()
    run_evals.call_model(client, "m", case, 0.7, 512, 42)
    kwargs = client.captured[-1]
    assert "top_p" not in kwargs and "extra_body" not in kwargs, f"baseline leaked knobs: {kwargs}"
    print("PASS call_model baseline (unset) sends no top_p/extra_body")

    # 3) end-to-end: env → main() → rows record the knobs (fake client, no network)
    fpath, row = _case_file()
    with open(fpath, "w") as f:
        f.write(json.dumps(row) + "\n")
    outdir = tempfile.mkdtemp()
    os.environ.update({
        "OPENAI_BASE_URL": "http://fake", "OPENAI_API_KEY": "fake-key",
        "EVALS_MODEL": "m", "EVALS_TEMPERATURE": "0.7",
        "EVALS_TOP_P": "0.9", "EVALS_TOP_K": "20", "EVALS_THINKING": "true",
    })
    rc = run_evals.main.__wrapped__ if hasattr(run_evals.main, "__wrapped__") else None
    # call main() with argv; it reads env + args
    old_argv = sys.argv
    sys.argv = ["run_evals.py", "--dataset", fpath, "--out", outdir, "--runs", "2", "--condition", "topk-20"]
    try:
        code = run_evals.main()
    finally:
        sys.argv = old_argv
    assert code == 0, f"main() exited {code}"
    rows = []
    for fn in os.listdir(outdir):
        if fn.endswith(".jsonl"):
            with open(os.path.join(outdir, fn)) as f:
                rows = [json.loads(l) for l in f if l.strip()]
    assert rows, "no output rows"
    for r in rows:
        assert r["top_p"] == 0.9 and r["top_k"] == 20 and r["enable_thinking"] is True, f"row not recorded: {r}"
        assert r["condition"] == "topk-20", f"condition not recorded: {r}"
        assert "fingerprint" in r, f"fingerprint missing: {r}"
    print(f"PASS end-to-end: {len(rows)} rows record top_p=0.9 top_k=20 thinking=true condition=fingerprint")

    print("\nALL TESTS PASSED (local, no network)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
