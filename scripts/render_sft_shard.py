"""
scripts/render_sft_shard.py

Render openagent-code's captured SFT trajectories into an alpha-dataset shard, so Arcus's
loader ingests them as pretraining text — the early "ramp" (raw text, no masking yet; the
masked chat-template form is a later concern when the SFT fraction grows past ~5%).

Source: openagent-code's converter output (`train/dataset/sft.jsonl`, per-step rows). We
reconstruct each SESSION's full conversation (ONE document per session, not per step — the
per-step rows repeat the prefix) by taking each session's last-step row (its `messages` is
the full prefix) plus its `completion`, then render to flat role-prefixed text.

Output: `alpha dataset/openagent_sft/shard-00000.jsonl.zst` — Arcus's `iter_shard_texts`
globs `**/*.jsonl.zst` and reads the `text` field, so it's picked up with NO loader change.

Re-run each round as the trajectory pool grows (the dataset gets bigger every cycle).

  python scripts/render_sft_shard.py
"""

from __future__ import annotations

import argparse
import io
import json
import os

DEFAULT_SFT = r"C:\Users\willi\OneDrive\Desktop\OpenCode\train\dataset\sft.jsonl"
DEFAULT_OUT = os.path.join("alpha dataset", "openagent_sft", "shard-00000.jsonl.zst")


def render_message(m: dict) -> str:
    """One chat message -> flat text. Role-tagged; tool calls inlined as text."""
    role = m.get("role", "?")
    parts = []
    if m.get("content"):
        parts.append(str(m["content"]))
    for tc in (m.get("tool_calls") or []):
        fn = tc.get("function", {})
        parts.append(f"[tool_call] {fn.get('name', '')}({fn.get('arguments', '')})")
    return f"<{role}>\n" + "\n".join(parts)


def render_session(messages, completion) -> str:
    msgs = list(messages or []) + ([completion] if completion else [])
    return "\n\n".join(render_message(m) for m in msgs)


def main() -> int:
    ap = argparse.ArgumentParser(description="Render captured SFT trajectories into an alpha-dataset shard")
    ap.add_argument("--sft", default=DEFAULT_SFT, help="openagent-code sft.jsonl (per-step rows)")
    ap.add_argument("--out", default=DEFAULT_OUT, help="output *.jsonl.zst shard under the alpha dataset")
    args = ap.parse_args()

    if not os.path.isfile(args.sft):
        print(f"no sft.jsonl at {args.sft} — run `python -m train.convert` in openagent-code first")
        return 1

    # Group per-step rows by session; keep the highest-step row per session (its `messages`
    # is the full conversation prefix, + its `completion` = the whole session).
    by_session = {}
    with open(args.sft, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            meta = row.get("meta") or {}
            sid = meta.get("session_id") or "unknown"
            step = meta.get("step") or 0
            cur = by_session.get(sid)
            if cur is None or step >= cur[0]:
                by_session[sid] = (step, row)

    docs = []
    for sid, (_step, row) in by_session.items():
        text = render_session(row.get("messages"), row.get("completion"))
        if text.strip():
            docs.append({"text": text, "source": "openagent_sft", "session_id": sid})

    import zstandard
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    cctx = zstandard.ZstdCompressor(level=10)
    with open(args.out, "wb") as fh, cctx.stream_writer(fh) as w:
        tw = io.TextIOWrapper(w, encoding="utf-8")
        for d in docs:
            tw.write(json.dumps(d, ensure_ascii=False) + "\n")
        tw.flush()

    chars = sum(len(d["text"]) for d in docs)
    print(f"rendered {len(docs)} session(s) -> {args.out}")
    print(f"  ~{chars/1e3:.1f}K chars  (~{chars/4/1e3:.1f}K tokens approx)  | source: {args.sft}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
