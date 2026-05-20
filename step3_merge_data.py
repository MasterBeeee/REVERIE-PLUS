"""
REVERIE+ Pipeline — Step 3: Merge Rationales into Conversation Records
=======================================================================
This script joins the original R1-Onevision source conversations (which
already carry positive answers and positive rationales) with the hard negative
answers and negative rationales produced in Steps 1–2.

Input
-----
- source JSONL  : R1-Onevision conversations, each record having at minimum
                  a two-turn conversation (human question → assistant answer).
- rationale JSONL: output of Step 2, keyed by the same sample id.

Output
------
A JSONL file where each record is a conversation extended with two extra
turns:
    turn 5 (human)    : "Explain why this is incorrect <incorrect_answer>"
    turn 6 (assistant): <negative_rationale>

Records whose source answer is malformed (empty after stripping <think>
blocks) are repaired or dropped; records without a matching rationale are
written as-is (positive-only).

Usage
-----
    python step3_merge_data.py \
        --source    <path/to/source.jsonl>    \
        --rationale <path/to/rationale.jsonl> \
        --output    <path/to/merged.jsonl>
"""

from __future__ import annotations

import argparse
import json
import re
from copy import deepcopy
from typing import Any

# ---------------------------------------------------------------------------
# Regex helpers
# ---------------------------------------------------------------------------

_THINK_RE = re.compile(r"<think>.*?</think>", flags=re.S)
_ANSWER_RE = re.compile(r"Answer\s*:\s*(.*)$", flags=re.S)


def _is_nonempty(x: Any) -> bool:
    return isinstance(x, str) and x.strip() != ""


def _answer_without_think(raw: str) -> str:
    return _THINK_RE.sub("", raw).strip()


# ---------------------------------------------------------------------------
# Conversation schema detection
# ---------------------------------------------------------------------------

def _schema(obj: dict) -> tuple[str, list, str, str]:
    """Return (messages_key, messages_list, role_key, text_key)."""
    if isinstance(obj.get("conversations"), list):
        return "conversations", obj["conversations"], "from", "value"
    if isinstance(obj.get("messages"), list):
        return "messages", obj["messages"], "role", "content"
    return "conversations", [], "from", "value"


# ---------------------------------------------------------------------------
# Answer repair / validation
# ---------------------------------------------------------------------------

def _validate_and_repair(obj: dict) -> dict | None:
    """
    Return a (possibly repaired) copy, or None if the record must be dropped.

    The second turn (assistant) may contain a raw chain-of-thought wrapped in
    <think>…</think>.  We strip that block; if the remainder is empty we try
    to extract an 'Answer:' line and append it.
    """
    msgs_key, msgs, role_key, text_key = _schema(obj)
    if len(msgs) < 2:
        return None
    if msgs[0].get(role_key) != "human":
        return None
    if msgs[1].get(role_key) not in ("assistant", "gpt"):
        return None

    raw = msgs[1].get(text_key, "")
    if not isinstance(raw, str):
        return None

    answer = _answer_without_think(raw)
    if _is_nonempty(answer):
        return obj  # normal case

    # Try to recover an 'Answer:' tail
    m = _ANSWER_RE.search(raw)
    if not m:
        return None
    extracted = m.group(1).strip().replace("</think>", "").strip()
    if not extracted:
        return None

    # Append the recovered answer after the chain-of-thought block
    repaired = deepcopy(obj)
    _, r_msgs, r_role_key, r_text_key = _schema(repaired)
    r_msgs[1][r_text_key] = raw.rstrip() + "\nAnswer: " + extracted
    return repaired


# ---------------------------------------------------------------------------
# Rationale appending
# ---------------------------------------------------------------------------

def _append_negative_rationale(
    source_obj: dict,
    incorrect_answer: str,
    rationale: str,
) -> dict:
    out = deepcopy(source_obj)
    msgs_key, msgs, role_key, text_key = _schema(out)

    human_text = f"Explain why this is incorrect: {incorrect_answer}".strip()
    msgs.append({role_key: "human", text_key: human_text})
    msgs.append({role_key: "assistant", text_key: rationale})
    return out


# ---------------------------------------------------------------------------
# Load rationale map
# ---------------------------------------------------------------------------

def _load_rationale_map(path: str) -> dict[str, dict]:
    mp: dict[str, dict] = {}
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"JSON decode error in {path} at line {lineno}: {exc}"
                ) from exc
            if "id" in obj:
                mp[str(obj["id"])] = obj
    return mp


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Step 3 – merge negative rationales into conversations"
    )
    parser.add_argument("--source", required=True,
                        help="Source JSONL (R1-Onevision export)")
    parser.add_argument("--rationale", required=True,
                        help="Rationale JSONL (output of Step 2)")
    parser.add_argument("--output", required=True,
                        help="Merged output JSONL")
    args = parser.parse_args()

    rationale_map = _load_rationale_map(args.rationale)

    total = kept = dropped = repaired = matched = appended = 0

    with open(args.source, "r", encoding="utf-8") as fin, \
         open(args.output, "w", encoding="utf-8") as fout:

        for lineno, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue
            try:
                src = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"JSON decode error in {args.source} at line {lineno}: {exc}"
                ) from exc

            total += 1
            out = _validate_and_repair(src)
            if out is None:
                dropped += 1
                continue
            if out is not src:
                repaired += 1

            # Look up the pre-generated rationale for this sample
            sample_id = str(out.get("id", ""))
            rat_entry = rationale_map.get(sample_id)
            if rat_entry is not None:
                matched += 1
                rationale = rat_entry.get("rationale", "")
                incorrect = rat_entry.get("incorrect_answer", "")
                if _is_nonempty(rationale):
                    out = _append_negative_rationale(
                        out,
                        incorrect_answer=incorrect.strip() if _is_nonempty(incorrect) else "",
                        rationale=rationale.strip(),
                    )
                    appended += 1

            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            kept += 1

    print(json.dumps({
        "total_source_rows": total,
        "kept_rows": kept,
        "dropped_rows": dropped,
        "repaired_rows": repaired,
        "matched_rationale_ids": matched,
        "appended_negative_rationale": appended,
        "output": args.output,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
