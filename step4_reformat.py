"""
REVERIE+ Pipeline — Step 4: Reformat & Quality Filter
======================================================
Normalises every conversation produced by Step 3 into a canonical multi-turn
format and removes low-quality samples.

Target formats
--------------
4-turn (positive supervision only):
    [user] question
    [assistant] answer
    [user] "Explain why"
    [assistant] positive rationale

6-turn (positive + negative supervision):
    [user] question
    [assistant] answer
    [user] "Explain why"
    [assistant] positive rationale
    [user] "Explain why this is incorrect: <incorrect_answer>"
    [assistant] negative rationale

Cleaning steps applied to each field
--------------------------------------
- question  : strip chain-of-thought instructions, normalise whitespace
- answer    : extract the final answer value (option letter, number, phrase)
- positive rationale: extracted from <think>…</think> blocks or explicit
  "Reasoning:" sections; trimmed of label prefixes
- negative rationale: same cleaning, with preference for labelled blocks

Quality filter
--------------
Samples are dropped if the positive rationale is empty or identical to the
answer (string-normalised), as these offer no additional supervision signal.

Usage
-----
    python step4_reformat.py \
        --input   <path/to/merged.jsonl or .json> \
        --output  <path/to/reformatted_all.json>  \
        --output-complete <path/to/reformatted_6turn.json>
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

# ---------------------------------------------------------------------------
# Compiled patterns
# ---------------------------------------------------------------------------

_THINK_TAG_RE = re.compile(
    r"<\s*(think[\w\-]*)[^>]*>(.*?)<\s*/\s*\1\s*>",
    re.IGNORECASE | re.DOTALL,
)
_GENERIC_TAG_RE = re.compile(
    r"</?\s*(think[\w\-]*|analysis|reasoning)\b[^>]*>",
    re.IGNORECASE,
)
_MULTI_BLANK_RE = re.compile(r"\n{3,}")
_LABEL_LINE_RE = re.compile(
    r"^\s*(?:\*\*|__)?\s*(?:final\s+)?(?:answer|rationale|output|solution|conclusion)"
    r"\s*[:：]\s*(?:\*\*|__)?\s*(.*)$",
    re.IGNORECASE,
)

_QUESTION_DROP_PATTERNS = [
    re.compile(r"^\s*first\s+perform\s+reasoning.*$", re.IGNORECASE),
    re.compile(r"^\s*let'?s\s+think\s+step\s+by\s+step\.?\s*$", re.IGNORECASE),
    re.compile(r"^\s*think\s+step\s+by\s+step\.?\s*$", re.IGNORECASE),
    re.compile(r"^\s*reason(?:ing)?\s+before\s+answer.*$", re.IGNORECASE),
    re.compile(r"^\s*please\s+think.*before.*answer.*$", re.IGNORECASE),
    re.compile(r"^\s*(show|provide|explain)\s+.*reasoning.*$", re.IGNORECASE),
    re.compile(r"^\s*answer\s+with\s+.*directly\.?\s*$", re.IGNORECASE),
    re.compile(r"^\s*respond\s+with\s+.*directly\.?\s*$", re.IGNORECASE),
    re.compile(r"^\s*output\s+format\s*:?.*$", re.IGNORECASE),
]


# ---------------------------------------------------------------------------
# Text normalisation utilities
# ---------------------------------------------------------------------------

def _norm_nl(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _strip_known_tags(text: str) -> str:
    text = _THINK_TAG_RE.sub(lambda m: m.group(2), text)
    return _GENERIC_TAG_RE.sub("", text)


def _strip_nested_labels(text: str) -> str:
    prev, cur = None, text.strip()
    while prev != cur:
        prev = cur
        m = _LABEL_LINE_RE.match(cur)
        if m:
            cur = m.group(1).strip()
    return cur


def _pick_last_labeled_block(text: str) -> str:
    for pat_str in [
        r"(?is)(?:^|\n)\s*(?:\*\*)?(?:final\s+)?rationale\s*[:：]\s*(.+?)\s*"
        r"(?=\n\s*(?:\*\*)?(?:final\s+)?(?:rationale|answer|output)\s*[:：]|\Z)",
        r"(?is)(?:^|\n)\s*(?:\*\*)?(?:final\s+)?answer\s*[:：]\s*(.+?)\s*"
        r"(?=\n\s*(?:\*\*)?(?:final\s+)?(?:rationale|answer|output)\s*[:：]|\Z)",
    ]:
        matches = list(re.finditer(pat_str, text))
        if matches:
            return matches[-1].group(1).strip()
    return text.strip()


def clean_question(text: str) -> str:
    text = _norm_nl(text).strip()
    has_img = text.startswith("<image>")
    if has_img:
        text = text[len("<image>"):].lstrip()
    lines = []
    for raw in text.split("\n"):
        line = raw.strip()
        if not line:
            lines.append("")
            continue
        if line.lower().startswith("question:"):
            line = line.split(":", 1)[1].strip()
        if any(p.match(line) for p in _QUESTION_DROP_PATTERNS):
            continue
        lines.append(line)
    cleaned = _MULTI_BLANK_RE.sub("\n\n", "\n".join(lines).strip())
    return ("<image>" + cleaned if has_img else cleaned).strip()


def clean_rationale(text: str, prefer_labeled: bool = True) -> str:
    text = _norm_nl(text)
    text = _strip_known_tags(text).replace("\u2019", "'")
    if prefer_labeled:
        text = _pick_last_labeled_block(text)
    out_lines = []
    for line in text.split("\n"):
        m = _LABEL_LINE_RE.match(line)
        if m:
            rest = _strip_nested_labels(m.group(1))
            if rest:
                out_lines.append(rest)
            continue
        s = line.strip()
        if re.fullmatch(r"\(.*\)", s):
            continue
        if re.fullmatch(r"(?i)(yes|no|that's it|that's the answer|perfect|done)[\.! ]*", s):
            continue
        out_lines.append(line)
    return _MULTI_BLANK_RE.sub("\n\n", "\n".join(out_lines).strip())


def _extract_final_value(text: str) -> str | None:
    boxed = re.findall(r"\\boxed\{([^{}]+)\}", text)
    if boxed:
        return boxed[-1].strip()
    m = re.search(r"(?is)final\s+answer\s*[:：]\s*(.+)$", text)
    if m:
        tail = m.group(1).strip().split("\n", 1)[0].strip()
        tail = _strip_nested_labels(tail)
        if nums := re.findall(r"[-+]?\d+(?:\.\d+)?", tail):
            return nums[-1]
        if opt := re.search(r"\b([A-H])\b", tail):
            return opt.group(1)
        if tail:
            return tail[:100]
    return None


def _norm_answer(value: str) -> str:
    v = _strip_nested_labels(value).strip().strip("`*\"' ")
    if m := re.fullmatch(r"([A-Ha-h])\s*[\.\):：\-]\s*.+", v):
        return m.group(1).upper()
    if m := re.fullmatch(r"([A-Ha-h])\.?", v):
        return m.group(1).upper()
    return v.rstrip(".")


def clean_answer(text: str) -> str:
    text = _norm_nl(text)
    text = _strip_known_tags(text).strip()
    if fv := _extract_final_value(text):
        return _norm_answer(fv)
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    for i, ln in enumerate(lines):
        m = _LABEL_LINE_RE.match(ln)
        if not m:
            continue
        lc = _norm_answer(m.group(1))
        if lc:
            return lc
        for nxt in lines[i + 1:]:
            nv = _norm_answer(nxt)
            if nv:
                return nv
    labeled = re.findall(r"(?is)(?:final\s+answer|answer)\s*[:：]\s*([^\n]{1,120})", text)
    if labeled:
        cand = _norm_answer(labeled[-1])
        if cand:
            return cand
    if lines:
        last = _norm_answer(lines[-1])
        if re.fullmatch(r"[A-Za-z]\.?", last):
            return last[0]
        if re.fullmatch(r"[A-Za-z0-9_\-+/=.]{1,20}", last):
            return last
        if nums := re.findall(r"[-+]?\d+(?:\.\d+)?", last):
            if len(last) <= 120:
                return nums[-1]
        if len(last) <= 80:
            return last
    text = re.sub(r"(?im)^\s*(?:final\s+answer|answer|output)\s*[:：]\s*", "", text).strip()
    return _norm_answer(_MULTI_BLANK_RE.sub("\n\n", text))


def _split_first_assistant(content: str) -> tuple[str, str]:
    """Extract (positive_rationale, answer) from a raw chain-of-thought turn."""
    content = _norm_nl(content).strip()
    think_parts = [m.group(2).strip()
                   for m in _THINK_TAG_RE.finditer(content) if m.group(2).strip()]
    rationale = "\n\n".join(think_parts).strip()
    without_think = _THINK_TAG_RE.sub("", content).strip()
    answer = without_think
    if not rationale:
        m = re.search(
            r"(?is)(?:reasoning|analysis)\s*[:：]\s*(.+?)\n+\s*"
            r"(?:final\s+answer|answer)\s*[:：]\s*(.+)$",
            content,
        )
        if m:
            rationale, answer = m.group(1).strip(), m.group(2).strip()
    return clean_rationale(rationale, prefer_labeled=False), clean_answer(answer)


def _build_pos_rationale_fallback(raw: str, answer: str) -> str:
    text = _norm_nl(_strip_known_tags(raw)).strip()
    text = re.sub(r"(?im)^\s*(?:final\s+answer|answer|output)\s*[:：]\s*", "", text)
    lines = text.split("\n")
    while lines and _norm_answer(lines[-1]).strip().lower() == _norm_answer(answer).strip().lower():
        lines.pop()
    text = clean_rationale("\n".join(lines).strip(), prefer_labeled=False)
    return text or _norm_answer(answer)


def _clean_user_prompt(text: str) -> str:
    text = _norm_nl(text)
    text = _strip_known_tags(text)
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------------------
# Schema helper
# ---------------------------------------------------------------------------

def _schema(obj: dict):
    if isinstance(obj.get("messages"), list):
        return "messages", obj["messages"], "role", "content"
    return "conversations", obj.get("conversations", []), "from", "value"


# ---------------------------------------------------------------------------
# Record conversion
# ---------------------------------------------------------------------------

def _with_meta(obj: dict, new_messages: list) -> dict:
    out = {"messages": new_messages}
    if "images" in obj:
        out["images"] = obj["images"]
    for k, v in obj.items():
        if k not in out and k not in {"messages", "images", "conversations"}:
            out[k] = v
    return out


def convert_record(obj: dict) -> tuple[dict, bool]:
    msgs_key, msgs, role_key, text_key = _schema(obj)
    n = len(msgs)
    if n < 2:
        return obj, False

    def _get(i: int) -> str:
        return msgs[i].get(text_key, "") if i < n else ""

    if n == 2:
        q = clean_question(_get(0))
        pos_rat, answer = _split_first_assistant(_get(1))
        if not pos_rat.strip():
            pos_rat = _build_pos_rationale_fallback(_get(1), answer)
        new_msgs = [
            {role_key: "user", text_key: q},
            {role_key: "assistant", text_key: answer},
            {role_key: "user", text_key: "Explain why"},
            {role_key: "assistant", text_key: pos_rat},
        ]
        return _with_meta(obj, new_msgs), True

    if n >= 6:
        q = clean_question(_get(0))
        answer = clean_answer(_get(1))
        pos_rat = clean_rationale(_get(3), prefer_labeled=False)
        if not pos_rat.strip():
            pos_rat = _build_pos_rationale_fallback(_get(1), answer)
        wrong_q = _clean_user_prompt(_get(4))
        neg_rat = clean_rationale(_get(5), prefer_labeled=True)
        if wrong_q.strip() and neg_rat.strip():
            new_msgs = [
                {role_key: "user", text_key: q},
                {role_key: "assistant", text_key: answer},
                {role_key: "user", text_key: "Explain why"},
                {role_key: "assistant", text_key: pos_rat},
                {role_key: "user", text_key: wrong_q},
                {role_key: "assistant", text_key: neg_rat},
            ]
            return _with_meta(obj, new_msgs), True
        new_msgs = [
            {role_key: "user", text_key: q},
            {role_key: "assistant", text_key: answer},
            {role_key: "user", text_key: "Explain why"},
            {role_key: "assistant", text_key: pos_rat},
        ]
        return _with_meta(obj, new_msgs), True

    # n == 4
    q = clean_question(_get(0))
    turn2_user = _clean_user_prompt(_get(2))
    answer = clean_answer(_get(1))
    if turn2_user.strip().lower() == "explain why":
        # Already in canonical 4-turn form
        pos_rat = clean_rationale(_get(3), prefer_labeled=False)
        if not pos_rat.strip():
            pos_rat = _build_pos_rationale_fallback(_get(1), answer)
        new_msgs = [
            {role_key: "user", text_key: q},
            {role_key: "assistant", text_key: answer},
            {role_key: "user", text_key: "Explain why"},
            {role_key: "assistant", text_key: pos_rat},
        ]
        return _with_meta(obj, new_msgs), True

    # Raw 4-turn: assistant turn contains full chain-of-thought + answer
    pos_rat, answer_from_raw = _split_first_assistant(_get(1))
    answer = clean_answer(answer_from_raw)
    if not pos_rat.strip():
        pos_rat = _build_pos_rationale_fallback(_get(1), answer)
    wrong_q = turn2_user
    neg_rat = clean_rationale(_get(3), prefer_labeled=True)
    if wrong_q.strip() and neg_rat.strip():
        new_msgs = [
            {role_key: "user", text_key: q},
            {role_key: "assistant", text_key: answer},
            {role_key: "user", text_key: "Explain why"},
            {role_key: "assistant", text_key: pos_rat},
            {role_key: "user", text_key: wrong_q},
            {role_key: "assistant", text_key: neg_rat},
        ]
        return _with_meta(obj, new_msgs), True
    new_msgs = [
        {role_key: "user", text_key: q},
        {role_key: "assistant", text_key: answer},
        {role_key: "user", text_key: "Explain why"},
        {role_key: "assistant", text_key: pos_rat},
    ]
    return _with_meta(obj, new_msgs), True


# ---------------------------------------------------------------------------
# Quality filter
# ---------------------------------------------------------------------------

def _norm_cmp(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (text or "").lower())


def _is_bad_positive(sample: dict) -> bool:
    msgs = sample.get("messages", [])
    if len(msgs) < 4:
        return True
    pos = (msgs[3].get("content") or "").strip()
    if not pos:
        return True
    ans = (msgs[1].get("content") or "").strip()
    return _norm_cmp(ans) == _norm_cmp(pos)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Step 4 – reformat conversations and apply quality filter"
    )
    parser.add_argument("--input", required=True,
                        help="Merged JSONL or JSON file (output of Step 3)")
    parser.add_argument("--output", required=True,
                        help="Output JSON (all kept samples)")
    parser.add_argument("--output-complete", default=None,
                        help="Output JSON (only 4-/6-turn converted samples)")
    parser.add_argument("--indent", type=int, default=2)
    args = parser.parse_args()

    in_path = Path(args.input)
    # Accept both .json (list) and .jsonl (line-delimited)
    text = in_path.read_text(encoding="utf-8").strip()
    if text.startswith("["):
        data = json.loads(text)
    else:
        data = [json.loads(ln) for ln in text.splitlines() if ln.strip()]

    converted_all, converted_complete = [], []
    n_converted = 0
    for item in data:
        out, was_converted = convert_record(item)
        converted_all.append(out)
        if was_converted:
            n_converted += 1
            converted_complete.append(out)

    before = len(converted_all)
    converted_all = [x for x in converted_all if not _is_bad_positive(x)]
    converted_complete = [x for x in converted_complete if not _is_bad_positive(x)]

    Path(args.output).write_text(
        json.dumps(converted_all, ensure_ascii=False, indent=args.indent),
        encoding="utf-8",
    )
    if args.output_complete:
        Path(args.output_complete).write_text(
            json.dumps(converted_complete, ensure_ascii=False, indent=args.indent),
            encoding="utf-8",
        )

    print(f"Total input   : {len(data)}")
    print(f"Converted     : {n_converted}")
    print(f"Unchanged     : {len(data) - n_converted}")
    print(f"Quality-dropped: {before - len(converted_all)}")
    print(f"Final output  : {len(converted_all)}")
    if args.output_complete:
        print(f"Complete (4-/6-turn): {len(converted_complete)}")


if __name__ == "__main__":
    main()
