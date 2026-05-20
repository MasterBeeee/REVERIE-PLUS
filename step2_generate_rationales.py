"""
REVERIE+ Pipeline — Step 2: High-Fidelity Negative Rationale Distillation
==========================================================================
For each sample that now carries a hard negative answer (from Step 1), we
query Seed 1.6 (Doubao-Seed) to generate a detailed negative rationale that
explicitly explains *why* the incorrect answer is wrong based on visual
evidence in the image.  Seed 1.6 also acts as the consistency verifier in
Step 3; using the same strong model for both annotation and filtering ensures
coherent and high-fidelity supervision signals (Sec. 3.2 of the paper).

Usage
-----
    python step2_generate_rationales.py \
        --input  <path/to/wrong_answers.jsonl>   \
        --output <path/to/with_rationales.jsonl> \
        --workers 8
"""

from __future__ import annotations

import argparse
import base64
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from openai import OpenAI

# ---------------------------------------------------------------------------
# Annotator model — Seed 1.6 (high-fidelity rationale distillation)
#
# Replace the values below with the base URL and model identifier for the
# Seed 1.6 deployment you have access to.  The model must expose an
# OpenAI-compatible /v1/chat/completions endpoint.
# Set the environment variable RATIONALE_API_KEY before running.
# ---------------------------------------------------------------------------
RATIONALE_API_BASE = "<SEED_API_BASE>"         # replace with your base URL
RATIONALE_API_KEY_ENV = "RATIONALE_API_KEY"
RATIONALE_MODEL = "doubao-seed-1-6"            # replace with the exact tag you deploy

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------
PROMPT_TEMPLATE = """\
Task: You are provided with an image, a question about that image, the \
correct answer, and an incorrect answer.  Your objective is to generate a \
comprehensive rationale explaining exactly why the incorrect answer is wrong.

Instructions:
1. Visual Analysis: examine the image meticulously, identifying specific \
visual elements that directly relate to the question.
2. Evidence-Based Rebuttal: construct a logical argument that disproves the \
incorrect answer.  Cite specific details from the image as evidence.
3. Contextual Clarity: ensure your explanation is clear and concise.  If \
necessary, apply general knowledge to contextualise why the visual evidence \
contradicts the incorrect answer.
4. Base the analysis on the image itself — do not merely compare the \
incorrect answer with the correct one.
5. Output Format: provide the rationale only.  Do not include introductory \
text, greetings, or the correct answer in your output.

Input Data:
Question: {QUESTION}
Correct Answer: {CORRECT_ANSWER}
Incorrect Answer: {INCORRECT_ANSWER}

Rationale:
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_client() -> OpenAI:
    api_key = os.environ.get(RATIONALE_API_KEY_ENV, "")
    if not api_key:
        raise EnvironmentError(
            f"Environment variable '{RATIONALE_API_KEY_ENV}' is not set."
        )
    return OpenAI(api_key=api_key, base_url=RATIONALE_API_BASE)


def _image_to_data_url(image_path: str) -> str:
    ext = os.path.splitext(image_path)[1].lower()
    mime = {".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".png": "image/png", ".webp": "image/webp"}.get(ext, "image/jpeg")
    with open(image_path, "rb") as fh:
        b64 = base64.b64encode(fh.read()).decode()
    return f"data:{mime};base64,{b64}"


def _load_processed_ids(output_file: str) -> set:
    ids: set = set()
    if not os.path.exists(output_file):
        return ids
    with open(output_file, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if "id" in obj:
                    ids.add(obj["id"])
            except json.JSONDecodeError:
                pass
    return ids


# ---------------------------------------------------------------------------
# Core worker
# ---------------------------------------------------------------------------

_write_lock = Lock()
_client: OpenAI | None = None   # built once in main()


def _worker(task: dict, output_file: str) -> str:
    sample_id = task["id"]
    prompt = (PROMPT_TEMPLATE
              .replace("{QUESTION}", task["question"])
              .replace("{CORRECT_ANSWER}", task["correct_answer"])
              .replace("{INCORRECT_ANSWER}", task["incorrect_answer"]))
    try:
        img_data = _image_to_data_url(task["image_path"])
        resp = _client.chat.completions.create(
            model=RATIONALE_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url",
                         "image_url": {"url": img_data}},
                    ],
                }
            ],
            reasoning_effort="minimal",   # faster; adjust if higher quality needed
        )
        rationale = resp.choices[0].message.content.strip()
    except Exception as exc:
        print(f"[WARN] id={sample_id} error: {exc}")
        return f"Failed: {sample_id}"

    result = {
        "id": sample_id,
        "image": task["image_path"],
        "question": task["question"],
        "correct_answer": task["correct_answer"],
        "incorrect_answer": task["incorrect_answer"],
        "rationale": rationale,
    }
    with _write_lock:
        with open(output_file, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(result, ensure_ascii=False) + "\n")
    return f"Success: {sample_id}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global _client
    parser = argparse.ArgumentParser(
        description="Step 2 – negative rationale distillation via Seed 1.6"
    )
    parser.add_argument("--input", required=True,
                        help="Output of Step 1 (JSONL with distractors)")
    parser.add_argument("--output", required=True,
                        help="Output JSONL with rationales added")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of parallel threads (default: 8)")
    args = parser.parse_args()

    _client = _build_client()
    processed_ids = _load_processed_ids(args.output)
    print(f"Already processed: {len(processed_ids)} items.")

    tasks = []
    with open(args.input, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            sample_id = obj.get("id")
            if sample_id in processed_ids:
                continue

            correct_answer = obj.get("correct_answer", "")
            incorrect_answer = obj.get("distractor", "")
            if not correct_answer or not incorrect_answer:
                print(f"[SKIP] id={sample_id}: empty answer field")
                continue

            image_path = obj.get("image", "") or obj.get("image_path", "")
            if not image_path or not os.path.exists(image_path):
                print(f"[SKIP] id={sample_id}: image not found ({image_path})")
                continue

            tasks.append({
                "id": sample_id,
                "image_path": image_path,
                "question": obj.get("question", ""),
                "correct_answer": correct_answer,
                "incorrect_answer": incorrect_answer,
            })

    print(f"Tasks to process: {len(tasks)}  |  threads: {args.workers}")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_worker, t, args.output): t["id"] for t in tasks}
        for i, future in enumerate(as_completed(futures), 1):
            try:
                msg = future.result()
            except Exception as exc:
                msg = f"Exception: {exc}"
            if i % 50 == 0:
                print(f"Progress: {i}/{len(tasks)}  last={msg}")

    print("Step 2 complete.")


if __name__ == "__main__":
    main()
