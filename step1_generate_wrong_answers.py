"""
REVERIE+ Pipeline — Step 1: Diverse Negative Answer Mining
==========================================================
For each sample in the R1-Onevision data foundation, we query a committee of
strong LVLMs to generate a plausible but factually *incorrect* answer
(hard negative).  Using multiple models yields a broader spectrum of failure
modes than a single annotator, increasing both the diversity and the
difficulty of the resulting negative answers (as described in Sec. 3.2 of
the paper).

At inference time a model is selected **uniformly at random** from the
committee for each sample, so no single model dominates the negatives.

Usage
-----
    python step1_generate_wrong_answers.py \
        --input  <path/to/input.jsonl> \
        --output <path/to/wrong_answers.jsonl> \
        --workers 8
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import random
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from openai import OpenAI

# ---------------------------------------------------------------------------
# Committee of LVLMs used for diverse negative mining.
# Each entry: (api_base_url, api_key_env_var, model_name)
#
# We used four models in the paper: GPT-4o-mini, GLM-4V-Flash,
# Doubao-Seed-1.6-Flash, and Gemini-2.0-Flash.  Replace the entries below
# with whichever models you have access to.  All models must expose an
# OpenAI-compatible /v1/chat/completions endpoint.
#
# Set the corresponding environment variables before running.
# ---------------------------------------------------------------------------
MODEL_COMMITTEE = [
    (
        "https://api.openai.com/v1",          # OpenAI-compatible base URL
        "OPENAI_API_KEY",                      # env var holding the key
        "gpt-4o-mini",                         # model identifier
    ),
    (
        "<ZHIPU_API_BASE>",                    # replace with your Zhipu base URL
        "ZHIPU_API_KEY",
        "glm-4v-flash",
    ),
    (
        "<ARK_API_BASE>",                      # replace with your Ark base URL
        "ARK_API_KEY",
        "doubao-seed-1-6-flash",
    ),
    (
        "https://generativelanguage.googleapis.com/v1beta/openai/",
        "GEMINI_API_KEY",
        "gemini-2.0-flash",
    ),
]

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------
PROMPT_TEMPLATE = """\
You are provided with an image, a question related to that image, and the \
correct answer. Your task is to generate a factually incorrect answer to the \
question.

Instructions:
- Analyze the context: carefully examine the image and the question. The \
incorrect answer must be a plausible "distractor" that is difficult to \
distinguish from the truth.
- Multiple Choice: if the question is a multiple-choice problem, select the \
option most likely to trap a user, other than the correct one.
- Alignment: your generated answer must match the format of the correct \
answer (e.g., if the correct answer is a single word, yours must be too) and \
be semantically relevant to the question.
- Output: output only the generated incorrect answer. Do not include \
explanations or any other text.

QUESTION AND ANSWER:
{QUESTION}
{ANSWER}

OUTPUT:
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_client(base_url: str, api_key_env: str) -> OpenAI:
    api_key = os.environ.get(api_key_env, "")
    if not api_key:
        raise EnvironmentError(
            f"Environment variable '{api_key_env}' is not set."
        )
    return OpenAI(api_key=api_key, base_url=base_url)


def _image_to_data_url(image_path: str) -> str:
    ext = os.path.splitext(image_path)[1].lower()
    mime = {".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".png": "image/png", ".webp": "image/webp"}.get(ext, "image/jpeg")
    with open(image_path, "rb") as fh:
        b64 = base64.b64encode(fh.read()).decode()
    return f"data:{mime};base64,{b64}"


def _extract_answer_from_conversation(conversations: list[dict]) -> str:
    """Extract the clean answer text from the assistant turn."""
    if len(conversations) < 2:
        return ""
    raw = conversations[1].get("value", "") or ""
    # Strip <think>…</think> blocks produced by reasoning models
    answer = re.sub(r"<think>.*?</think>", "", raw, flags=re.S).strip()
    if not answer:
        m = re.search(r"Answer\s*:\s*(.*)$", raw, flags=re.S)
        if m:
            answer = m.group(1).strip()
    return answer


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


def _worker(task: dict, output_file: str) -> str:
    sample_id = task["id"]
    image_path = task["image_path"]
    question = task["question"]
    answer = task["answer"]

    # ---- random model selection from the committee ----
    base_url, api_key_env, model_name = random.choice(MODEL_COMMITTEE)
    client = _build_client(base_url, api_key_env)

    prompt = (PROMPT_TEMPLATE
              .replace("{QUESTION}", question)
              .replace("{ANSWER}", answer))
    try:
        img_data = _image_to_data_url(image_path)
        resp = client.chat.completions.create(
            model=model_name,
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
        )
        distractor = resp.choices[0].message.content.strip()
    except Exception as exc:
        print(f"[WARN] id={sample_id} model={model_name} error: {exc}")
        return f"Failed: {sample_id}"

    result = {
        "id": sample_id,
        "image": image_path,
        "question": question,
        "correct_answer": answer,
        "distractor": distractor,
    }
    with _write_lock:
        with open(output_file, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(result, ensure_ascii=False) + "\n")
    return f"Success: {sample_id}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Step 1 – diverse negative answer mining"
    )
    parser.add_argument("--input", required=True,
                        help="Path to the input JSONL file (R1-Onevision export)")
    parser.add_argument("--output", required=True,
                        help="Path to the output JSONL file")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of parallel threads (default: 8)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility (optional)")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

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

            image_path = obj.get("image_path") or obj.get("image", "")
            conversations = obj.get("conversations", [])
            question = ""
            if conversations and conversations[0].get("from") == "human":
                question = conversations[0].get("value", "")
            answer = _extract_answer_from_conversation(conversations)

            if not image_path or not os.path.exists(image_path):
                print(f"[SKIP] id={sample_id}: image not found ({image_path})")
                continue
            if not answer:
                print(f"[SKIP] id={sample_id}: empty answer")
                continue

            tasks.append({
                "id": sample_id,
                "image_path": image_path,
                "question": question,
                "answer": answer,
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

    print("Step 1 complete.")


if __name__ == "__main__":
    main()
