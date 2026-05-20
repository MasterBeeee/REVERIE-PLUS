# REVERIE+ Data Construction Pipeline

<p align="center">
  <a href="https://huggingface.co/datasets/YOUR_HF_DATASET"><img src="https://img.shields.io/badge/Dataset-HuggingFace-yellow" alt="Dataset"></a>
  <a href="https://github.com/MasterBeeee/REVERIE-PLUS"><img src="https://img.shields.io/badge/GitHub-REVERIE--PLUS-blue" alt="GitHub"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-green" alt="License"></a>
</p>

This repository provides the data construction pipeline for the **REVERIE+** dataset introduced in:

> **REVERIE+: Generalized Reflective Instruction Tuning for Hallucination Mitigation in Advanced VLMs**
> Mingyang Bi*, Jinrui Zhang*, Xiangchen Wang, Xue Jiang, Yuhang Lu, Peng Wang, Feng Zheng
> *Preprint, 2025 (under review)*
> \* Equal contribution

REVERIE+ extends the [REVERIE](https://github.com/zjr2000/REVERIE) dataset (ECCV 2024) by building on the [R1-Onevision](https://huggingface.co/datasets/Fancy-MLLM/R1-Onevision) data foundation and substantially expanding domain diversity, task complexity, and annotation depth through a pipeline that adds hard negative answers and high-fidelity negative rationale annotations.

---

## Table of Contents

- [Overview](#overview)
- [Pipeline](#pipeline)
- [Output Format](#output-format)
- [Source Datasets](#source-datasets)
- [Requirements](#requirements)
- [Environment Variables](#environment-variables)
- [Usage](#usage)
- [Prompts](#prompts)
- [Citation](#citation)

---

## Overview

Hallucination in large vision-language models (LVLMs) partly stems from insufficient fine-grained reasoning supervision during training. REVERIE+ addresses this by augmenting existing instruction-tuning data with *reflective rationales*:

- **Positive rationale** (reused from R1-Onevision): explains *why* the correct answer is right.
- **Negative rationale** (newly generated): explains *why* a plausible but incorrect alternative is wrong, grounded in visual evidence.

Training on both rationale types discourages shortcut learning and improves visual faithfulness and hallucination robustness.

---

## Pipeline

```
R1-Onevision (source data foundation)
        │
        ▼
Step 1  step1_generate_wrong_answers.py
        │  A committee of four LVLMs — each sample assigned a model drawn
        │  uniformly at random — generates a hard negative (distractor) answer.
        │  Diverse failure modes arise from the multi-model strategy.
        │
        ▼  wrong_answers.jsonl
        │
Step 2  step2_generate_rationales.py
        │  Seed 1.6 generates a detailed negative rationale for each distractor,
        │  explicitly exposing discriminative visual evidence and reasoning errors.
        │
        ▼  with_rationales.jsonl
        │
Step 3  step3_merge_data.py
        │  Merges source conversations (positive answer + positive rationale
        │  from R1-Onevision) with the new negative answers and rationales.
        │  Malformed answer turns are repaired or dropped.
        │
        ▼  merged.jsonl
        │
Step 4  step4_reformat.py
        │  Normalises every record to canonical 4-turn or 6-turn format,
        │  cleans chain-of-thought artefacts, and applies a quality filter
        │  (drops empty / degenerate positive rationales).
        │
        ▼  reverie_plus.json   ← ready for LLaMA-Factory / training
```

---

## Output Format

Each training sample follows one of two canonical conversation schemas.

### 4-turn (positive supervision only)

| Turn | Role      | Content                              |
|:----:|-----------|--------------------------------------|
| 1    | user      | `<image>` + question                 |
| 2    | assistant | answer                               |
| 3    | user      | `Explain why`                        |
| 4    | assistant | positive rationale                   |

### 6-turn (positive + negative supervision)

| Turn | Role      | Content                                                  |
|:----:|-----------|----------------------------------------------------------|
| 1    | user      | `<image>` + question                                     |
| 2    | assistant | answer                                                   |
| 3    | user      | `Explain why`                                            |
| 4    | assistant | positive rationale                                       |
| 5    | user      | `Explain why this is incorrect: <incorrect_answer>`      |
| 6    | assistant | negative rationale                                       |

---

## Source Datasets

REVERIE+ is built on top of [R1-Onevision](https://huggingface.co/datasets/Fancy-MLLM/R1-Onevision), which aggregates 17 upstream source datasets spanning four domain categories. After consistency-based filtering, REVERIE+ retains **138,089** instances from these sources.

### Science (2 datasets)

| Dataset | Description | Paper / URL |
|---------|-------------|-------------|
| **ScienceQA** | Multi-modal science QA covering natural science, social science, and language science with detailed lecture annotations | [Lu et al., NeurIPS 2022](https://arxiv.org/abs/2209.09513) · [HF](https://huggingface.co/datasets/derek-thomas/ScienceQA) |
| **AI2D** | Diagrams from science textbooks with multiple-choice questions about diagram components and relationships | [Kembhavi et al., CVPR 2016](https://arxiv.org/abs/1603.07396) · [HF](https://huggingface.co/datasets/lmms-lab/ai2d) |

### Math (5 datasets)

| Dataset | Description | Paper / URL |
|---------|-------------|-------------|
| **Geo170K-QA** | 170K geometry QA pairs generated from GeoQA and UniGeo with detailed solution annotations | [Gao et al., 2023](https://arxiv.org/abs/2312.11370) · [HF](https://huggingface.co/datasets/Luckyjhg/Geo170K) |
| **GeoMVerse** | Synthetically generated geometry problems with controllable multi-hop difficulty levels | [Kazemi et al., 2023](https://arxiv.org/abs/2312.12241) · [HF](https://huggingface.co/datasets/HuggingFaceM4/the_cauldron) |
| **Geometry3K** | 3,002 multi-step geometry problems from middle/high school math textbooks | [Lu et al., ACL-Findings 2021](https://arxiv.org/abs/2105.04165) · [GitHub](https://github.com/lupantech/InterGPS) |
| **IconQA** | Abstract diagram understanding with diverse icon images, testing relational and spatial reasoning | [Lu et al., NeurIPS 2021](https://arxiv.org/abs/2110.13214) · [HF](https://huggingface.co/datasets/HuggingFaceM4/the_cauldron) |
| **RAVEN** | Visual IQ-test style abstract reasoning over figure matrices (Raven's Progressive Matrices) | [Zhang et al., CVPR 2019](https://arxiv.org/abs/1905.12180) · [GitHub](https://github.com/WellyZhang/RAVEN) |

### General VQA (3 datasets)

| Dataset | Description | Paper / URL |
|---------|-------------|-------------|
| **Visual7W** | 327K QA pairs grounded in MS-COCO images, covering *what*, *where*, *when*, *who*, *why*, *how*, and *which* | [Zhu et al., CVPR 2016](https://arxiv.org/abs/1511.03416) · [HF](https://huggingface.co/datasets/HuggingFaceM4/the_cauldron) |
| **VizWiz** | VQA questions from blind users based on real photos taken with mobile phones; includes unanswerable cases | [Gurari et al., CVPR 2018](https://arxiv.org/abs/1802.08218) · [HF](https://huggingface.co/datasets/lmms-lab/VizWiz-VQA) |
| **VSR** | True/False questions about spatial relationships between objects in natural images | [Liu et al., EMNLP 2023](https://arxiv.org/abs/2205.00363) · [HF](https://huggingface.co/datasets/HuggingFaceM4/the_cauldron) |

### Document / Chart / Screen (7 datasets)

| Dataset | Description | Paper / URL |
|---------|-------------|-------------|
| **DVQA** | Bar chart QA requiring structure understanding, data retrieval, and arithmetic reasoning | [Kafle et al., CVPR 2018](https://arxiv.org/abs/1801.08163) · [HF](https://huggingface.co/datasets/HuggingFaceM4/the_cauldron) |
| **RoBUT-WTQ** | Robustness-testing table QA benchmark derived from WikiTableQuestions with perturbation variants | [Zhao et al., ACL 2023](https://arxiv.org/abs/2306.14321) · [HF](https://huggingface.co/datasets/Teradata/robut-wtq) |
| **Chart2Text** | Generating natural-language summaries from bar and line charts, testing chart comprehension | [Kantharaj et al., ACL 2022](https://arxiv.org/abs/2203.06279) · [HF](https://huggingface.co/datasets/HuggingFaceM4/the_cauldron) |
| **DocVQA** | QA over document images (forms, tables, printed text) requiring document layout understanding | [Mathew et al., WACV 2021](https://arxiv.org/abs/2007.00398) · [HF](https://huggingface.co/datasets/lmms-lab/DocVQA) |
| **InfographicVQA** | QA over real-world infographics requiring joint visual and textual reasoning | [Mathew et al., WACV 2022](https://arxiv.org/abs/2104.12756) · [HF](https://huggingface.co/datasets/lmms-lab/infographicVQA) |
| **Screen2Words** | Generating concise screen captions for mobile UI screenshots | [Wang et al., UIST 2021](https://arxiv.org/abs/2108.03353) · [HF](https://huggingface.co/datasets/HuggingFaceM4/the_cauldron) |
| **VisText** | Rich captioning and QA for data visualisations (charts, plots) combining visual and semantic content | [Tang et al., ACL 2023](https://arxiv.org/abs/2307.05356) · [HF](https://huggingface.co/datasets/HuggingFaceM4/the_cauldron) |

> **Note on InterGPS / Geometry3K**: the R1-Onevision repository uses the folder name `intergps` for the Geometry3K subset, consistent with the InterGPS solver paper that introduced this split.

---

## Requirements

```bash
pip install openai tqdm
```

Python ≥ 3.10 is required (the scripts use the `X | Y` type-union syntax).

---

## Environment Variables

All API credentials are read from environment variables — never hard-coded.  
Set the following before running:

| Variable            | Used in  | Description                                            |
|---------------------|----------|--------------------------------------------------------|
| `OPENAI_API_KEY`    | Step 1   | OpenAI-compatible key (e.g. GPT-4o-mini)              |
| `ZHIPU_API_KEY`     | Step 1   | Zhipu AI key (GLM-4V-Flash)                           |
| `ARK_API_KEY`       | Step 1   | Volcengine Ark key (Doubao-Seed-1.6-Flash)            |
| `GEMINI_API_KEY`    | Step 1   | Google Gemini key (Gemini-2.0-Flash)                  |
| `RATIONALE_API_KEY` | Step 2   | Key for Seed 1.6 rationale annotation (Volcengine Ark)|

You can use any subset of the four committee models in Step 1 — simply remove entries from `MODEL_COMMITTEE` in `step1_generate_wrong_answers.py`.

---

## Usage

```bash
# Step 1 — generate hard negative answers (multi-model committee, random selection)
python step1_generate_wrong_answers.py \
    --input   /path/to/r1_onevision.jsonl \
    --output  wrong_answers.jsonl \
    --workers 8 \
    --seed    42          # optional, for reproducibility

# Step 2 — generate negative rationales with Seed 1.6
python step2_generate_rationales.py \
    --input   wrong_answers.jsonl \
    --output  with_rationales.jsonl \
    --workers 8

# Step 3 — merge into multi-turn conversation records
python step3_merge_data.py \
    --source    /path/to/r1_onevision.jsonl \
    --rationale with_rationales.jsonl \
    --output    merged.jsonl

# Step 4 — reformat to canonical format and apply quality filter
python step4_reformat.py \
    --input           merged.jsonl \
    --output          reverie_plus.json \
    --output-complete reverie_plus_6turn.json   # 6-turn samples only
```

The final `reverie_plus.json` can be registered directly in LLaMA-Factory's `dataset_info.json` for training.

---

## Prompts

Full prompt templates for both annotation stages are documented in [`prompts.md`](prompts.md).

| Stage  | Model(s)                                              | Purpose                            |
|--------|-------------------------------------------------------|------------------------------------|
| Step 1 | GPT-4o-mini / GLM-4V-Flash / Doubao-Seed / Gemini-Flash | Hard negative answer generation   |
| Step 2 | Seed 1.6                                              | Negative rationale distillation    |

---

## Notes

- **Random model selection (Step 1)**: each sample is assigned one model drawn uniformly at random from the four-model committee via `random.choice()`. This ensures no single model dominates the negative answers and maximises failure-mode diversity. Pass `--seed <int>` for a reproducible run.
- **Resume support**: Steps 1 and 2 check the output file for already-processed IDs and skip them on restart, so interrupted runs are safe to continue.
- **Consistency-based filtering**: Step 4 drops samples whose positive rationale is empty or identical to the answer (string-normalised), as these provide no additional supervision signal. The Seed 1.6 annotator also performs intra-consistency checks between positive and negative rationales during generation (Step 2).
- **Pre-generated dataset**: the full REVERIE+ dataset (pre-generated, ready to use) is released separately on Hugging Face at [YOUR_HF_LINK].

---

