# REVERIE+ Data Generation Prompts

This file documents the two prompts used in the REVERIE+ data construction
pipeline, corresponding to the two annotation stages described in the paper
(Sec. 3.2).

---

## Prompt 1 — Hard Negative Answer Mining (Step 1)

Used by the **committee of LVLMs** (GPT-4o-mini / GLM-4V-Flash /
Doubao-Seed-1.6-Flash / Gemini-2.0-Flash) to generate a plausible but
factually incorrect answer for each sample.  For each sample a model is
drawn uniformly at random from the committee.

```
You are provided with an image, a question related to that image, and the
correct answer. Your task is to generate a factually incorrect answer to the
question.

Instructions:
- Analyze the context: carefully examine the image and the question. The
  incorrect answer must be a plausible "distractor" that is difficult to
  distinguish from the truth.
- Multiple Choice: if the question is a multiple-choice problem, select the
  option most likely to trap a user, other than the correct one.
- Alignment: your generated answer must match the format of the correct
  answer (e.g., if the correct answer is a single word, yours must be too)
  and be semantically relevant to the question.
- Output: output only the generated incorrect answer. Do not include
  explanations or any other text.

QUESTION AND ANSWER:
{QUESTION}
{ANSWER}

OUTPUT:
```

---

## Prompt 2 — Negative Rationale Distillation (Step 2)

Used exclusively by **Seed 1.6** to generate a detailed negative rationale
that explicitly explains why the incorrect answer is wrong based on evidence
in the image.

```
Task: You are provided with an image, a question about that image, the
correct answer, and an incorrect answer. Your objective is to generate a
comprehensive rationale explaining exactly why the incorrect answer is wrong.

Instructions:
1. Visual Analysis: examine the image meticulously, identifying specific
   visual elements that directly relate to the question.
2. Evidence-Based Rebuttal: construct a logical argument that disproves the
   incorrect answer. Cite specific details from the image as evidence.
3. Contextual Clarity: ensure your explanation is clear and concise. If
   necessary, apply general knowledge to contextualise why the visual
   evidence contradicts the incorrect answer.
4. Base the analysis on the image itself — do not merely compare the
   incorrect answer with the correct one.
5. Output Format: provide the rationale only. Do not include introductory
   text, greetings, or the correct answer in your output.

Input Data:
Question: {QUESTION}
Correct Answer: {CORRECT_ANSWER}
Incorrect Answer: {INCORRECT_ANSWER}

Rationale:
```
