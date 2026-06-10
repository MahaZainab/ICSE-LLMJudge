#!/usr/bin/env python3

import json
import os
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

JUDGE_MODEL_ID = "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct"
INPUT_FILE     = "RQ1_generate_cs1qa/llama3.2_1b_instruct_predictions.json"
OUTPUT_JSON    = "RQ1_judge_cs1qa/llama3.2_1b_instruct_judged.json"
OUTPUT_CSV     = "RQ1_judge_cs1qa/llama3.2_1b_instruct_judged.csv"
OUTPUT_PLOT    = "RQ1_judge_cs1qa/llama3.2_1b_instruct_judged.png"
SAVE_EVERY     = 25
MAX_NEW_TOKENS = 256
HF_CACHE       = os.getenv("HF_HOME", "")

SYSTEM_PROMPT = """You are an expert evaluator assessing the quality of teaching assistant (TA) responses to student programming questions in an introductory Python course.
You will receive:
- A student's code
- The student's question about their code
- A reference answer (a correct TA response)
- A predicted answer (a model-generated TA response)

Your task is to evaluate the predicted answer against the reference answer using four dimensions.
For each dimension provide an integer score from 1 to 5.

### Accuracy
Assess whether the predicted answer is factually correct about the student's code and consistent with the reference answer.
Consider semantic equivalence: a predicted answer that conveys the same correct information as the reference but in different words must be treated as correct.
Do NOT penalize correct answers merely because they are phrased differently from the reference.
  5: Fully correct — factually accurate and consistent with the reference answer
  4: Mostly correct — minor inaccuracies that do not affect the core meaning
  3: Partially correct — some key facts are accurate but important details are wrong or missing
  2: Mostly incorrect — a few relevant facts but major errors dominate
  1: Completely incorrect — does not address the reference answer at all

### Completeness
Assess whether the predicted answer fully addresses what the student is asking.
Evaluate whether the answer would resolve the student's specific need, not just whether it matches the reference length.
  5: Fully addresses the student's need
  4: Mostly addresses it with minor omissions
  3: Partially addresses it — some aspects covered but important parts missing
  2: Mostly misses what the student is asking
  1: Entirely fails to address the student's question

### Clarity
Assess how clearly the predicted answer is expressed for a beginner programmer.
Score this dimension INDEPENDENTLY of factual correctness.
A clear but factually wrong answer can still score 5 on clarity.
  5: Perfectly clear and appropriate for a beginner
  4: Mostly clear with minor phrasing issues
  3: Somewhat clear but could confuse a beginner
  2: Difficult for a beginner to follow
  1: Incomprehensible

### Relevance
Assess whether the predicted answer directly addresses the student's specific question without going off-topic.
  5: Fully on-topic — directly addresses the student's question
  4: Mostly on-topic — minor tangents that do not distract from the answer
  3: Partially relevant — addresses some aspects but drifts from the question
  2: Mostly off-topic — misses what the student was asking
  1: Completely irrelevant — does not address the student's question at all

Example:
{
  "code": "for i in range(10):\\n    print(i)\\n    i = i + 2",
  "question": "Why is my loop not skipping by 2?",
  "reference_answer": "In a for loop Python reassigns i on each iteration so i = i + 2 has no effect. Use range(0, 10, 2) instead.",
  "predicted_answer": "The loop variable is controlled by Python so you cannot change it manually inside the loop.",
  "evaluation": {
    "accuracy":     {"score": 3},
    "completeness": {"score": 3},
    "clarity":      {"score": 4},
    "relevance":    {"score": 5}
  }
}

Final Instructions:
Base your evaluation strictly on the content provided. Do not hallucinate.
Be consistent and objective. Score each dimension independently.
For open-ended explanation questions, evaluate whether the explanation is logically sound and helps the student, not whether it matches the reference word-for-word.
Respond ONLY with a JSON object in this exact format with no additional text:
{
  "accuracy":     {"score": <1-5>},
  "completeness": {"score": <1-5>},
  "clarity":      {"score": <1-5>},
  "relevance":    {"score": <1-5>}
}"""

def build_user_prompt(code, question, reference, prediction):
    return (
        f"Student's code:\n```python\n{code}\n```\n\n"
        f"Student's question:\n{question}\n\n"
        f"Reference Answer:\n{reference}\n\n"
        f"Predicted Answer:\n{prediction}"
    )

def save_append(path, new_data):
    try:
        with open(path, "r", encoding="utf-8") as f:
            existing = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        existing = []
    existing_map = {item.get("id"): item for item in existing}
    for item in new_data:
        existing_map[item.get("id")] = item   # overwrite, not skip
    combined = list(existing_map.values())
    with open(path, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)

def extract_scores(response_text):
    try:
        start = response_text.find("{")
        end   = response_text.rfind("}") + 1
        if start != -1 and end > start:
            parsed = json.loads(response_text[start:end])
        else:
            parsed = json.loads(response_text)
    except json.JSONDecodeError:
        print(f"  Parse error: {response_text[:200]}")
        return {}
    results = {}
    for metric, details in parsed.items():
        if isinstance(details, dict):
            score = details.get("score")
            if isinstance(score, int) and 1 <= score <= 5:
                results[metric] = {"score": score}
    return results

def export_csv(records, path):
    df = pd.DataFrame(records)
    cols = ["id", "dataset", "category", "question_type", "code",
            "question", "answer", "prediction", "accuracy",
            "completeness", "clarity", "relevance"]
    df = df[[c for c in cols if c in df.columns]]
    df.to_csv(path, index=False)
    print(f"CSV saved to {path}")

def visualize(records, path):
    df = pd.DataFrame(records)
    metrics = ["accuracy", "completeness", "clarity", "relevance"]
    avgs, stds = [], []
    for m in metrics:
        valid = pd.to_numeric(df[m], errors="coerce").dropna()
        avgs.append(valid.mean() if len(valid) > 0 else 0)
        stds.append(valid.std()  if len(valid) > 1 else 0)
    plt.figure(figsize=(8, 5))
    plt.bar(metrics, avgs, yerr=stds, capsize=5,
            color="skyblue", edgecolor="black")
    plt.title("Average Judge Scores per Dimension — CS1QA")
    plt.ylabel("Score (1–5)")
    plt.ylim(1, 5)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to {path}")

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device count:   {torch.cuda.device_count()}")
print(f"Loading judge:  {JUDGE_MODEL_ID}")

tokenizer = AutoTokenizer.from_pretrained(
    JUDGE_MODEL_ID,
    cache_dir=HF_CACHE,
    trust_remote_code=True,
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    JUDGE_MODEL_ID,
    cache_dir=HF_CACHE,
    dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
model.eval()
print(f"Model loaded on: {model.device}\n")

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    dataset = json.load(f)
print(f"Loaded {len(dataset)} records from {INPUT_FILE}\n")

results     = []
csv_records = []

for i, item in enumerate(tqdm(dataset, desc="Judging")):
    code       = item.get("code", "")
    question   = item.get("question", "")
    reference  = item.get("answer", "")
    prediction = item.get("prediction", "")
    category   = item.get("category", "unknown")
    q_type     = item.get("question_type", category)
    q_id       = item.get("id", f"q{i+1}")

    user_prompt = build_user_prompt(code, question, reference, prediction)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_prompt},
    ]

    try:
        formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        formatted = (
            f"### System:\n{SYSTEM_PROMPT}\n\n"
            f"### User:\n{user_prompt}\n\n"
            f"### Assistant:\n"
        )

    inputs = tokenizer(
        formatted,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=8192,
    ).to(model.device)

    try:
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                use_cache=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        response   = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        scores     = extract_scores(response)
        if not scores:
            print(f"  Warning: no scores parsed at record {i+1}. Raw response: {response[:200]}")
    except Exception as e:
        print(f"  Error at record {i+1}: {e}")
        scores = {}

    acc  = scores.get("accuracy",     {}).get("score", None)
    comp = scores.get("completeness", {}).get("score", None)
    clar = scores.get("clarity",      {}).get("score", None)
    rel  = scores.get("relevance",    {}).get("score", None)

    print(f"[{i+1}/{len(dataset)}] category={category} | type={q_type}")
    print(f"  acc={acc} comp={comp} clar={clar} rel={rel}\n")

    result = {
        "id":           q_id,
        "dataset":      "cs1qa",
        "category":     category,
        "question_type": q_type,
        "code":         code,
        "question":     question,
        "answer":       reference,
        "prediction":   prediction,
        "accuracy":     acc,
        "completeness": comp,
        "clarity":      clar,
        "relevance":    rel,
    }
    results.append(result)

    csv_records.append({
        "id":            q_id,
        "dataset":       "cs1qa",
        "category":      category,
        "question_type": q_type,
        "code":          code,
        "question":      question,
        "answer":        reference,
        "prediction":    prediction,
        "accuracy":      acc,
        "completeness":  comp,
        "clarity":       clar,
        "relevance":     rel,
    })

    if (i + 1) % SAVE_EVERY == 0 or (i + 1) == len(dataset):
        save_append(OUTPUT_JSON, results)
        print(f"  Checkpoint saved at record {i+1}")
        results = []

export_csv(csv_records, OUTPUT_CSV)
visualize(csv_records, OUTPUT_PLOT)
print(f"\nDone. Results saved to {OUTPUT_JSON}")
