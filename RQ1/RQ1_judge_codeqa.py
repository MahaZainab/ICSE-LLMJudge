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

JUDGE_MODEL_ID = "Qwen/Qwen2.5-Coder-7B-Instruct"
INPUT_FILE     = "CodeQA_predictions.json"
OUTPUT_JSON    = "CodeQA_judged.json"
OUTPUT_CSV     = "CodeQA_judged.csv"
OUTPUT_PLOT    = "CodeQA_judged.png"
SAVE_EVERY     = 25
MAX_NEW_TOKENS = 256
HF_CACHE       = os.getenv("HF_HOME", "")

SYSTEM_PROMPT = """You are an expert evaluator assessing the quality of answers to source code comprehension questions.
You will receive:
- A code snippet
- A question about that code
- A reference answer (correct answer)
- A predicted answer (model-generated answer)

Your task is to evaluate the predicted answer against the reference answer using four dimensions.
For each dimension provide an integer score from 1 to 5.

### Accuracy
Assess whether the predicted answer is factually correct based on the code and consistent with the reference answer.
Consider semantic equivalence: a predicted answer that conveys the same meaning as the reference but in different words must be treated as correct.
Do NOT penalize correct answers merely because they are phrased differently from the reference.
  5: Fully correct — matches the reference answer in meaning and factual content
  4: Mostly correct — minor inaccuracies that do not affect the core meaning
  3: Partially correct — some key facts are accurate but important details are wrong or missing
  2: Mostly incorrect — a few relevant facts but major errors dominate
  1: Completely incorrect — does not address the reference answer at all

### Completeness
Assess whether the predicted answer covers all the important information required by the question.
A short answer that fully addresses the question scores 5. Do NOT penalize brevity if the question requires a brief answer.
  5: Fully complete — covers all essential content the question asks for
  4: Mostly complete — minor omissions that do not affect overall understanding
  3: Partially complete — covers some key points but misses important content
  2: Mostly incomplete — only a small fragment of the required content is present
  1: Entirely incomplete — omits almost all key information

### Clarity
Assess how clearly and understandably the predicted answer is expressed.
Score this dimension INDEPENDENTLY of factual correctness.
A grammatically correct but factually wrong answer can still score 5 on clarity.
  5: Perfectly clear and easy to understand
  4: Mostly clear with minor phrasing issues
  3: Somewhat clear but awkwardly expressed
  2: Difficult to understand
  1: Incomprehensible or incoherent

### Relevance
Assess whether the predicted answer directly addresses the question without going off-topic.
  5: Fully on-topic — directly answers what was asked
  4: Mostly on-topic — minor tangents that do not distract from the answer
  3: Partially relevant — addresses some aspects but drifts from the main point
  2: Mostly off-topic — misses the main point of the question
  1: Completely irrelevant — does not address the question at all

Example:
{
  "code": "def aggregate_metadata_get_by_host context host key None return IMPL aggregate_metadata_get_by_host context host key",
  "question": "What does the code get?",
  "reference_answer": "metadata for all aggregates that host belongs to",
  "predicted_answer": "host metadata",
  "evaluation": {
    "accuracy":     {"score": 2},
    "completeness": {"score": 2},
    "clarity":      {"score": 4},
    "relevance":    {"score": 3}
  }
}

Final Instructions:
Base your evaluation strictly on the content provided. Do not hallucinate.
Be consistent and objective. Score each dimension independently.
Respond ONLY with a JSON object in this exact format with no additional text:
{
  "accuracy":     {"score": <1-5>},
  "completeness": {"score": <1-5>},
  "clarity":      {"score": <1-5>},
  "relevance":    {"score": <1-5>}
}"""

def build_user_prompt(code, question, reference, prediction):
    return (
        f"Code:\n{code}\n\n"
        f"Question:\n{question}\n\n"
        f"Reference Answer:\n{reference}\n\n"
        f"Predicted Answer:\n{prediction}"
    )

def save_append(path, new_data):
    try:
        with open(path, "r", encoding="utf-8") as f:
            existing = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        existing = []
    existing_ids = {item.get("id") for item in existing if "id" in item}
    filtered = [item for item in new_data if item.get("id") not in existing_ids]
    combined = existing + filtered
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
    cols = ["id", "dataset", "category", "code", "question",
            "answer", "prediction", "accuracy", "completeness",
            "clarity", "relevance"]
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
    plt.title("Average Judge Scores per Dimension — CodeQA")
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
    torch_dtype=torch.float16,
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
        max_length=4096,
    ).to(model.device)

    try:
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        response   = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        scores     = extract_scores(response)
    except Exception as e:
        print(f"  Error at record {i+1}: {e}")
        scores = {}

    acc  = scores.get("accuracy",     {}).get("score", None)
    comp = scores.get("completeness", {}).get("score", None)
    clar = scores.get("clarity",      {}).get("score", None)
    rel  = scores.get("relevance",    {}).get("score", None)

    print(f"[{i+1}/{len(dataset)}] category={category}")
    print(f"  acc={acc} comp={comp} clar={clar} rel={rel}\n")

    result = {
        "id":           q_id,
        "dataset":      "codeqa",
        "category":     category,
        "code":         code,
        "question":     question,
        "answer":       reference,
        "prediction":   prediction,
        "accuracy":     {"score": acc},
        "completeness": {"score": comp},
        "clarity":      {"score": clar},
        "relevance":    {"score": rel},
    }
    results.append(result)

    csv_records.append({
        "id":           q_id,
        "dataset":      "codeqa",
        "category":     category,
        "code":         code,
        "question":     question,
        "answer":       reference,
        "prediction":   prediction,
        "accuracy":     acc,
        "completeness": comp,
        "clarity":      clar,
        "relevance":    rel,
    })

    if (i + 1) % SAVE_EVERY == 0 or (i + 1) == len(dataset):
        save_append(OUTPUT_JSON, results)
        print(f"  Checkpoint saved at record {i+1}")
        results = []

export_csv(csv_records, OUTPUT_CSV)
visualize(csv_records, OUTPUT_PLOT)
print(f"\nDone. Results saved to {OUTPUT_JSON}")
