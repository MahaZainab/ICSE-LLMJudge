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
INPUT_FILE     = "RQ1_generate_cs1qa/llama3.2_3b_instruct_predictions.json"
OUTPUT_JSON    = "RQ1_judge_cs1qa/llama3.2_3b_instruct_judged.json"
OUTPUT_CSV     = "RQ1_judge_cs1qa/llama3.2_3b_instruct_judged.csv"
OUTPUT_PLOT    = "RQ1_judge_cs1qa/llama3.2_3b_instruct_judged.png"
SAVE_EVERY     = 25
MAX_NEW_TOKENS = 128
HF_CACHE       = os.getenv("HF_HOME", "")

SYSTEM_PROMPT = """You are an expert evaluator and educator with deep experience assessing the quality of teaching assistant (TA) responses to student programming questions in an introductory Python course.

You will be given a student's code, the student's question about that code, and a predicted TA response.
Your task is to evaluate the predicted response across four dimensions: Accuracy, Completeness, Clarity, and Relevance.
You must read the student's code and question yourself to determine what a correct and helpful TA response looks like.
There is no reference answer provided.

Dataset context:
The questions come from CS1QA, a dataset of real student-TA interactions from an introductory Python programming course.
Students ask questions about bugs, errors, logic issues, and concepts in their own code.
A good TA response in this context:
- Identifies the issue correctly without giving the answer away directly
- Guides the student toward the fix rather than handing it to them
- Uses beginner-friendly language appropriate for an introductory course
- Is concise — 1 to 4 sentences is the expected length

Scoring dimensions (score each independently on a 1 to 5 integer scale):

ACCURACY — Is the predicted response factually correct about the student's code?
  5: Completely correct — accurately identifies the issue or concept in the student's code
  4: Mostly correct — minor inaccuracy that does not affect the core guidance
  3: Partially correct — identifies something relevant but misses or misstates a key detail
  2: Mostly incorrect — contains a relevant element but dominated by factual errors about the code
  1: Completely wrong — contradicts what the code actually does or is entirely off-base

COMPLETENESS — Does the predicted response fully address what the student is asking?
  5: Fully addresses the student's need — would resolve their confusion
  4: Mostly addresses it — minor omission that does not significantly affect usefulness
  3: Partially addresses it — covers some aspects but leaves an important part unresolved
  2: Mostly incomplete — only touches the surface of what the student needs
  1: Entirely fails to address the student's question

CLARITY — How clearly is the predicted response expressed for a beginner programmer?
  5: Perfectly clear — unambiguous and easy for a beginner to understand
  4: Mostly clear — minor phrasing awkwardness that does not impede understanding
  3: Somewhat clear — understandable with effort but could confuse a beginner
  2: Unclear — confusing or uses terminology a beginner would not follow
  1: Incomprehensible — incoherent or completely inaccessible to a beginner

RELEVANCE — Does the predicted response directly address the student's specific question?
  5: Fully relevant — directly and precisely addresses what the student asked
  4: Mostly relevant — minor tangent that does not distract from the answer
  3: Partially relevant — addresses a related but different aspect of the problem
  2: Mostly irrelevant — misses the main point of the student's question
  1: Completely irrelevant — does not address the student's question at all

IMPORTANT: Read the student's code carefully before scoring — your accuracy score must be grounded in the code, not in assumptions.
IMPORTANT: Score each dimension INDEPENDENTLY. A clearly written response can still be inaccurate. An accurate response can still be irrelevant.
IMPORTANT: A short response is NOT incomplete if it fully addresses the student's need — good TA responses are concise by design.
IMPORTANT: For open-ended questions, evaluate whether the explanation is logically sound and helpful to the student.

CRITICAL: DO NOT USE ANY REFERENCE ANSWER — none is provided. Evaluate based on the code and question alone.
CRITICAL: DO NOT PENALIZE A RESPONSE FOR PHRASING DIFFERENCES IF THE MEANING AND GUIDANCE ARE CORRECT.
CRITICAL: DO NOT CONFLATE DIMENSIONS — score CLARITY independently of ACCURACY.
CRITICAL: DO NOT PENALIZE A RESPONSE FOR NOT GIVING THE COMPLETE SOLUTION — good TA responses guide, not solve.
CRITICAL: DO NOT HALLUCINATE FACTS ABOUT THE CODE. If you are uncertain, score conservatively.
CRITICAL: DO NOT PRODUCE ANY TEXT OUTSIDE THE JSON OBJECT — no explanation, no preamble, no reasoning, no markdown.

Calibration example:

Student's code:
for i in range(10):
    print(i)
    i = i + 2
Student's question: Why is my loop not skipping by 2?
Predicted TA response: The loop variable is controlled by Python so you cannot change it manually inside the loop.
Output: {"accuracy": {"score": 3}, "completeness": {"score": 3}, "clarity": {"score": 4}, "relevance": {"score": 5}}

Respond ONLY with a valid JSON object in exactly this format:
{
  "accuracy":     {"score": <1-5>},
  "completeness": {"score": <1-5>},
  "clarity":      {"score": <1-5>},
  "relevance":    {"score": <1-5>}
}"""

def build_user_prompt(code, question, prediction):
    return (
        f"Student's code:\n```python\n{code}\n```\n\n"
        f"Student's question:\n{question}\n\n"
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
    torch_dtype="auto",
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

    user_prompt = build_user_prompt(code, question, prediction)

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
                temperature=None,
                top_p=None,
                top_k=None,
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
