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
INPUT_FILE     = "RQ1_generate_codeqa/llama3.2_3b_instruct_predictions.json"
OUTPUT_JSON    = "RQ1_judge_codeqa/llama3.2_3b_instruct_judged1.json"
OUTPUT_CSV     = "RQ1_judge_codeqa/llama3.2_3b_instruct_judged1.csv"
OUTPUT_PLOT    = "RQ1_judge_codeqa/llama3.2_3b_instruct_judged1.png"
SAVE_EVERY     = 25
MAX_NEW_TOKENS = 128
HF_CACHE       = os.getenv("HF_HOME", "")

SYSTEM_PROMPT = """You are an expert software engineer and code evaluator with deep experience assessing the quality of answers to source code comprehension questions.

You will be given a code snippet, a question about that code, and a predicted answer.
Your task is to evaluate the predicted answer across four dimensions: Accuracy, Completeness, Clarity, and Relevance.

Dataset context:
The questions come from CodeQA, a free-form question-answering benchmark built from real Python and Java code on GitHub.
Questions are derived from code comments (docstrings, Javadocs) and cover four types of code understanding:
- Functionality: what the code does, returns, creates, or produces
- Purpose: why the code exists or what problem it solves
- Property: attributes, parameters, types, conditions, or constraints in the code
- Workflow: how the code operates step-by-step or how data flows through it
Correct answers in CodeQA are typically concise — often a short phrase or a single sentence, not a paragraph.

Scoring dimensions (score each independently on a 1 to 5 integer scale):

ACCURACY — Does the predicted answer correctly reflect what the code actually does?
  5: Completely correct — fully consistent with the code's actual behavior
  4: Mostly correct — minor factual slip that does not change the core meaning
  3: Partially correct — captures something true but misses or misstates a key detail
  2: Mostly incorrect — contains a relevant element but dominated by factual errors
  1: Completely wrong — contradicts the code or addresses something entirely different

COMPLETENESS — Does the predicted answer cover everything the question asks for?
  5: Fully complete — addresses everything the question asks at the right level of detail
  4: Mostly complete — minor omission that does not significantly affect the answer
  3: Partially complete — addresses part of the question but misses an important aspect
  2: Mostly incomplete — only a surface fragment of what is required is present
  1: Entirely incomplete — fails to address the question in any meaningful way

CLARITY — How clearly does the predicted answer communicate its point?
  5: Perfectly clear — unambiguous and easy to understand
  4: Mostly clear — minor phrasing awkwardness that does not impede understanding
  3: Somewhat clear — understandable with effort but awkwardly expressed
  2: Unclear — confusing or ambiguous to the point of impeding understanding
  1: Incomprehensible — incoherent, self-contradictory, or unreadable

RELEVANCE — Does the predicted answer directly target what the question is asking?
  5: Fully relevant — directly and precisely answers the question asked
  4: Mostly relevant — minor tangent that does not distract from the answer
  3: Partially relevant — addresses a related but different aspect of the code
  2: Mostly irrelevant — misses the main point of the question
  1: Completely irrelevant — does not address the question at all

IMPORTANT: Read the code carefully before scoring — your accuracy score must be grounded in the code, not in assumptions.
IMPORTANT: Semantic equivalence counts as correct. "the name", "Name of the user", and "it returns the user's name" can all be correct for the same question if the code supports them.
IMPORTANT: Score each dimension INDEPENDENTLY. A clear answer can still be inaccurate. An accurate answer can still be irrelevant.
IMPORTANT: A short answer is NOT incomplete if it fully addresses the question — CodeQA answers are intentionally concise.

CRITICAL: DO NOT USE ANY REFERENCE ANSWER — none is provided. Evaluate based on the code alone.
CRITICAL: DO NOT PENALIZE AN ANSWER FOR PHRASING OR VOCABULARY DIFFERENCES IF THE MEANING IS CORRECT.
CRITICAL: DO NOT CONFLATE DIMENSIONS — score CLARITY independently of ACCURACY.
CRITICAL: DO NOT HALLUCINATE FACTS ABOUT THE CODE. If you are uncertain, score conservatively.
CRITICAL: DO NOT PRODUCE ANY TEXT OUTSIDE THE JSON OBJECT — no explanation, no preamble, no reasoning, no markdown.

Calibration examples:

Example 1 — fully correct, concise answer:
Code: def get_suite ( self , suite_dict , label = None ) : suite = unittest.TestSuite ( ) for test_name in suite_dict : suite.addTest ( self.get_test ( test_name ) ) return suite
Question: What does the code return?
Predicted Answer: a test suite
Output: {"accuracy": {"score": 5}, "completeness": {"score": 5}, "clarity": {"score": 5}, "relevance": {"score": 5}}

Example 2 — accurate but answers HOW instead of WHAT:
Code: def get_suite ( self , suite_dict , label = None ) : suite = unittest.TestSuite ( ) for test_name in suite_dict : suite.addTest ( self.get_test ( test_name ) ) return suite
Question: What does the code return?
Predicted Answer: It creates a new TestSuite object and iterates through the suite_dict to add each test by name before returning the populated suite object.
Output: {"accuracy": {"score": 5}, "completeness": {"score": 4}, "clarity": {"score": 5}, "relevance": {"score": 3}}

Example 3 — factually wrong but clearly written and partially on-topic:
Code: def is_valid_age ( age ) : return isinstance ( age , int ) and age >= 0 and age <= 120
Question: What does the code check?
Predicted Answer: whether the input is a string
Output: {"accuracy": {"score": 1}, "completeness": {"score": 2}, "clarity": {"score": 5}, "relevance": {"score": 3}}

Respond ONLY with a valid JSON object in exactly this format:
{
  "accuracy":     {"score": <1-5>},
  "completeness": {"score": <1-5>},
  "clarity":      {"score": <1-5>},
  "relevance":    {"score": <1-5>}
}"""

def build_user_prompt(code, question, prediction):
    return (
        f"Code:\n{code}\n\n"
        f"Question:\n{question}\n\n"
        f"Predicted Answer:\n{prediction}"
    )

# FIX 1: overwrite existing records by ID instead of skipping them
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
        max_length=8192,        # Qwen2.5-Coder-7B supports up to 128k; 8192 is safe for typical inputs
    ).to(model.device)

    try:
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                use_cache=False,                    # FIX 4: avoids DynamicCache seen_tokens error
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

    print(f"[{i+1}/{len(dataset)}] category={category}")
    print(f"  acc={acc} comp={comp} clar={clar} rel={rel}\n")

    # FIX 5: store raw scores in results (JSON) to match csv_records — no more nested {"score": null}
    result = {
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
