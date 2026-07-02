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
INPUT_FILE     = "RQ1_generate_codeqa/filtered/llama3.2_3b_instruct_predictions.json"
OUTPUT_JSON    = "RQ1_judge_codeqa/filtered/llama3.2_3b_instruct_judged1.json"
OUTPUT_CSV     = "RQ1_judge_codeqa/filtered/llama3.2_3b_instruct_judged1.csv"
OUTPUT_PLOT    = "RQ1_judge_codeqa/filtered/llama3.2_3b_instruct_judged1.png"
SAVE_EVERY     = 25
MAX_NEW_TOKENS = 200
HF_CACHE       = os.getenv("HF_HOME", "")

SYSTEM_PROMPT = """You are an expert software engineer and code evaluator with deep experience assessing the quality of answers to source code comprehension questions.

You will be given a code snippet, a question about that code, a reference answer, and a predicted answer.
Use the reference answer as a gold-standard anchor — semantic equivalence to the reference counts as correct.
Your task is to evaluate the predicted answer across four dimensions: Accuracy, Completeness, Clarity, and Relevance.

Dataset context:
The questions come from CodeQA, a free-form question-answering benchmark built from real Python and Java code on GitHub.
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

IMPORTANT: Semantic equivalence to the reference answer counts as correct. A predicted answer that paraphrases the reference using different vocabulary but the same meaning must be scored as accurate — do NOT penalise for word choice or abstraction level.
IMPORTANT: For "Why" questions, an answer that correctly identifies the cause using plain language is as valid as one that names the exact code symbol, provided the meaning is equivalent.
IMPORTANT: Clarity must be scored on how well the answer communicates its point — not on whether it is factually correct. A factually wrong answer can still score 5 on Clarity; a factually correct answer can still score low on Clarity.
IMPORTANT: Score each dimension INDEPENDENTLY.
IMPORTANT: A short answer is NOT incomplete if it fully addresses the question.

CRITICAL: DO NOT PENALISE FOR PHRASING OR VOCABULARY DIFFERENCES IF THE MEANING IS CORRECT.
CRITICAL: DO NOT PRODUCE ANY TEXT OUTSIDE THE JSON OBJECT — no explanation, no preamble, no markdown.

Calibration examples:

Example 1 — semantically correct paraphrase of a code condition:
Code: def connect(self): ... if not can_reconnect(e): raise ...
Question: Why does the function disconnect the client?
Reference Answer: Due to a timeout / etc.
Predicted Answer: The client is disconnected because it encounters an unrecoverable error.
Output: {"accuracy": {"score": 4}, "completeness": {"score": 4}, "clarity": {"score": 5}, "relevance": {"score": 5}}

Example 2 — factually inverted answer, clearly written:
Code: def connect(self): ... if not can_reconnect(e): raise ...
Question: Why does the function disconnect the client?
Reference Answer: Due to a timeout / etc.
Predicted Answer: No, the function reconnects the client.
Output: {"accuracy": {"score": 1}, "completeness": {"score": 1}, "clarity": {"score": 3}, "relevance": {"score": 1}}


Respond ONLY with a valid JSON object in exactly this format:
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
    existing_map = {item.get("id"): item for item in existing}
    for item in new_data:
        existing_map[item.get("id")] = item
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
