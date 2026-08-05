#!/usr/bin/env python3
"""
Big code-focused LLM baseline for CS1QA -- DeepSeek-Coder-33B-Instruct.

Cloned from RQ1_generate_cs1qa.py with two changes only:
  1. GENERATOR_MODEL_ID swapped to the big model.
  2. Memory/latency instrumentation added (load time, peak GPU memory,
     per-example generation latency).
SYSTEM_PROMPT, FEW_SHOT_EXAMPLES, build_user_prompt, generation settings
(greedy, MAX_NEW_TOKENS=256, max_length=4096) and record schema are all
copied verbatim from RQ1_generate_cs1qa.py -- nothing in the prompt or
decoding logic was touched.
"""

import json
import os
import time
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

GENERATOR_MODEL_ID = "deepseek-ai/deepseek-coder-33b-instruct"
INPUT_FILE         = "CS1QA_dataset.json"
OUTPUT_FILE        = "deepseek_coder_33b_instruct_cs1qa_predictions.json"
STATS_FILE         = "deepseek_coder_33b_instruct_cs1qa_stats.json"
SAVE_EVERY         = 25
MAX_NEW_TOKENS     = 256
HF_CACHE           = os.getenv("HF_HOME", "")

FEW_SHOT_EXAMPLES = [
    {
        "code": (
            "def is_triangle(a, b, c):\n"
            "    if float(max(a,b,c)) < float(a+b+c) - float(max(a,b,c)):\n"
            "        print('YES')\n"
            "    else:\n"
            "        print('NO')\n"
            "a = input('Side a: ')\n"
            "b = input('Side b: ')\n"
            "c = input('Side c: ')\n"
            "is_triangle(a, b, c)"
        ),
        "question": "Why do I get an error?",
        "answer": (
            "When a value is received as an input, a, b, c are always variables of type String.\n"
            "Think about what type max() and float() expect — "
            "you may need to convert the inputs before passing them in."
        ),
    },
    {
        "code": (
            "from cs1robots import *\n"
            "create_world()\n"
            "hubo = Robot(beepers=10)\n"
            "def hubo.nine():\n"
            "    for i in range(9):\n"
            "        hubo.move()"
        ),
        "question": "I want to use the for statement, but I keep getting an error. I don't know what's wrong with def.",
        "answer": (
            "Function names cannot contain \".\". "
            "Try defining it as def nine() and calling it as nine() instead of hubo.nine(). "
            "Functions like hubo.move() are special — they are already defined inside the Robot class."
        ),
    },
    {
        "code": (
            "s = []\n"
            "f = open('countries.csv', 'r')\n"
            "line = f.readline()\n"
            "for line in f:\n"
            "    s.append(line.strip())\n"
            "f.close()\n"
            "for i in range(len(s)):\n"
            "    cc = s[i][1:3]"
        ),
        "question": "When moving elements such as country name from a file to a list, the length is different, so it is a little difficult. Can you help me?",
        "answer": (
            "If you use a function called split(), you can cut a string at any separator you want. "
            "CSV files are separated by commas, so try splitting on \",\" — "
            "that way each field becomes its own element and you can access the country name directly by index!"
        ),
    },
]

SYSTEM_PROMPT = (
    "You are a teaching assistant (TA) for an introductory Python programming course.\n"
    "A student has asked you a question about their code.\n"
    "\n"
    "Your role:\n"
    "- Respond in the style of a human TA during a real office hours session.\n"
    "- Keep your answer short and conversational — 1 to 4 sentences is sufficient.\n"
    "- Explain the concept or point out the issue clearly.\n"
    "- Nudge the student toward the correct fix without giving it away.\n"
    "- You may confirm when a student's understanding is correct.\n"
    "- You may clarify a concept with a brief beginner-friendly explanation.\n"
    "- You may hint at what to look for or what function/concept to explore.\n"
    "\n"
    "IMPORTANT: Base your answer ONLY on the provided code and question.\n"
    "IMPORTANT: Match the tone and length of the examples — conversational, concise, and pedagogical.\n"
    "IMPORTANT: If the student's code has multiple issues, focus on the most critical one first.\n"
    "\n"
    "CRITICAL: DO NOT REWRITE THE STUDENT'S CODE FOR THEM.\n"
    "CRITICAL: DO NOT PROVIDE COMPLETE CORRECTED CODE UNDER ANY CIRCUMSTANCES.\n"
    "CRITICAL: DO NOT HAND THE ANSWER TO THEM DIRECTLY — guide, do not solve.\n"
    "CRITICAL: DO NOT REPEAT OR PARAPHRASE THE QUESTION IN YOUR ANSWER.\n"
    "CRITICAL: DO NOT ADD INFORMATION THAT IS NOT PRESENT IN THE CODE OR QUESTION."
)

def build_user_prompt(examples, code, question):
    lines = []
    for i, ex in enumerate(examples, 1):
        lines.append(f"--- Example {i} ---")
        lines.append(f"Student's code:\n```python\n{ex['code']}\n```")
        lines.append(f"Student's question: {ex['question']}")
        lines.append(f"TA Answer: {ex['answer']}")
    lines.append("--- End of Examples ---")
    lines.append("")
    lines.append("Now answer the following question in the same style as the examples above.")
    lines.append("")
    lines.append(f"Student's code:\n```python\n{code}\n```")
    lines.append(f"Student's question: {question}")
    lines.append("TA Answer:")
    return "\n".join(lines)

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

def save_stats(path, stats):
    dirname = os.path.dirname(path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device count:   {torch.cuda.device_count()}")
print(f"Loading generator: {GENERATOR_MODEL_ID}")

# --- Memory/time instrumentation: reset peak-memory counter before load ---
if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()
load_start = time.perf_counter()

tokenizer = AutoTokenizer.from_pretrained(
    GENERATOR_MODEL_ID,
    cache_dir=HF_CACHE,
    trust_remote_code=True,
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    GENERATOR_MODEL_ID,
    cache_dir=HF_CACHE,
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True,
)
model.eval()

load_time_s = time.perf_counter() - load_start
peak_mem_after_load_mb = (
    torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else None
)
print(f"Model loaded on: {model.device}")
print(f"Load time: {load_time_s:.2f}s | Peak GPU memory after load: {peak_mem_after_load_mb:.1f} MB\n")

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    dataset = json.load(f)
print(f"Loaded {len(dataset)} records from {INPUT_FILE}\n")

results = []
latencies_s = []

for i, item in enumerate(tqdm(dataset, desc="Generating")):
    code     = item.get("code", "")
    question = item.get("question", "")
    gold     = item.get("answer", "")
    category = item.get("_category", "unknown")
    q_type   = item.get("questionType", category)
    q_id     = item.get("id", f"q{i+1}")

    user_prompt = build_user_prompt(FEW_SHOT_EXAMPLES, code, question)

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

    gen_start = time.perf_counter()
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
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    gen_time_s = time.perf_counter() - gen_start
    latencies_s.append(gen_time_s)

    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    prediction = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    print(f"[{i+1}/{len(dataset)}] category={category} | type={q_type} | latency={gen_time_s:.2f}s")
    print(f"  Q:    {question}")
    print(f"  Gold: {gold}")
    print(f"  Pred: {prediction}\n")

    results.append({
        "id":            q_id,
        "dataset":       "cs1qa",
        "category":      category,
        "question_type": q_type,
        "code":          code,
        "question":      question,
        "answer":        gold,
        "prediction":    prediction,
        "latency_s":     round(gen_time_s, 4),
    })

    if (i + 1) % SAVE_EVERY == 0 or (i + 1) == len(dataset):
        save_append(OUTPUT_FILE, results)
        print(f"  Checkpoint saved at record {i+1}")
        results = []

# --- Final run-level stats, for the accuracy-vs-memory/time tradeoff table ---
peak_mem_overall_mb = (
    torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else None
)
stats = {
    "model_id":                 GENERATOR_MODEL_ID,
    "dataset":                  "cs1qa",
    "n_examples":                len(latencies_s),
    "load_time_s":               round(load_time_s, 2),
    "peak_gpu_mem_after_load_mb": round(peak_mem_after_load_mb, 1) if peak_mem_after_load_mb else None,
    "peak_gpu_mem_overall_mb":    round(peak_mem_overall_mb, 1) if peak_mem_overall_mb else None,
    "avg_latency_s":              round(sum(latencies_s) / len(latencies_s), 4) if latencies_s else None,
    "total_generation_time_s":    round(sum(latencies_s), 2),
}
save_stats(STATS_FILE, stats)
print(f"Stats saved to {STATS_FILE}")
print(json.dumps(stats, indent=2))

print(f"\nDone. Predictions saved to {OUTPUT_FILE}")
