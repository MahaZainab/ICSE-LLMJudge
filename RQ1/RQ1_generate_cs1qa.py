#!/usr/bin/env python3

import json
import os
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

GENERATOR_MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
INPUT_FILE         = "CS1QA_sampled.json"
OUTPUT_FILE        = "CS1QA_predictions.json"
SAVE_EVERY         = 25
MAX_NEW_TOKENS     = 256
HF_CACHE           = os.getenv("HF_HOME", "")

FEW_SHOT_EXAMPLES = [
    {
        "code": (
            "for i in range(10):\n"
            "    print(i)\n"
            "    i = i + 2"
        ),
        "question": "Why is my loop not skipping by 2?",
        "answer": (
            "In a for loop, Python reassigns i automatically on each iteration, "
            "so setting i = i + 2 inside the loop has no effect. "
            "Use range(0, 10, 2) instead."
        ),
    },
    {
        "code": (
            "def add(a, b):\n"
            "    return a + b\n"
            "result = add(3, 4)\n"
            "print(result)"
        ),
        "question": "Is it okay to call the function before defining it?",
        "answer": (
            "No. In Python you must define the function before calling it. "
            "Move the function definition above the line where you call it."
        ),
    },
    {
        "code": (
            "numbers = [1, 2, 3, 4, 5]\n"
            "total = 0\n"
            "for n in numbers:\n"
            "    total += n\n"
            "print(total)"
        ),
        "question": "What does this code do?",
        "answer": (
            "It calculates the sum of all numbers in the list and prints the result, which is 15."
        ),
    },
]

SYSTEM_PROMPT = (
    "You are a teaching assistant (TA) for an introductory Python programming course. "
    "A student has asked you a question about their code. "
    "Provide a clear, helpful, and concise answer appropriate for a beginner programmer. "
    "Base your answer only on the provided code and question. "
    "Do not repeat the question. "
    "Study the examples carefully to understand the expected answer style and tone."
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

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device count:   {torch.cuda.device_count()}")
print(f"Loading generator: {GENERATOR_MODEL_ID}")

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
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)
model.eval()
print(f"Model loaded on: {model.device}\n")

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    dataset = json.load(f)
print(f"Loaded {len(dataset)} records from {INPUT_FILE}\n")

results = []

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
    prediction = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    print(f"[{i+1}/{len(dataset)}] category={category} | type={q_type}")
    print(f"  Q:    {question}")
    print(f"  Gold: {gold}")
    print(f"  Pred: {prediction}\n")

    results.append({
        "id":           q_id,
        "dataset":      "cs1qa",
        "category":     category,
        "question_type": q_type,
        "code":         code,
        "question":     question,
        "answer":       gold,
        "prediction":   prediction,
    })

    if (i + 1) % SAVE_EVERY == 0 or (i + 1) == len(dataset):
        save_append(OUTPUT_FILE, results)
        print(f"  Checkpoint saved at record {i+1}")
        results = []

print(f"\nDone. Predictions saved to {OUTPUT_FILE}")
