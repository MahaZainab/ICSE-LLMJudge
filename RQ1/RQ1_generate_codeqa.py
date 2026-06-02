#!/usr/bin/env python3

import json
import os
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

GENERATOR_MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
INPUT_FILE         = "CodeQA_sampled.json"
OUTPUT_FILE        = "CodeQA_predictions.json"
SAVE_EVERY         = 25
MAX_NEW_TOKENS     = 128
HF_CACHE           = os.getenv("HF_HOME", "")

FEW_SHOT_EXAMPLES = [
    {
        "code":     "def get_name ( self ) : return self . _name",
        "question": "What does the code return?",
        "answer":   "the name"
    },
    {
        "code":     "def is_empty ( self ) : return len ( self . items ) == 0",
        "question": "Does the code check if the list is empty?",
        "answer":   "Yes"
    },
    {
        "code":     "def sort_list ( items ) : return sorted ( items , reverse = True )",
        "question": "How does the code sort the items?",
        "answer":   "in descending order"
    },
]

SYSTEM_PROMPT = (
    "You are an expert software engineer specializing in source code comprehension. "
    "You will be given a code snippet and a question about that code. "
    "Answer the question directly and concisely based only on the provided code. "
    "Do not repeat the question. "
    "Do not add explanations or information not present in the code. "
    "Study the examples carefully to understand the expected answer style and length."
)

def build_user_prompt(examples, code, question):
    lines = []
    for i, ex in enumerate(examples, 1):
        lines.append(f"--- Example {i} ---")
        lines.append(f"Code:\n{ex['code']}")
        lines.append(f"Question: {ex['question']}")
        lines.append(f"Answer: {ex['answer']}")
    lines.append("--- End of Examples ---")
    lines.append("")
    lines.append("Now answer the following question in the same style as the examples above.")
    lines.append("")
    lines.append(f"Code:\n{code}")
    lines.append(f"Question: {question}")
    lines.append("Answer:")
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

    print(f"[{i+1}/{len(dataset)}] category={category}")
    print(f"  Q:    {question}")
    print(f"  Gold: {gold}")
    print(f"  Pred: {prediction}\n")

    results.append({
        "id":         q_id,
        "dataset":    "codeqa",
        "category":   category,
        "code":       code,
        "question":   question,
        "answer":     gold,
        "prediction": prediction,
    })

    if (i + 1) % SAVE_EVERY == 0 or (i + 1) == len(dataset):
        save_append(OUTPUT_FILE, results)
        print(f"  Checkpoint saved at record {i+1}")
        results = []

print(f"\nDone. Predictions saved to {OUTPUT_FILE}")
