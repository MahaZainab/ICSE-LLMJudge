#!/usr/bin/env python3

import json
import os
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

GENERATOR_MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
INPUT_FILE         = "../DataPreprocessing/filtered_dataset.json"
OUTPUT_FILE        = "RQ1_generate_codeqa/filtered/llama3.2_3b_instruct_predictions.json"
SAVE_EVERY         = 25
MAX_NEW_TOKENS     = 128
HF_CACHE           = os.getenv("HF_HOME", "")

FEW_SHOT_EXAMPLES = [
    {
        "code":     "def sort_list ( items ) : return sorted ( items , reverse = True )",
        "question": "How does the code sort the items?",
        "answer":   "in descending order"
    },
    {
        "code":     "def on_timeout ( self ) : self . retries += 1\nif self . retries > 3 : self . stop ( )",
        "question": "When does the code stop retrying?",
        "answer":   "after more than 3 retries"
    },
    {
        "code":     "def save_log ( entry ) : with open ( 'logs/app.log' , 'a' ) as f : f . write ( entry )",
        "question": "Where does the code write the log entry?",
        "answer":   "to the file logs/app.log"
    },
    {
        "code":     "def validate_input ( data ) : if not data : raise ValueError ( 'empty input' )",
        "question": "For what purpose does the code raise a ValueError?",
        "answer":   "to signal that the input data is empty"
    },
    {
        "code":     "def cache_result ( self , key , value ) : self . _cache [ key ] = value",
        "question": "Why does the code store the value in a dictionary?",
        "answer":   "to cache the result for later lookup by key"
    },
]

SYSTEM_PROMPT = (
    "You are an expert software engineer with deep experience in source code comprehension, "
    "code review, and software documentation in Python.\n"
    "You will be given a Python code snippet and a natural language question about that code.\n"
    "\n"
    "Your task:\n"
    "- Read the code carefully before answering.\n"
    "- Answer the question directly and concisely based solely on what the code does.\n"
    "- Match the style and length of the examples provided — a short phrase or single sentence is expected.\n"
    "\n"
    "IMPORTANT: Study the examples carefully before answering.\n"
    "IMPORTANT: The examples define the expected answer style, format, and length — match them exactly.\n"
    "IMPORTANT: If the question asks WHAT the code returns, answer with the thing that is returned, not a description of how it works.\n"
    "\n"
    "CRITICAL: DO NOT REPEAT OR PARAPHRASE THE QUESTION IN YOUR ANSWER.\n"
    "CRITICAL: DO NOT ADD EXPLANATIONS OR INFORMATION NOT PRESENT IN THE CODE.\n"
    "CRITICAL: DO NOT PRODUCE A PARAGRAPH WHEN A PHRASE IS SUFFICIENT.\n"
    "CRITICAL: DO NOT HALLUCINATE — base your answer ONLY on the provided code snippet."
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
    torch_dtype="auto",
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
