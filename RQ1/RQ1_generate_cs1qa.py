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
    "You are a teaching assistant (TA) for an introductory Python programming course. "
    "A student has asked you a question about their code. "
    "Respond in the style of a human TA during office hours: "
    "keep answers short and conversational (1–4 sentences), "
    "explain the concept or point out the issue rather than rewriting the student's code for them, "
    "and nudge the student toward the fix without handing it to them directly. "
    "You may confirm when a student's understanding is correct, "
    "clarify a concept with a brief explanation, "
    "or hint at what to look for — but avoid providing complete corrected code. "
    "Base your answer only on the provided code and question. "
    "Do not repeat the question."
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
