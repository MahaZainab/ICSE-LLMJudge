#!/usr/bin/env python3
"""
RQ1 -- SLM Stats Probe (CodeQA, all 10 models)

Purpose: get real load-time / peak-GPU-memory / per-example-latency numbers
for every SLM in Table 2, WITHOUT re-running full generation+judging on the
whole ~15.7k-example CodeQA set. Runs a 100-example subset per model, just
to measure timing/memory -- predictions are written too (so you can eyeball
sanity), but this is not meant to replace or overwrite your existing full
predictions files.

Reuses, verbatim, from your existing scripts:
  - FEW_SHOT_EXAMPLES, SYSTEM_PROMPT, build_chat_user_prompt()
    (from RQ1_verify_codeqa_5examples.py)
  - Per-model trust_remote_code / stop-token fixes you already debugged for
    Phi-3.5-mini-instruct (<|end|> stop token, trust_remote_code=False) and
    StableCode-Instruct-3B (<|im_end|> stop token, leaked-tag stripping)

Generalizes those two fixes into a broader default terminator set applied
to ALL 10 models, not just the two you'd already caught -- covers Llama-3
(<|eot_id|>), ChatML (<|im_end|>), Phi (<|end|>), and Gemma (<end_of_turn>)
turn-end tokens. Unknown/absent tokens for a given model are silently
skipped (same approach as your existing get_terminators() functions), so
this is a safe superset, not a behavior change for models that were already
working correctly with just eos_token_id.

Outputs one *_stats.json per model into OUTPUT_DIR, in the same schema as
the big-LLM stats files (RQ_bigLLM_generate_*.py), so
make_bigllm_stats_table.py picks all of them up in one aggregated table
automatically -- no changes needed to that script.

Usage:
  python RQ1_stats_codeqa_alltier.py
  python RQ1_stats_codeqa_alltier.py --model gemma2_9b_it
"""

import argparse
import gc
import json
import os
import re
import time

import torch
import transformers
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# ── CONFIG ────────────────────────────────────────────────────────────────────
INPUT_FILE       = "filtered_dataset.json"   # <-- CHANGE if your full CodeQA file has a different name
OUTPUT_DIR       = "RQ1_stats_codeqa"
PROBE_SIZE       = 100        # examples per model, for a stable avg latency
FULL_DATASET_N   = 15754      # for the projected full-dataset time estimate (matches the big-LLM runs)
MAX_NEW_TOKENS   = 128        # matches RQ1 CodeQA generation settings
HF_CACHE         = os.getenv("HF_HOME", "")

os.makedirs(OUTPUT_DIR, exist_ok=True)

MIN_TRANSFORMERS_VERSION = (4, 44)  # first release with native Phi-3 support

# ── MODEL REGISTRY ─────────────────────────────────────────────────────────────
# (short_key, hf_model_id, trust_remote_code, params_billions)
# trust_remote_code=False for Phi-3.5 matches the fix in RQ1_generate_cs1qa_phi.py
# (avoids the DynamicCache.seen_tokens crash from its stale custom modeling code).
MODELS = [
    ("llama3.2_1b_instruct",         "meta-llama/Llama-3.2-1B-Instruct",         True,  1.0),
    ("deepseek_coder_1.3b_instruct", "deepseek-ai/deepseek-coder-1.3b-instruct", True,  1.3),
    ("gemma2_2b_it",                 "google/gemma-2-2b-it",                     True,  2.0),
    ("llama3.2_3b_instruct",         "meta-llama/Llama-3.2-3B-Instruct",         True,  3.0),
    ("phi3.5_mini_instruct",         "microsoft/Phi-3.5-mini-instruct",          False, 3.8),
    ("stablecode_instruct_3b",       "stabilityai/stable-code-instruct-3b",      True,  3.0),
    ("llama3.1_8b_instruct",         "meta-llama/Llama-3.1-8B-Instruct",         True,  8.0),
    ("codellama_7b_instruct",        "codellama/CodeLlama-7b-Instruct-hf",       True,  7.0),
    ("gemma2_9b_it",                 "google/gemma-2-9b-it",                     True,  9.0),
    ("deepseek_coder_6.7b_instruct", "deepseek-ai/deepseek-coder-6.7b-instruct", True,  6.7),
]

# ── FEW-SHOT EXAMPLES (identical to RQ1_verify_codeqa_5examples.py) ───────────
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

# ── SYSTEM PROMPT (identical to RQ1_verify_codeqa_5examples.py) ───────────────
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

# ── HELPERS ───────────────────────────────────────────────────────────────────
def build_chat_user_prompt(examples, code, question):
    """Identical to RQ1_verify_codeqa_5examples.py's build_chat_user_prompt."""
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


def get_terminators(tokenizer):
    """
    Generalized version of the fix applied in RQ1_generate_cs1qa_phi.py and
    RQ1_verify_codeqa_5examples_stablecode.py: resolve a superset of known
    chat-turn-end tokens across model families, in addition to the base
    eos_token_id, so no model's stop condition is silently wrong the way
    Phi-3.5 and StableCode's were in the original all-in-one script. Unknown
    tokens for a given tokenizer resolve to unk_token_id and are filtered
    out, so this is a safe no-op for models that didn't need the fix.
    """
    ids = set()
    if tokenizer.eos_token_id is not None:
        ids.add(tokenizer.eos_token_id)

    candidate_tokens = [
        "<|end|>",          # Phi-3.5
        "<|im_end|>",       # ChatML (StableCode, etc.)
        "<|eot_id|>",       # Llama-3 family
        "<end_of_turn>",    # Gemma-2 family
        "<|endoftext|>",
    ]
    for tok in candidate_tokens:
        try:
            tid = tokenizer.convert_tokens_to_ids(tok)
            if tid is not None and tid != tokenizer.unk_token_id:
                ids.add(tid)
        except Exception:
            pass

    return list(ids)


LEAKED_TAG_PATTERN = re.compile(r"<\|im_(end|start)\|>|<\|end\|>|<\|eot_id\|>|<end_of_turn>|<\|endoftext\|>")

def strip_leaked_special_tokens(text):
    """Belt-and-suspenders cleanup, same approach as the StableCode fix script."""
    cleaned = LEAKED_TAG_PATTERN.sub("", text).strip()
    return cleaned, cleaned != text.strip()


def unload_model(model, tokenizer):
    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def run_probe(short_key, model_id, trust_remote_code, params_b, dataset):
    print(f"\n{'='*60}")
    print(f"Model  : {model_id}  ({params_b}B)")
    print(f"Probe N: {len(dataset)}")
    print(f"{'='*60}")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    load_start = time.perf_counter()

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, cache_dir=HF_CACHE, trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id, cache_dir=HF_CACHE, torch_dtype="auto",
        device_map="auto", trust_remote_code=trust_remote_code,
    )
    model.eval()

    load_time_s = time.perf_counter() - load_start
    peak_mem_after_load_mb = (
        torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else None
    )
    print(f"  Loaded on: {model.device} | load_time={load_time_s:.2f}s "
          f"| peak_mem_after_load={peak_mem_after_load_mb:.1f}MB")

    terminators = get_terminators(tokenizer)

    latencies_s = []
    predictions = []

    for i, item in enumerate(tqdm(dataset, desc=f"Probing [{short_key}]")):
        code     = item.get("code", "")
        question = item.get("question", "")
        gold     = item.get("answer", "")
        category = item.get("_category", item.get("category", "unknown"))
        q_id     = item.get("id", f"q{i+1}")

        user_prompt = build_chat_user_prompt(FEW_SHOT_EXAMPLES, code, question)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ]
        try:
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        except Exception:
            formatted = (
                f"### System:\n{SYSTEM_PROMPT}\n\n"
                f"### User:\n{user_prompt}\n\n"
                f"### Assistant:\n"
            )

        inputs = tokenizer(
            formatted, return_tensors="pt", padding=True,
            truncation=True, max_length=4096,
        ).to(model.device)

        gen_start = time.perf_counter()
        try:
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    top_k=None,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=terminators,
                )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            gen_time_s = time.perf_counter() - gen_start
            new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
            prediction = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            prediction, _ = strip_leaked_special_tokens(prediction)
        except Exception as e:
            print(f"  ERROR at record {i+1} ({q_id}): {e}")
            gen_time_s = time.perf_counter() - gen_start
            prediction = ""

        latencies_s.append(gen_time_s)
        predictions.append({
            "id": q_id, "category": category, "question": question,
            "answer": gold, "prediction": prediction, "latency_s": round(gen_time_s, 4),
        })

    peak_mem_overall_mb = (
        torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else None
    )
    avg_latency_s = sum(latencies_s) / len(latencies_s) if latencies_s else None
    projected_full_h = (
        (avg_latency_s * FULL_DATASET_N) / 3600 if avg_latency_s else None
    )

    stats = {
        "model_id":                    model_id,
        "params_b":                    params_b,
        "dataset":                     "codeqa",
        "n_examples":                  len(dataset),
        "probe_only":                  True,
        "load_time_s":                 round(load_time_s, 2),
        "peak_gpu_mem_after_load_mb":  round(peak_mem_after_load_mb, 1) if peak_mem_after_load_mb else None,
        "peak_gpu_mem_overall_mb":     round(peak_mem_overall_mb, 1) if peak_mem_overall_mb else None,
        "avg_latency_s":               round(avg_latency_s, 4) if avg_latency_s else None,
        "total_generation_time_s":     round(sum(latencies_s), 2),
        "projected_full_dataset_time_h": round(projected_full_h, 2) if projected_full_h else None,
        "projected_full_dataset_n":    FULL_DATASET_N,
    }

    stats_path = os.path.join(OUTPUT_DIR, f"{short_key}_codeqa_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"  Saved {stats_path}")
    print(json.dumps(stats, indent=2))

    preds_path = os.path.join(OUTPUT_DIR, f"{short_key}_codeqa_probe_predictions.json")
    with open(preds_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)

    unload_model(model, tokenizer)
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None,
                         help="Short key of a single model to probe (e.g. gemma2_9b_it). "
                              "If omitted, all 10 models run sequentially.")
    args = parser.parse_args()

    ver_str = transformers.__version__
    ver_tuple = tuple(int(x) for x in ver_str.split(".")[:2])
    print(f"transformers version: {ver_str}")
    if ver_tuple < MIN_TRANSFORMERS_VERSION:
        print(f"  WARNING: transformers {ver_str} may predate native Phi-3 support "
              f"(need >= {'.'.join(map(str, MIN_TRANSFORMERS_VERSION))}).")

    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Device count:   {torch.cuda.device_count()}")

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        full_dataset = json.load(f)
    print(f"Full dataset size: {len(full_dataset)} records")

    subset = full_dataset[:PROBE_SIZE]
    print(f"Probing on first {len(subset)} records\n")

    targets = MODELS
    if args.model:
        targets = [m for m in MODELS if m[0] == args.model]
        if not targets:
            valid = [m[0] for m in MODELS]
            raise ValueError(f"Unknown model key '{args.model}'. Valid keys: {valid}")

    all_stats = []
    for short_key, model_id, trust_remote_code, params_b in targets:
        stats = run_probe(short_key, model_id, trust_remote_code, params_b, subset)
        all_stats.append(stats)

    print("\n" + "="*60)
    print(f"Probed {len(all_stats)} model(s). Stats files in: {OUTPUT_DIR}")
    print("Run make_bigllm_stats_table.py on this folder (or merge folders) to build the combined table.")
    print("="*60)


if __name__ == "__main__":
    main()
