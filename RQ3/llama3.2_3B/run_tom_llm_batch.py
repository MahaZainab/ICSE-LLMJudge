"""
Batch runner — ToM-LLM condition (large 7B teacher).
Grant-plot task: No-Intervention (already run) vs ToM-LLM.

Single input file: the RQ2 open-coding file (nested {case: {bucket: [record,
...]}}) already has code, question, reference answer, the No-Intervention
first_answer ("prediction"), and its judge scores. Stages 1-2 (student draft
+ baseline judge) are NOT redone — they're loaded straight from that file.

Pipeline per item:
  [precomputed] first_answer + scores_no_intervention
                              -> [Teacher-LLM 7B] -> ToM diagnosis
                                                   -> [Student] -> revised_answer -> [Judge] -> scores_tom_llm

Usage:
    python run_tom_llm_batch.py --limit 40    # quick test run
    python run_tom_llm_batch.py               # full set

Output: JSON list written to results/tom_llm_codeqa.json, one record per item:
    {
      "id": "q16",
      "dataset": "CodeQA",
      "category": "...",
      "code": "...",
      "question": "...",
      "reference_answer": "...",
      "first_answer": "...",
      "scores_no_intervention": {"accuracy": int|None, "completeness": int|None, "clarity": int|None, "relevance": int|None},
      "teacher_analysis": {"intent": "...", "misconception": "...", "understanding": "...", "guidance": "..."},
      "revised_answer": "...",
      "scores_tom_llm": {"accuracy": int|None, "completeness": int|None, "clarity": int|None, "relevance": int|None}
    }
"""

import argparse
import gc
import json
import os
import time

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
STUDENT_MODEL_ID     = "meta-llama/Llama-3.2-3B-Instruct"
TEACHER_LLM_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
JUDGE_MODEL_ID        = "Qwen/Qwen2.5-Coder-7B-Instruct"

MAX_TOKENS_TEACHER = 420
MAX_TOKENS_REVISED = 128
MAX_TOKENS_JUDGE   = 128

DATASET_NAME = "CodeQA"
OUT_DIR = "results"

# Precomputed No-Intervention pass (first_answer + judge scores), already run —
# reused here instead of recomputing Stage 1 (student draft) and Stage 2
# (baseline judge). Nested as {case: {bucket: [record, ...]}} in the source
# file; flattened by id at load time.
PRECOMPUTED_PATH = "llama32_3b_instruct_open_coding.json"

TEACHER_FEW_SHOT = {
    "code": "def get_average ( scores ) : return sum ( scores ) / len ( scores ) if scores else 0",
    "question": "Why does the code check if scores is empty?",
    "first_answer": "It checks the length of the scores list.",
    "response": (
        "QUESTION INTENT: The question asks WHY the empty check exists — the causal "
        "reason/purpose behind it, not a description of what the check does mechanically.\n"
        "MISCONCEPTION HYPOTHESIS: The student's answer restates WHAT the code does "
        "(checks the length) rather than WHY it does it. This suggests the student read "
        "the condition literally but did not trace it forward to its consequence — a "
        "ZeroDivisionError if scores is empty and len(scores) is used as a divisor. "
        "The student described the mechanism instead of the cause.\n"
        "STUDENT UNDERSTANDING: The student correctly identified the code being asked about "
        "but answered the WHAT instead of the WHY — they have not yet reasoned about what "
        "happens on the line below the check.\n"
        "GUIDANCE: Look at the return statement — scores is used as a divisor via len(scores). "
        "Ask yourself what would happen on that line if scores were empty, and let that consequence "
        "be the reason in your answer, not the check itself."
    ),
}

TEACHER_SYSTEM = (
    "You are an expert tutor coaching a student who is answering a source code comprehension question.\n"
    "You will be given a code snippet, a question, and the student's first answer.\n"
    "There is NO reference answer. Reason about the code yourself.\n\n"
    "Respond using EXACTLY these four labeled sections and nothing else:\n\n"
    "QUESTION INTENT: what the question is really asking for and what kind of answer would satisfy it "
    "(e.g. WHAT is returned vs HOW it works vs WHY it exists).\n"
    "MISCONCEPTION HYPOTHESIS: infer the likely reasoning process behind the student's answer — "
    "not just whether it is right or wrong, but WHY they likely arrived at it (e.g. read the wrong "
    "variable, described the mechanism instead of the cause, confused a related but distinct part "
    "of the code, stopped reasoning one step too early). Name the likely process, not just the gap.\n"
    "STUDENT UNDERSTANDING: what the student's first answer reveals — what they got right and what they missed.\n"
    "GUIDANCE: specific, targeted hints pointing the student to the exact part of the code "
    "they should focus on and how their answer should change.\n\n"
    "CRITICAL: DO NOT WRITE THE STUDENT'S FINAL ANSWER — diagnose and direct only.\n"
    "CRITICAL: USE EXACTLY THE FOUR LABELED SECTIONS — no preamble, no extra text, no markdown.\n\n"
    f"--- Example ---\n"
    f"Code:\n{TEACHER_FEW_SHOT['code']}\n\n"
    f"Question:\n{TEACHER_FEW_SHOT['question']}\n\n"
    f"Student's first answer:\n{TEACHER_FEW_SHOT['first_answer']}\n\n"
    f"{TEACHER_FEW_SHOT['response']}\n"
    f"--- End of Example ---"
)

STUDENT_REVISED_FEW_SHOT = {
    "code": "def get_average ( scores ) : return sum ( scores ) / len ( scores ) if scores else 0",
    "question": "Why does the code check if scores is empty?",
    "first_answer": "It checks the length of the scores list.",
    "guidance": (
        "GUIDANCE: Look at the return statement — scores is used as a divisor via len(scores). "
        "Ask yourself what would happen on that line if scores were empty, and let that consequence "
        "be the reason in your answer, not the check itself."
    ),
    "revised_answer": "to avoid a division by zero when the list is empty",
}

STUDENT_REVISED_SYSTEM = (
    "You are an expert software engineer answering a source code comprehension question.\n"
    "A tutor has reviewed your first answer and provided an analysis with targeted guidance.\n"
    "Use the tutor's guidance to write an improved answer.\n\n"
    "Your task:\n"
    "- Read the code, your first answer, and the tutor's guidance carefully before revising.\n"
    "- Answer the question directly and concisely based solely on what the code does.\n"
    "- Match the style and length of the example provided — a short phrase or single sentence is expected.\n\n"
    "IMPORTANT: Keep the answer concise — a short phrase or single sentence, same style as a normal CodeQA answer.\n"
    "IMPORTANT: If the question asks WHAT the code returns, answer with the thing that is returned, not a description of how it works.\n\n"
    "CRITICAL: BASE YOUR ANSWER ONLY ON THE PROVIDED CODE AND THE TUTOR'S GUIDANCE.\n"
    "CRITICAL: DO NOT REPEAT OR PARAPHRASE THE QUESTION IN YOUR ANSWER.\n"
    "CRITICAL: DO NOT ADD EXPLANATIONS OR INFORMATION NOT PRESENT IN THE CODE.\n"
    "CRITICAL: DO NOT PRODUCE A PARAGRAPH WHEN A PHRASE IS SUFFICIENT.\n"
    "CRITICAL: DO NOT DESCRIBE WHAT THE TUTOR SAID — INCORPORATE IT SILENTLY INTO YOUR ANSWER.\n"
    "CRITICAL: OUTPUT ONLY THE IMPROVED ANSWER — no preamble, no explanation, no markdown.\n\n"
    "--- Example ---\n"
    f"Code:\n{STUDENT_REVISED_FEW_SHOT['code']}\n\n"
    f"Question: {STUDENT_REVISED_FEW_SHOT['question']}\n\n"
    f"Your first answer: {STUDENT_REVISED_FEW_SHOT['first_answer']}\n\n"
    f"Tutor's guidance:\n{STUDENT_REVISED_FEW_SHOT['guidance']}\n\n"
    f"Improved Answer: {STUDENT_REVISED_FEW_SHOT['revised_answer']}\n"
    "--- End of Example ---"
)

JUDGE_SYSTEM = """You are an expert software engineer and code evaluator with deep experience assessing the quality of answers to source code comprehension questions.

You will be given a code snippet, a question about that code, a reference answer, and a predicted answer.
Your task is to evaluate the predicted answer across four dimensions: Accuracy, Completeness, Clarity, and Relevance.

Dataset context:
The questions come from CodeQA, a free-form question-answering benchmark built from real Python and Java code on GitHub.
Questions are derived from code comments (docstrings, Javadocs) and cover four types of code understanding:
- Functionality: what the code does, returns, creates, or produces
- Purpose: why the code exists or what problem it solves
- Property: attributes, parameters, types, conditions, or constraints in the code
- Workflow: how the code operates step-by-step or how data flows through it
Correct answers in CodeQA are typically concise — often a short phrase or a single sentence, not a paragraph.

Input sections:
<code>
The source code snippet to evaluate
</code>

<question>
The specific question about the code
</question>

<reference_answer>
The ground-truth answer for the question
</reference_answer>

<predicted_answer>
The answer generated by the model
</predicted_answer>

Scoring dimensions (score each independently on a 1 to 5 integer scale):

ACCURACY — Does the predicted answer correctly match the reference answer's meaning and reflect what the code actually does?
  5: Completely correct — semantically equivalent to the reference answer and fully consistent with the code
  4: Mostly correct — minor factual slip or wording difference that does not change the core meaning
  3: Partially correct — captures something true but misses or misstates a key detail from the reference answer
  2: Mostly incorrect — contains a relevant element but is dominated by factual errors or gives the wrong specific reason
  1: Completely wrong — contradicts the reference answer/code, gives an unsupported reason, or addresses something entirely different

COMPLETENESS — Does the predicted answer cover everything the reference answer and question require?
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

RELEVANCE — Does the predicted answer directly target what the question and reference answer are asking for?
  5: Fully relevant — directly and precisely answers the exact question asked
  4: Mostly relevant — minor tangent that does not distract from the answer
  3: Partially relevant — discusses the same code but answers a related or different question
  2: Mostly irrelevant — misses the main point or specific reason asked by the question
  1: Completely irrelevant — does not address the question at all

IMPORTANT: Use the reference answer as the gold answer, but read the code carefully to understand and verify its meaning.
IMPORTANT: Semantic equivalence counts as correct. "the name", "Name of the user", and "it returns the user's name" can all be correct for the same question if they match the reference answer's meaning and the code supports them.
IMPORTANT: Score each dimension INDEPENDENTLY. A clear answer can still be inaccurate. An accurate statement can still be irrelevant if it does not answer the specific question.
IMPORTANT: A short answer is NOT incomplete if it fully addresses the question — CodeQA answers are intentionally concise.
IMPORTANT: Do NOT give a high score just because the predicted answer is fluent, clear, or grammatically correct.
IMPORTANT: Do NOT give a high score just because the predicted answer mentions something from the code.
IMPORTANT: Do NOT give a high Accuracy or Relevance score if the answer is only generally related to the code but misses what the question and reference answer specifically ask.
IMPORTANT: Do NOT infer missing reasoning that is not stated in the predicted answer.
IMPORTANT: If the question asks WHAT, the answer should identify the requested object, value, behavior, return value, or result.
IMPORTANT: If the question asks HOW, the answer should describe the correct process, mechanism, or workflow.
IMPORTANT: If the question asks WHY or FOR WHAT PURPOSE, the answer must give the correct specific cause, reason, purpose, or design rationale reflected in the reference answer and supported by the code.
IMPORTANT: If the question asks WHERE or WHEN, the answer should identify the correct location, condition, timing, or situation in the code.

WHY-QUESTION AND CAUSAL REASONING RULES:
Many difficult samples are WHY or PURPOSE questions.
For WHY questions, Accuracy depends on whether the predicted answer gives the correct specific cause, reason, or purpose asked by the question and reflected in the reference answer.
Do NOT give high Accuracy or Relevance scores to answers that are only generally related to the code but miss the specific reason.
Generic causal answers should be penalized when they do not identify the actual reason in the reference answer/code.
Repeated generic answers such as "because it will be handled by the parent process" should receive low Accuracy and low Relevance unless that exact explanation is clearly supported by the reference answer and code.
If the answer is clear but gives the wrong cause, Clarity may be high, but Accuracy and Relevance should be low.

CRITICAL: USE THE REFERENCE ANSWER AS THE GOLD ANSWER. Evaluate whether the predicted answer is semantically equivalent to the reference answer and supported by the code.
CRITICAL: DO NOT PENALIZE AN ANSWER FOR PHRASING OR VOCABULARY DIFFERENCES IF THE MEANING MATCHES THE REFERENCE ANSWER.
CRITICAL: DO NOT CONFLATE DIMENSIONS — score CLARITY independently of ACCURACY.
CRITICAL: DO NOT HALLUCINATE FACTS ABOUT THE CODE. If you are uncertain, score conservatively.
CRITICAL: DO NOT PRODUCE ANY TEXT OUTSIDE THE JSON OBJECT — no explanation, no preamble, no reasoning, no markdown.

Calibration examples:

Example 1 — fully correct, concise answer:
Code: def get_suite ( self , suite_dict , label = None ) : suite = unittest.TestSuite ( ) for test_name in suite_dict : suite.addTest ( self.get_test ( test_name ) ) return suite
Question: What does the code return?
Reference Answer: a test suite
Predicted Answer: a test suite
Output: {"accuracy": {"score": 5}, "completeness": {"score": 5}, "clarity": {"score": 5}, "relevance": {"score": 5}}

Example 2 — accurate but answers HOW instead of WHAT:
Code: def get_suite ( self , suite_dict , label = None ) : suite = unittest.TestSuite ( ) for test_name in suite_dict : suite.addTest ( self.get_test ( test_name ) ) return suite
Question: What does the code return?
Reference Answer: a test suite
Predicted Answer: It creates a new TestSuite object and iterates through the suite_dict to add each test by name before returning the populated suite object.
Output: {"accuracy": {"score": 5}, "completeness": {"score": 4}, "clarity": {"score": 5}, "relevance": {"score": 3}}

Example 3 — factually wrong but clearly written and partially on-topic:
Code: def is_valid_age ( age ) : return isinstance ( age , int ) and age >= 0 and age <= 120
Question: What does the code check?
Reference Answer: whether the age is an integer between 0 and 120
Predicted Answer: whether the input is a string
Output: {"accuracy": {"score": 1}, "completeness": {"score": 2}, "clarity": {"score": 5}, "relevance": {"score": 3}}

Example 4 — related to the code but does not answer the specific question:
Code: def save_file ( path , data ) : with open ( path , "w" ) as f : f.write ( data ) return True
Question: What does the function return?
Reference Answer: True
Predicted Answer: It writes data to a file at the given path.
Output: {"accuracy": {"score": 2}, "completeness": {"score": 2}, "clarity": {"score": 5}, "relevance": {"score": 3}}

Example 5 — WHY question with generic or unsupported reasoning:
Code: def cleanup ( parent , child ) : if child.parent == parent : parent.remove ( child )
Question: Why does the code check whether child.parent equals parent?
Reference Answer: to make sure the child belongs to that parent before removing it
Predicted Answer: because it will be handled by the parent process
Output: {"accuracy": {"score": 2}, "completeness": {"score": 2}, "clarity": {"score": 4}, "relevance": {"score": 2}}

Example 6 — WHY question with correct specific reason:
Code: def cleanup ( parent , child ) : if child.parent == parent : parent.remove ( child )
Question: Why does the code check whether child.parent equals parent?
Reference Answer: to make sure the child belongs to that parent before removing it
Predicted Answer: It verifies that the child is actually associated with the given parent before calling remove.
Output: {"accuracy": {"score": 5}, "completeness": {"score": 5}, "clarity": {"score": 5}, "relevance": {"score": 5}}

Respond ONLY with a valid JSON object in exactly this format:
{
  "accuracy":     {"score": <1-5>},
  "completeness": {"score": <1-5>},
  "clarity":      {"score": <1-5>},
  "relevance":    {"score": <1-5>}
}"""


# ----------------------------------------------------------------------------
# Model utilities
# ----------------------------------------------------------------------------
def load_model(model_id, label):
    print(f"[{label}] Loading tokenizer...")
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    print(f"[{label}] Loading model weights...")
    mdl = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype="auto", device_map="auto", trust_remote_code=True
    )
    mdl.eval()
    print(f"[{label}] Ready on {mdl.device}")
    return tok, mdl


def unload(model, label):
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"[{label}] Unloaded, VRAM freed.")


def run_model(model, tokenizer, system_prompt, user_prompt, max_new_tokens):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    try:
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        formatted = (
            f"### System:\n{system_prompt}\n\n### User:\n{user_prompt}\n\n### Assistant:\n"
        )
    inputs = tokenizer(
        formatted, return_tensors="pt", padding=True, truncation=True, max_length=4096,
    ).to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=False, temperature=None, top_p=None,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def run_judge(model, tokenizer, code, question, reference, prediction):
    user_prompt = (
        f"Code:\n{code}\n\nQuestion:\n{question}\n\n"
        f"Reference Answer:\n{reference}\n\nPredicted Answer:\n{prediction}"
    )
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": user_prompt},
    ]
    try:
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        formatted = f"### System:\n{JUDGE_SYSTEM}\n\n### User:\n{user_prompt}\n\n### Assistant:\n"
    inputs = tokenizer(
        formatted, return_tensors="pt", padding=True, truncation=True, max_length=8192,
    ).to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs, max_new_tokens=MAX_TOKENS_JUDGE,
            do_sample=False, use_cache=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    try:
        start = response.find("{")
        end = response.rfind("}") + 1
        parsed = json.loads(response[start:end])
    except (json.JSONDecodeError, ValueError):
        print(f"  [judge parse warning] {response[:150]}")
        return {"accuracy": None, "completeness": None, "clarity": None, "relevance": None}
    out = {}
    for key in ("accuracy", "completeness", "clarity", "relevance"):
        details = parsed.get(key, {})
        score = details.get("score") if isinstance(details, dict) else details
        out[key] = score if isinstance(score, int) and 1 <= score <= 5 else None
    return out


def build_teacher_user(code, question, first):
    return f"Code:\n{code}\n\nQuestion:\n{question}\n\nStudent's first answer:\n{first}"


def build_student_revised_user(code, question, first, teacher_analysis):
    return (
        f"Code:\n{code}\n\nQuestion:\n{question}\n\n"
        f"Your first answer:\n{first}\n\nTutor's analysis and guidance:\n{teacher_analysis}\n\nImproved Answer:"
    )


def parse_teacher(text):
    labels = {
        "intent": "QUESTION INTENT:",
        "misconception": "MISCONCEPTION HYPOTHESIS:",
        "understanding": "STUDENT UNDERSTANDING:",
        "guidance": "GUIDANCE:",
    }
    out = {k: "" for k in labels}
    positions = {k: text.find(v) for k, v in labels.items()}
    for k, idx in positions.items():
        if idx == -1:
            continue
        start = idx + len(labels[k])
        ends = [p for p in positions.values() if p > idx]
        end = min(ends) if ends else len(text)
        out[k] = text[start:end].strip()
    return out


# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------
def load_items(path, limit=None):
    """Build items directly from the open-coding file — no separate
    CodeQA_clean_final_v2.json needed. That file is nested as
    {case: {bucket: [record, ...]}}; flatten it, and since each record
    already carries 'prediction' (== first_answer) and the four judge
    dimensions (== scores_no_intervention) from the earlier run, populate
    those here too so Stages 1-2 below don't need to be redone.
    """
    with open(path, "r") as f:
        raw = json.load(f)

    def walk(obj):
        if isinstance(obj, dict):
            for v in obj.values():
                yield from walk(v)
        elif isinstance(obj, list):
            yield from obj

    items = []
    seen_ids = set()
    for rec in walk(raw):
        if rec["id"] in seen_ids:
            raise ValueError(f"Duplicate id '{rec['id']}' in {path} — ids must be unique.")
        seen_ids.add(rec["id"])
        items.append({
            "id": rec["id"],
            "dataset": DATASET_NAME,
            "category": rec.get("category", ""),
            "code": rec["code"],
            "question": rec["question"],
            "reference_answer": rec["answer"],
            "first_answer": rec["prediction"],
            "scores_no_intervention": {
                k: rec.get(k) for k in ("accuracy", "completeness", "clarity", "relevance")
            },
        })
    if limit:
        items = items[:limit]
    return items


# ----------------------------------------------------------------------------
# Main batch loop
# ----------------------------------------------------------------------------
def checkpoint(results, out_path):
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                         help="cap number of items (for a quick test run)")
    parser.add_argument("--save-every", type=int, default=10,
                         help="checkpoint results to disk every N items")
    parser.add_argument("--precomputed-path", type=str, default=PRECOMPUTED_PATH,
                         help="path to the already-run No-Intervention file "
                              "(code, question, reference, first_answer, and "
                              "judge scores) — the only input file needed; "
                              "Stages 1-2 are skipped since it's already run")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "tom_llm_codeqa.json")

    # ---- Stages 1-2 (student draft + baseline judge) are already done ----
    # load_items reads them straight out of the precomputed file below.
    all_items = load_items(args.precomputed_path, limit=args.limit)
    print(f"Loaded {len(all_items)} CodeQA items (first_answer + "
          f"scores_no_intervention already populated from {args.precomputed_path})")

    results = [{**item, "teacher_analysis": None, "revised_answer": None,
                "scores_tom_llm": None} for item in all_items]
    checkpoint(results, out_path)

    # ---- Stage 3: Teacher-LLM diagnoses every item's first_answer ----
    teacher_tok, teacher_model = load_model(TEACHER_LLM_MODEL_ID, "Teacher-LLM")
    for i, r in enumerate(results):
        t0 = time.time()
        teacher_raw = run_model(
            teacher_model, teacher_tok, TEACHER_SYSTEM,
            build_teacher_user(r["code"], r["question"], r["first_answer"]), MAX_TOKENS_TEACHER,
        )
        r["_teacher_raw"] = teacher_raw  # kept temporarily for stage 4 prompt
        r["teacher_analysis"] = parse_teacher(teacher_raw)
        if (i + 1) % args.save_every == 0:
            checkpoint(results, out_path)
        print(f"[stage3 teacher {i+1}/{len(results)}] {r['id']} ({time.time()-t0:.1f}s)")
    unload(teacher_model, "Teacher-LLM")
    checkpoint(results, out_path)

    # ---- Stage 4: Student revises every item using teacher guidance ----
    student_tok, student_model = load_model(STUDENT_MODEL_ID, "Student")
    for i, r in enumerate(results):
        t0 = time.time()
        r["revised_answer"] = run_model(
            student_model, student_tok, STUDENT_REVISED_SYSTEM,
            build_student_revised_user(r["code"], r["question"], r["first_answer"], r["_teacher_raw"]),
            MAX_TOKENS_REVISED,
        )
        del r["_teacher_raw"]
        if (i + 1) % args.save_every == 0:
            checkpoint(results, out_path)
        print(f"[stage4 revise {i+1}/{len(results)}] {r['id']} ({time.time()-t0:.1f}s)")
    unload(student_model, "Student")
    checkpoint(results, out_path)

    # ---- Stage 5: Judge scores the post-intervention (ToM-LLM) answer ----
    judge_tok, judge_model = load_model(JUDGE_MODEL_ID, "Judge")
    for i, r in enumerate(results):
        r["scores_tom_llm"] = run_judge(
            judge_model, judge_tok, r["code"], r["question"],
            r["reference_answer"], r["revised_answer"],
        )
        if (i + 1) % args.save_every == 0:
            checkpoint(results, out_path)
        print(f"[stage5 judge-tom {i+1}/{len(results)}] {r['id']} -> {r['scores_tom_llm']}")
    unload(judge_model, "Judge")

    checkpoint(results, out_path)
    print(f"\nDone. Wrote {len(results)} records to {out_path}")
    print("Each record has scores_no_intervention (pre) and scores_tom_llm (post) for a direct paired comparison.")


if __name__ == "__main__":
    main()
