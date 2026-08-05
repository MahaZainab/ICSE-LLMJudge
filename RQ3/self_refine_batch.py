
import argparse
import gc
import json
import os
import time

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


STUDENT_MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
JUDGE_MODEL_ID    = "Qwen/Qwen2.5-Coder-7B-Instruct"

MAX_TOKENS_FIRST    = 128
MAX_TOKENS_FEEDBACK = 200
MAX_TOKENS_REFINE   = 128
MAX_TOKENS_JUDGE    = 128

DATASET_NAME = "CodeQA"
DATA_PATH = "CodeQA_dataset.json"
OUT_DIR = "results"

.
# ----------------------------------------------------------------------------
FEW_SHOT_EXAMPLES = [
    {"code": "def sort_list ( items ) : return sorted ( items , reverse = True )",
     "question": "How does the code sort the items?", "answer": "in descending order"},
    {"code": "def on_timeout ( self ) : self . retries += 1\nif self . retries > 3 : self . stop ( )",
     "question": "When does the code stop retrying?", "answer": "after more than 3 retries"},
    {"code": "def save_log ( entry ) : with open ( 'logs/app.log' , 'a' ) as f : f . write ( entry )",
     "question": "Where does the code write the log entry?", "answer": "to the file logs/app.log"},
    {"code": "def validate_input ( data ) : if not data : raise ValueError ( 'empty input' )",
     "question": "For what purpose does the code raise a ValueError?",
     "answer": "to signal that the input data is empty"},
    {"code": "def cache_result ( self , key , value ) : self . _cache [ key ] = value",
     "question": "Why does the code store the value in a dictionary?",
     "answer": "to cache the result for later lookup by key"},
]

NOINTV_SYSTEM = (
    "You are an expert software engineer with deep experience in source code comprehension, "
    "code review, and software documentation in Python.\n"
    "You will be given a Python code snippet and a natural language question about that code.\n\n"
    "Your task:\n"
    "- Read the code carefully before answering.\n"
    "- Answer the question directly and concisely based solely on what the code does.\n"
    "- Match the style and length of the examples provided — a short phrase or single sentence is expected.\n\n"
    "IMPORTANT: Study the examples carefully before answering.\n"
    "IMPORTANT: The examples define the expected answer style, format, and length — match them exactly.\n"
    "IMPORTANT: If the question asks WHAT the code returns, answer with the thing that is returned, not a description of how it works.\n\n"
    "CRITICAL: DO NOT REPEAT OR PARAPHRASE THE QUESTION IN YOUR ANSWER.\n"
    "CRITICAL: DO NOT ADD EXPLANATIONS OR INFORMATION NOT PRESENT IN THE CODE.\n"
    "CRITICAL: DO NOT PRODUCE A PARAGRAPH WHEN A PHRASE IS SUFFICIENT.\n"
    "CRITICAL: DO NOT HALLUCINATE — base your answer ONLY on the provided code snippet."
)


FEEDBACK_FEW_SHOT = {
    "code": "def get_average ( scores ) : return sum ( scores ) / len ( scores ) if scores else 0",
    "question": "Why does the code check if scores is empty?",
    "draft": "It checks the length of the scores list.",
    "feedback": (
        "Accuracy: The draft describes WHAT the check does (measures length) but not WHY it "
        "exists. It misses the actual reason.\n"
        "Relevance: The question asks for a cause/purpose, and the draft does not give one — "
        "low relevance to what was asked.\n"
        "Completeness: Incomplete — the consequence of skipping the check (division by zero via "
        "len(scores) as a divisor) is never mentioned.\n"
        "Actionable fix: Look at the return statement — scores is used as a divisor. State the "
        "purpose as avoiding a division-by-zero error when the list is empty, not the check's mechanics."
    ),
}

FEEDBACK_SYSTEM = (
    "You are an expert software engineer reviewing your own draft answer to a source code "
    "comprehension question.\n"
    "You will be given a code snippet, a question, and your draft answer.\n"
    "Give feedback that is ACTIONABLE (suggests a concrete fix) and SPECIFIC "
    "(points at the exact part of the draft or code that is wrong or missing) — "
    "generic feedback like 'be more accurate' is not useful and must be avoided.\n\n"
    "Check the draft against the code:\n"
    "- Accuracy: does it correctly reflect what the code does?\n"
    "- Relevance: does it answer exactly what is asked?\n"
    "- Completeness: does it cover what the question asks?\n"
    "- Clarity: how clearly does it communicate its point?\n\n"
    "CRITICAL: DO NOT REWRITE THE ANSWER — feedback only.\n"
    "CRITICAL: BASE FEEDBACK ONLY ON THE PROVIDED CODE.\n"
    "CRITICAL: END WITH A LINE STARTING 'Actionable fix:' NAMING THE CONCRETE CHANGE NEEDED.\n\n"
    "--- Example ---\n"
    f"Code:\n{FEEDBACK_FEW_SHOT['code']}\n\n"
    f"Question:\n{FEEDBACK_FEW_SHOT['question']}\n\n"
    f"Draft Answer:\n{FEEDBACK_FEW_SHOT['draft']}\n\n"
    f"Feedback:\n{FEEDBACK_FEW_SHOT['feedback']}\n"
    "--- End of Example ---"
)


REFINE_FEW_SHOT = {
    "code": FEEDBACK_FEW_SHOT["code"],
    "question": FEEDBACK_FEW_SHOT["question"],
    "draft": FEEDBACK_FEW_SHOT["draft"],
    "feedback": FEEDBACK_FEW_SHOT["feedback"],
    "refined": "to avoid a division by zero when the list is empty",
}

REFINE_SYSTEM = (
    "You are an expert software engineer answering a source code comprehension question.\n"
    "You have a previous draft answer and feedback on it. Write an improved answer.\n\n"
    "Your task:\n"
    "- Read the code, your draft, and the feedback carefully before revising.\n"
    "- Answer the question directly and concisely based solely on what the code does.\n"
    "- Match the style and length of the example provided — a short phrase or single sentence is expected.\n\n"
    "IMPORTANT: Keep the answer concise — a short phrase or single sentence, same style as a normal CodeQA answer.\n"
    "IMPORTANT: If the question asks WHAT the code returns, answer with the thing that is returned, not a description of how it works.\n\n"
    "CRITICAL: DO NOT REPEAT OR PARAPHRASE THE QUESTION IN YOUR ANSWER.\n"
    "CRITICAL: DO NOT ADD EXPLANATIONS OR INFORMATION NOT PRESENT IN THE CODE.\n"
    "CRITICAL: DO NOT PRODUCE A PARAGRAPH WHEN A PHRASE IS SUFFICIENT.\n"
    "CRITICAL: DO NOT DESCRIBE THE FEEDBACK — INCORPORATE IT SILENTLY INTO YOUR ANSWER.\n"
    "CRITICAL: OUTPUT ONLY THE IMPROVED ANSWER — no preamble, no explanation, no markdown.\n\n"
    "--- Example ---\n"
    f"Code:\n{REFINE_FEW_SHOT['code']}\n\n"
    f"Question: {REFINE_FEW_SHOT['question']}\n\n"
    f"Your draft answer: {REFINE_FEW_SHOT['draft']}\n\n"
    f"Feedback:\n{REFINE_FEW_SHOT['feedback']}\n\n"
    f"Improved Answer: {REFINE_FEW_SHOT['refined']}\n"
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


def build_nointv_user(code, question):
    lines = []
    for i, ex in enumerate(FEW_SHOT_EXAMPLES, 1):
        lines.append(f"--- Example {i} ---")
        lines.append(f"Code:\n{ex['code']}")
        lines.append(f"Question: {ex['question']}")
        lines.append(f"Answer: {ex['answer']}")
    lines.append("--- End of Examples ---\n")
    lines.append("Now answer the following question in the same style as the examples above.\n")
    lines.append(f"Code:\n{code}")
    lines.append(f"Question: {question}")
    lines.append("Answer:")
    return "\n".join(lines)


def build_feedback_user(code, question, draft):
    return f"Code:\n{code}\n\nQuestion:\n{question}\n\nDraft Answer:\n{draft}\n\nFeedback:"


def build_refine_user(code, question, draft, feedback):
    return (
        f"Code:\n{code}\n\nQuestion:\n{question}\n\n"
        f"Your draft answer:\n{draft}\n\nFeedback:\n{feedback}\n\nImproved Answer:"
    )



def load_items(limit=None):
    """Load CodeQA items and assign IDs as q<n> (1-indexed), matching the
    id.get("id", f"q{i+1}") fallback used across your RQ1/RQ3 scripts.
    """
    with open(DATA_PATH, "r") as f:
        raw = json.load(f)
    items = []
    for idx, rec in enumerate(raw):
        items.append({
            "id": rec.get("id", f"q{idx + 1}"),
            "dataset": DATASET_NAME,
            "category": rec.get("_category") or rec.get("questionType") or "",
            "code": rec["code"],
            "question": rec["question"],
            "reference_answer": rec["answer"],
        })
    if limit:
        items = items[:limit]
    return items



def checkpoint(results, out_path):
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                         help="cap number of items (for a quick test run)")
    parser.add_argument("--save-every", type=int, default=10,
                         help="checkpoint results to disk every N items")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "self_refine_codeqa.json")

    all_items = load_items(limit=args.limit)
    print(f"Loaded {len(all_items)} CodeQA items")

    results = [{**item, "first_answer": None, "self_feedback": None,
                "refined_answer": None, "scores_no_intervention": None,
                "scores_self_refine": None} for item in all_items]

    student_tok, student_model = load_model(STUDENT_MODEL_ID, "Student")
    for i, r in enumerate(results):
        t0 = time.time()
        r["first_answer"] = run_model(
            student_model, student_tok, NOINTV_SYSTEM,
            build_nointv_user(r["code"], r["question"]), MAX_TOKENS_FIRST,
        )
        r["self_feedback"] = run_model(
            student_model, student_tok, FEEDBACK_SYSTEM,
            build_feedback_user(r["code"], r["question"], r["first_answer"]), MAX_TOKENS_FEEDBACK,
        )
        r["refined_answer"] = run_model(
            student_model, student_tok, REFINE_SYSTEM,
            build_refine_user(r["code"], r["question"], r["first_answer"], r["self_feedback"]),
            MAX_TOKENS_REFINE,
        )
        if (i + 1) % args.save_every == 0:
            checkpoint(results, out_path)
        print(f"[stage1 draft+feedback+refine {i+1}/{len(results)}] {r['id']} ({time.time()-t0:.1f}s)")
    unload(student_model, "Student")
    checkpoint(results, out_path)

    judge_tok, judge_model = load_model(JUDGE_MODEL_ID, "Judge")
    for i, r in enumerate(results):
        r["scores_no_intervention"] = run_judge(
            judge_model, judge_tok, r["code"], r["question"],
            r["reference_answer"], r["first_answer"],
        )
        r["scores_self_refine"] = run_judge(
            judge_model, judge_tok, r["code"], r["question"],
            r["reference_answer"], r["refined_answer"],
        )
        if (i + 1) % args.save_every == 0:
            checkpoint(results, out_path)
        print(f"[stage2 judge {i+1}/{len(results)}] {r['id']} "
              f"before={r['scores_no_intervention']} after={r['scores_self_refine']}")
    unload(judge_model, "Judge")

    checkpoint(results, out_path)
    print(f"\nDone. Wrote {len(results)} records to {out_path}")
    print("Each record has scores_no_intervention (pre) and scores_self_refine (post) for a direct paired comparison.")


if __name__ == "__main__":
    main()
