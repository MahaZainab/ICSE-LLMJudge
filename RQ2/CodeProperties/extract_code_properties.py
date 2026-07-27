
import json
import re
from collections import Counter

KEYWORD_GROUPS = {
    "loop": [" for ", " while "],
    "branch": [" if ", " elif ", " else "],
    "boolean_op": [" and ", " or ", " not "],
    "exception": [" try ", " except ", " finally ", " raise "],
}


def _count_kw(code, phrases):
    padded = f" {code} "
    return sum(padded.count(p) for p in phrases)


def _string_literal_count(code):
    # counts quote-delimited spans; works because quote chars are preserved
    singles = re.findall(r"'[^']*'", code)
    doubles = re.findall(r'"[^"]*"', code)
    return len(singles) + len(doubles)


def _recursion_flag(code):
    tokens = code.split()
    if not tokens or tokens[0] != "def" or len(tokens) < 2:
        return False
    fn_name = tokens[1]
    return fn_name in tokens[2:]


def extract_lexical(code):
    """Keyword/lexical properties. Works on CodeQA's stripped-punctuation code."""
    tokens = code.split()
    loop_c = _count_kw(code, KEYWORD_GROUPS["loop"])
    branch_c = _count_kw(code, KEYWORD_GROUPS["branch"])
    bool_c = _count_kw(code, KEYWORD_GROUPS["boolean_op"])
    exc_c = _count_kw(code, KEYWORD_GROUPS["exception"])

    return {
        "token_count": len(tokens),
        "loop_count": loop_c,
        "branch_count": branch_c,
        "boolean_op_count": bool_c,
        "exception_count": exc_c,
        "nested_def_count": max(tokens.count("def") - 1, 0),
        "lambda_count": tokens.count("lambda"),
        "return_count": tokens.count("return"),
        "string_literal_count": _string_literal_count(code),
        "bracket_literal_count": code.count("[") + code.count("{"),
        "cyclomatic_complexity_proxy": 1 + loop_c + branch_c + bool_c + exc_c,
        "recursion_flag": _recursion_flag(code),
    }


def extract_full(code):
    """Adds structural properties that require intact punctuation (CS1QA)."""
    base = extract_lexical(code)

    call_matches = re.findall(r"\b(\w+)\s*\(", code)
    call_count = sum(1 for name in call_matches if name != "def")

    assignment_count = len(re.findall(r"(?<![=!<>])=(?!=)", code))

    def_match = re.search(r"\bdef\s+\w+\s*\((.*?)\)\s*:", code)
    if def_match and def_match.group(1).strip():
        param_count = len(def_match.group(1).split(","))
    else:
        param_count = 0

    max_depth, depth = 0, 0
    for ch in code:
        if ch in "([{":
            depth += 1
            max_depth = max(max_depth, depth)
        elif ch in ")]}":
            depth = max(depth - 1, 0)

    indexing_count = len(re.findall(r"\w\[", code))

    base.update({
        "call_count": call_count,
        "param_count": param_count,
        "assignment_count": assignment_count,
        "max_nesting_depth": max_depth,
        "indexing_count": indexing_count,
    })
    return base


def process_file(path, code_key="code", full=False):
    data = json.load(open(path, encoding="utf-8"))
    extractor = extract_full if full else extract_lexical
    for ex in data:
        ex["properties"] = extractor(ex[code_key])
    return data


if __name__ == "__main__":
    codeqa = process_file("CodeQA_clean_final_v2.json", full=False)
    cs1qa = process_file("CS1QA_clean_final_v2.json", full=True)

    for name, data in [("CodeQA", codeqa), ("CS1QA", cs1qa)]:
        print(f"\n=== {name}: {len(data)} records ===")
        keys = data[0]["properties"].keys()
        for k in keys:
            vals = [ex["properties"][k] for ex in data]
            if isinstance(vals[0], bool):
                print(f"  {k}: {sum(vals)} True ({sum(vals)/len(vals):.1%})")
            else:
                print(f"  {k}: mean={sum(vals)/len(vals):.2f}, max={max(vals)}")

    with open("CodeQA_with_properties.json", "w", encoding="utf-8") as f:
        json.dump(codeqa, f, indent=2)
    with open("CS1QA_with_properties.json", "w", encoding="utf-8") as f:
        json.dump(cs1qa, f, indent=2)
