import json
from collections import defaultdict

# CASE LOGIC (threshold = 2, dimensions: accuracy, completeness,
# clarity, relevance)
#
# Each record's four dimension scores can be in one of three states:
#   FAIL     : score < 2
#   BORDER   : score == 2
#   PASS     : score > 2
#
#
#   EXCLUDE
#   Case 1 — only PASS present       : all 4 dims > 2
#             SLM performed well across all dimensions.
#             Not interesting for error analysis.
#
#   INCLUDE
#   Case 2 — only BORDER present     : all 4 dims == 2
#             SLM failed uniformly — borderline on every dimension.
#
#   Case 3 — only FAIL present       : all 4 dims < 2
#             Total comprehension breakdown — SLM failed
#             severely on every dimension.
#
#   Case 4 — FAIL + BORDER only      : some dims < 2, some == 2,
#                                       none > 2
#             Partial breakdown with no redeeming quality —
#             SLM either failed or was borderline, nothing passable.
#
#   Case 5 — FAIL + PASS only        : some dims < 2, some > 2,
#                                       none == 2
#             Uneven failure — SLM got some aspects right but
#             was severely wrong on others.
#             Sub-bucketed by which dimension(s) scored < 2 (the
#             failure anchor), then grouped by identical score profile.
#
#   Case 6 — FAIL + BORDER + PASS    : all three states present
#             Mixed profile — SLM failed on some, was borderline
#             on others, and passable on others.
#             Sub-bucketed by which dimension(s) scored < 2 (the
#             failure anchor), then grouped by identical score profile.
#
#   Case 7 — BORDER + PASS only      : some dims == 2, some > 2,
#                                       none < 2
#             No outright failure but nothing fully correct either —
#             SLM was borderline on some dimensions while passing
#             others.
#             Sub-bucketed by which dimension(s) scored == 2 (the
#             borderline anchor), then grouped by identical score
#             profile.

INPUT_FILE  = "RQ1_judge_codeqa_difficult_samples_judged.json"
OUTPUT_FILE = "RQ1_open_coding_filtered.json"

DIMS = ["accuracy", "completeness", "clarity", "relevance"]


def classify(record):
    """
    Returns (case_label, sub_bucket) for a record.
    sub_bucket is None for cases without further splitting.
    For Cases 5, 6, 7 it is a tuple of the anchor dimension name(s).
    """
    scores = {d: record[d] for d in DIMS}

    has_fail   = any(v <  2 for v in scores.values())
    has_border = any(v == 2 for v in scores.values())
    has_pass   = any(v >  2 for v in scores.values())

    # --- Case 1: EXCLUDE ---
    if has_pass and not has_fail and not has_border:
        return "Case1_exclude", None

    # --- Case 2: only BORDER ---
    if has_border and not has_fail and not has_pass:
        return "Case2_all_border", None

    # --- Case 3: only FAIL ---
    if has_fail and not has_border and not has_pass:
        return "Case3_all_fail", None

    # --- Case 4: FAIL + BORDER only ---
    if has_fail and has_border and not has_pass:
        return "Case4_fail_border", None

    # --- Case 5: FAIL + PASS only ---
    # Sub-bucket by which dim(s) < 2 (failure anchor)
    if has_fail and has_pass and not has_border:
        fail_dims = tuple(d for d in DIMS if scores[d] < 2)
        return "Case5_fail_pass", fail_dims

    # --- Case 6: FAIL + BORDER + PASS ---
    # Sub-bucket by which dim(s) < 2 (failure anchor)
    if has_fail and has_border and has_pass:
        fail_dims = tuple(d for d in DIMS if scores[d] < 2)
        return "Case6_fail_border_pass", fail_dims

    # --- Case 7: BORDER + PASS only ---
    # Sub-bucket by which dim(s) == 2 (borderline anchor)
    if has_border and has_pass and not has_fail:
        border_dims = tuple(d for d in DIMS if scores[d] == 2)
        return "Case7_border_pass", border_dims

    return "Case_unknown", None


def score_profile(record):
    return tuple(record[d] for d in DIMS)


def main():
    with open(INPUT_FILE) as f:
        data = json.load(f)

    excluded = []
    included = defaultdict(lambda: defaultdict(list))  # case -> sub_bucket -> records

    for record in data:
        case_label, sub_bucket = classify(record)

        if case_label == "Case1_exclude":
            excluded.append(record)
        else:
            bucket_key = str(sub_bucket) if sub_bucket else "all"
            included[case_label][bucket_key].append(record)

    # --- Summary ---
    print("=" * 60)
    print("OPEN CODING FILTER SUMMARY")
    print("=" * 60)
    print(f"\nEXCLUDED (Case 1 — all dims > 2): {len(excluded)} records\n")

    total_included = 0
    output = {}

    for case_label in sorted(included.keys()):
        buckets = included[case_label]
        case_total = sum(len(v) for v in buckets.values())
        total_included += case_total
        print(f"{case_label}  ({case_total} records)")
        output[case_label] = {}

        for bucket_key in sorted(buckets.keys()):
            records = buckets[bucket_key]
            # Group by score profile within each sub-bucket
            profile_groups = defaultdict(list)
            for r in records:
                profile_groups[score_profile(r)].append(r)

            print(f"  sub-bucket: {bucket_key}")
            output[case_label][bucket_key] = {}

            for profile, group in sorted(profile_groups.items()):
                profile_str = (f"acc={profile[0]} comp={profile[1]} "
                               f"clar={profile[2]} rel={profile[3]}")
                ids = [r["id"] for r in group]
                cats = [r["category"] for r in group]
                print(f"    profile [{profile_str}]  n={len(group)}  "
                      f"ids={ids}  categories={cats}")
                output[case_label][bucket_key][profile_str] = group

        print()

    print(f"TOTAL INCLUDED FOR OPEN CODING: {total_included} records")
    print("=" * 60)

    # --- Save output ---
    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nFiltered records saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
