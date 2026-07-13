"""
Remove Yes/No and What -type questions from the CodeQA dataset.

Uses the existing `_category` field in each record to filter out
entries where _category is "Yes/No" or "What".
"""

import json

INPUT_PATH = "CodeQA_clean_final_v2.json"
OUTPUT_PATH = "CodeQA_clean_final_v2_filtered.json"

EXCLUDED_CATEGORIES = {"Yes/No", "What"}


def filter_dataset(input_path: str, output_path: str) -> None:
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    filtered = [d for d in data if d.get("_category") not in EXCLUDED_CATEGORIES]

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(filtered, f, indent=2, ensure_ascii=False)

    print(f"Original: {len(data)} records")
    print(f"Filtered: {len(filtered)} records")
    print(f"Removed:  {len(data) - len(filtered)} records")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    filter_dataset(INPUT_PATH, OUTPUT_PATH)
