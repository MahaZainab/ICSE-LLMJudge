import json
from openpyxl import Workbook
from openpyxl.styles import Font

INPUT_JSON = "llama3_2_3b_instruct_open_coding_filtered.json"
OUTPUT_XLSX = "open_coding_llama3_2_3b_instruct_open_coding_filtered.xlsx"

with open(INPUT_JSON) as f:
    data = json.load(f)

wb = Workbook()
sheet = wb.active
sheet.title = "Open Coding"

headers = ["id", "dataset", "category", "code", "question", "answer", "prediction",
           "accuracy", "completeness", "clarity", "relevance"]
sheet.append(headers)
for cell in sheet[1]:
    cell.font = Font(bold=True)

for case, combos in data.items():
    for combo, patterns in combos.items():
        for pattern, records in patterns.items():
            for r in records:
                sheet.append([
                    r.get("id"), r.get("dataset"), r.get("category"),
                    r.get("code"), r.get("question"), r.get("answer"), r.get("prediction"),
                    r.get("accuracy"), r.get("completeness"), r.get("clarity"), r.get("relevance"),
                ])

widths = [8, 10, 12, 60, 40, 20, 20, 8, 10, 8, 8]
for i, w in enumerate(widths, start=1):
    sheet.column_dimensions[sheet.cell(row=1, column=i).column_letter].width = w

sheet.freeze_panes = "A2"

wb.save(OUTPUT_XLSX)
print(f"Saved {OUTPUT_XLSX}")
