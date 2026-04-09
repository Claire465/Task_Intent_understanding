import json
import os

instruction_file = os.path.join("Data_processing", "Processed_data", "03_gemini_category_data.json")
model_output_file = os.path.join("evaluation_precision", "step3_gemini_entropy.json")
output_file = os.path.join("evaluation_data", "step4_gemini.json")

with open(instruction_file, "r", encoding="utf-8") as f:
    instruction_data = json.load(f)

with open(model_output_file, "r", encoding="utf-8") as f:
    model_outputs = json.load(f)

from collections import defaultdict

model_outputs_grouped = defaultdict(list)
for item in model_outputs:
    model_outputs_grouped[item["main_id"]].append(item)

merged = []
for record in instruction_data:
    main_id = record["main_id"]
    scoring_questions = record["scoring_questions"]
    model_qs = model_outputs_grouped[main_id]

    merged_scoring_questions = []
    for i, q in enumerate(scoring_questions):
        merged_q = {
            "point_id": q.get("point_id", i),
            "question": q["question"],
            "constraint_dimensions": q.get("constraint_dimensions", []),
        }
        if i < len(model_qs):
            model_entry = model_qs[i]
            merged_q["model_output"] = {
                "output": model_entry.get("output"),
                "generations": model_entry.get("generations", []),
                "p_true": model_entry.get("p_true"),
                "semantic_ids": model_entry.get("semantic_ids", []),
                "semantic_entropy": model_entry.get("semantic_entropy")
            }
        else:
            merged_q["model_output"] = None  

        merged_scoring_questions.append(merged_q)

    merged.append({
        "main_id": main_id,
        "instruction": record["instruction"],
        "task_types": record.get("task_types"),
        "constraint_dimensions": record.get("constraint_dimensions", []),
        "scoring_questions": merged_scoring_questions
    })


with open(output_file, "w", encoding="utf-8") as f:
    json.dump(merged, f, indent=2, ensure_ascii=False)

print(f"✅ {output_file}")
