import json

def load_data(input_file):
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def save_data(data,output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

def process_data(data):
    original_questions = []

    for idx, item in enumerate(data):
        constraint_dims = [dim for sq in item["scoring_questions"] for dim in sq["constraint_dimensions"]]

        original_questions.append({

                "main_id": idx,
                "instruction": item["original_question"],
                "task_types": item["task_type"],
                "constraint_dimensions": constraint_dims,
                "scoring_questions": [
                    {
                        "point_id": i,
                        "question": sq["question"],
                        "constraint_dimensions": sq["constraint_dimensions"],
                    }
                    for i, sq in enumerate(item["scoring_questions"])
                ],
                "sub_instructions": {
                    f"sub_instruction_{idx}": {
                        "instruction": item["original_question"],
                        "scoring_questions": [sq["question"] for sq in item["scoring_questions"]]
                    }
                }

        })

    return original_questions

def main():
    input_file_path = r"Processed_data\01_gemini_interpretation.json"
    original_data_path = r"Processed_data\02_gemini_data.json"
    data = load_data(input_file_path)
    original_questions = process_data(data)
    save_data(original_questions, original_data_path)


if __name__ == "__main__":
    main()
