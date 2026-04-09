import json
import os
import argparse
from collections import defaultdict


def main(args):
    file_paths = [
        os.path.join(args.eval_dir, f"step1_referee_gemini{i}.jsonl")
        for i in range(1, args.num_runs + 1)
    ]

    grouped_data = defaultdict(list)
    for path in file_paths:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                key = (item['main_id'], item['point_id'])
                grouped_data[key].append(item)

    result = []
    for (main_id, point_id), items in grouped_data.items():
        representative = items[0]
        question = representative.get('question', '')
        output = representative.get('output', '')
        generations = [entry['point_explanation'].strip() for entry in items if 'point_explanation' in entry]
        point_judges = [entry.get('point_judge', False) for entry in items]
        true_count = sum(1 for x in point_judges if x is True)
        total_count = len(point_judges)
        p_true = true_count / total_count if total_count > 0 else 0.0
        result.append({
            "main_id": main_id,
            "point_id": point_id,
            "question": question,
            "output": output,
            "generations": generations,
            "p_true": round(p_true, 3)
        })

    output_path = os.path.join(args.eval_dir, "step1_referee_gemini_integration.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_dir", type=str, default="evaluation_data",
                        help="Directory containing step1 jsonl files")
    parser.add_argument("--num_runs", type=int, default=5,
                        help="Number of evaluator runs to aggregate")
    args = parser.parse_args()
    main(args)
