import json
from collections import defaultdict

file_paths = [
    'evaluation_data\step1_referee_modelName1.jsonl',
    'evaluation_data\step1_referee_modelName2.jsonl',
    'evaluation_data\step1_referee_modelName3.jsonl',
    'evaluation_data\step1_referee_modelName4.jsonl',
    'evaluation_data\step1_referee_modelName5.jsonl'
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

with open('step1_referee_modelName_integration.json', 'w', encoding='utf-8') as f:
    json.dump(result, f, indent=2, ensure_ascii=False)

