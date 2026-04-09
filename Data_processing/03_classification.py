import json
import os
import time
import argparse
import re
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import requests

Baseurl = "Baseurl"
Skey = "Skey"
url = Baseurl + "/v1/chat/completions"

CATEGORIZATION_PROMPT = '''You are a medical task classification expert. Given a medical evaluation question, your job is to classify it into one of the following 10 categories by returning its category code (A to J) and explanation.

Categories:
A. Completeness: Does the question test whether the response covers all relevant causes, symptoms, or histories?
B. Diagnostic Relevance: Does the question evaluate if the response focuses on core pathological mechanisms or clinical manifestations?
C. Treatment Appropriateness: Does it assess whether the treatment suggested is reasonable and safe?
D. Clinical Safety: Does the question involve identifying emergency signs or recommending seeking care?
E. Differentiation: Does it check whether the response distinguishes between different possible causes (e.g., viral vs. bacterial)?
F. Actionability: Is the question about whether the response gives actionable or personalized advice?
G. Tone/Empathy/Clarity: Does it evaluate whether the language is professional, calm, and non-alarming?
H. Caution/Uncertainty Management: Does it involve warning against self-diagnosis or clarifying uncertainty boundaries?
I. Pathophysiology/Explanation: Does the question ask if the response explains underlying mechanisms?
J. Personalization/Practicality: Is it about whether the response integrates patient context to offer practical advice?

Example Output Format:
"""
category: "F"
"""

Now classify this evaluation question:
"{evaluation_question}"
'''

def load_jsonl(file_path):
    _data = []
    with open(file_path, 'r') as f:
        for data in f:
            jline = json.loads(data)
            _data.append(jline)
    return _data


def get_payload(line):
    question = line['question']
    content = CATEGORIZATION_PROMPT.replace("{evaluation_question}", question)
    payload = json.dumps({
        "model": "Model_Name",
        "max_tokens": 1024,
        "messages": [
            {
                "role": "user",
                "content": content
            }
        ],
    })
    return payload


def save_jsonl(entry, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    filtered_entry = entry.copy()
    filtered_entry.pop('payload', None)  

    with open(save_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(filtered_entry, ensure_ascii=False) + "\n")


def get_answer(input_data: dict, retry=30):
    entry, save_path = input_data['data'], input_data['save_path']
    try:
        payload = get_payload(entry)
        headers = {
            'Accept': 'application/json',
            'Authorization': f'Bearer {Skey}',
            'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
            'Content-Type': 'application/json'
        }
        response = requests.request("POST", url, headers=headers, data=payload)
        if response.status_code == 200:
            data = response.json()
            generation = data['choices'][0]['message']['content'] if 'choices' in data else None
        else:
            print(response.text) 
            raise Exception("API request failed with status code: " + str(response.status_code))

        entry['category_response'] = generation
        entry['payload'] = payload
        match = re.search(r'category\s*:\s*"?([A-J])"?', generation, re.IGNORECASE)
        if match:
            entry['category'] = match.group(1)
        else:
            entry['category'] = "Unknown"
        save_jsonl(entry, save_path)
        return entry
    except Exception as e:
        time.sleep(1.2)
        retry -= 1
        if retry < 0:
            entry['category'] = "Error"
            entry['category_response'] = "None"
            entry['payload'] = payload
            save_jsonl(entry, save_path)
            return entry
        return get_answer(input_data, retry=retry)


def run_classification(save_path, datas, num_pool):
    processed_keys = set()
    if os.path.exists(save_path):
        with open(save_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    processed_keys.add((entry.get('main_id'), entry.get('point_id')))
                except:
                    continue
    processed = len(processed_keys)

    _input = []
    for data in datas:
        if not data:
            continue
        if (data.get('main_id'), data.get('point_id')) not in processed_keys:
            _input.append({"data": data, "eval_model": "claude", "save_path": save_path})

    with tqdm(total=len(datas), initial=processed, desc='Processing', ncols=100) as pbar:
        def update_pbar(_):
            pbar.update(1)

        with ThreadPoolExecutor(max_workers=num_pool) as executor:
            futures = []
            for input_data in _input:
                future = executor.submit(get_answer, input_data)
                future.add_done_callback(update_pbar)
                futures.append(future)

            for future in futures:
                try:
                    future.result()
                except Exception as e:
                    print(f"Error in task: {e}")


def get_data(data_path):
    with open(data_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    datas = []
    for i, d in enumerate(raw_data):
        for j, q in enumerate(d['scoring_questions']):
            datas.append({
                "main_id": i,
                "point_id": j,
                "question": q['question']
            })
    return datas

def inject_category_and_overwrite_constraints(input_json_path, categorized_jsonl_path, output_path):
    with open(input_json_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    id2category = {}
    with open(categorized_jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                item = json.loads(line)
                key = (item['main_id'], item['point_id'])
                id2category[key] = item.get('category', 'Unknown')
            except:
                continue

    for i, d in enumerate(raw_data):
        ordered_categories = []
        for j, q in enumerate(d.get("scoring_questions", [])):
            key = (i, j)
            cat = id2category.get(key, "Unknown")
            q["constraint_dimensions"] = [cat]
            ordered_categories.append(cat)
        d["constraint_dimensions"] = ordered_categories
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(raw_data, f, ensure_ascii=False, indent=2)

    print(f"✅{output_path}")

def main_run(args):
    input_file = os.path.join(args.dir, "02_gemini_data.json")
    output_file = os.path.join(args.output_path, "03_gemini_data.jsonl")
    final_output_file = os.path.join(args.output_path, "03_gemini_category_data.json")
    datas = get_data(input_file)
    run_classification(output_file, datas, args.num_pool)
    inject_category_and_overwrite_constraints(input_file, output_file, final_output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default="Processed_data")
    parser.add_argument("--num_pool", type=int, default=1)
    parser.add_argument("--output_path", type=str, default="Processed_data")
    args = parser.parse_args()
    main_run(args)
