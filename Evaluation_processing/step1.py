import json
import os
import time
import argparse
import re
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import traceback
import requests
from prompts.RAL_evaluator import EVALUATION_PROMPT

Baseurl = "baseurl"
Skey = "skey"
url = Baseurl + "/v1/chat/completions"
SYS_MSG = EVALUATION_PROMPT


def load_jsonl(file_path):
    _data = []
    with open(file_path, 'r') as f:
        for data in f:
            jline = json.loads(data)
            _data.append(jline)
    return _data


def get_payload(line):
    instruction = line['instruction'][:6000]
    question = line['question']
    output = line['output'][:4000] if line['output'] is not None else 'None'
    content = SYS_MSG.format(input=instruction, output=output, question=question)
    payload = json.dumps({
        "model": "model_name",
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

        if generation is None or generation == "":
            return get_answer(input_data, retry=retry - 1)

        re_result = re.findall(r'Answer:\s*(Yes|No)', generation, re.IGNORECASE)
        if re_result:
            entry['point_judge'] = re_result[0].strip().lower() == "yes"
        else:
            if "yes" in generation.lower() and "no" not in generation.lower():
                entry['point_judge'] = True
            else:
                entry['point_judge'] = False
        entry['point_explanation'] = generation
        entry['payload'] = payload
        save_jsonl(entry, save_path)
        return entry
    except Exception as e:
        time.sleep(1.2)
        retry -= 1
        if retry < 0:
            entry['point_judge'] = False
            entry['point_explanation'] = "None"
            entry['payload'] = payload
            save_jsonl(entry, save_path)
            return entry
        print(f"retry:{retry}")
        print(e)
        print(traceback.format_exc())
        return get_answer(input_data, retry=retry)


def run_evaluation(save_path, datas, num_pool):
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


def get_data(data_path, llm_output_path):
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    outputs = []
    with open(llm_output_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            try:
                outputs.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[ERROR] Line {idx + 1} JSON decode error: {e}")
                print(f"Line content: {line}")

    datas = []
    for i, (d, o) in enumerate(zip(data, outputs)):
        for j, q in enumerate(d['scoring_questions']):
            datas.append({
                "main_id": i,
                "point_id": j,
                "instruction": d['instruction'],
                "question": q['question'],
                "output": o['generated'],
            })
    return datas


def main_run(args):
    input_file = os.path.join(args.dir, "03_gemini_category_data.json")
    llm_output_file = os.path.join(args.llm_output_path, "04_gemini_generation.jsonl")
    output_file = os.path.join(args.output_path, f"step1_referee_gemini{args.run_id}.jsonl")
    datas = get_data(input_file, llm_output_file)
    run_evaluation(output_file, datas, args.num_pool)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default=os.path.join("Data_processing", "Processed_data"))
    parser.add_argument("--llm_output_path", type=str, default=os.path.join("Data_processing", "Processed_data"))
    parser.add_argument("--num_pool", type=int, default=1)
    parser.add_argument("--output_path", type=str, default="evaluation_data")
    parser.add_argument("--run_id", type=int, default=1, help="Evaluator run index (1-5), used to name output file")
    args = parser.parse_args()
    main_run(args)
