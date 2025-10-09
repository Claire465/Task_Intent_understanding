import json
import os
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import argparse

MAX_API_RETRY = 30
BASE_URL = "baseurl"
API_KEY = "sk"

def get_response(instruction, model_name):
    answer = None
    for _ in range(MAX_API_RETRY):
        try:
            payload = {
                "model": model_name, 
                "messages": [
                    {"role": "user", "content": instruction}
                ],
                "max_tokens": 1024,
                "top_p": 0.8,
                "temperature": 0.7
            }
            headers = {
                'Accept': 'application/json',
                'Authorization': f'Bearer {API_KEY}',
                'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
                'Content-Type': 'application/json'
            }
            url = f"{BASE_URL}/v1/chat/completions"
            response = requests.post(url, headers=headers, json=payload)
            response.encoding = 'utf-8'
            data = response.json()
            print(response.status_code)  
            print(f"API Response: {data}")
            if 'choices' in data:
                answer = data['choices'][0]['message']['content']
                return answer 
        except requests.exceptions.RequestException as e:
            print(f"fail: {e}")
    return answer 


def main(args):
    input_file = os.path.join(args.dir, "03_gemini_category_data.json")

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    model_name = "Model_name"  
    output_file = os.path.join(args.output_dir, f"04_gemini_generation.jsonl")

    with open(output_file, "w", encoding="utf-8") as f: 
        pass
    with ThreadPoolExecutor(max_workers=args.worker) as executor:
        tasks = {executor.submit(get_response, item["instruction"], model_name): item for item in data}

        for future in tqdm(tasks, total=len(tasks)):
            item = tasks[future]
            answer = future.result()
            if answer:
                result = {
                    "main_id": item["main_id"],
                    "model": model_name,
                    "instruction": item["instruction"],
                    "generated": answer
                }
                with open(output_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
    print(f"✅  {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, help="input_file", default="Processed_data")
    parser.add_argument("--output_dir", type=str, help="output_file", default="Processed_data")
    parser.add_argument("--worker", type=int, help="Number of concurrent threads", default=1)
    args = parser.parse_args()
    main(args)
