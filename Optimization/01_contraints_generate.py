import json
import os
import sys
import re
from tqdm import tqdm
import requests
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import get_prompt


MAX_API_RETRY = 30
Baseurl = "url"
Skey = "skey"
def get_response(query, prompt):
    for _ in range(MAX_API_RETRY):
        try:
            payload = {
                "model": "model_name",
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 1024,
                "top_p": 0.8,
                "temperature": 0.7
            }
            # 构建请求头
            headers = {
                'Accept': 'application/json',
                'Authorization': f'Bearer {Skey}',
                'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
                'Content-Type': 'application/json'
            }
            url = f"{Baseurl}/v1/chat/completions"
            response = requests.post(url, headers=headers, json=payload)
            response.encoding = 'utf-8'
            data = response.json()
            print(response.status_code)
            print(f"API Response: {data}")  # Debug output to inspect the full response
            content = data['choices'][0]['message']['content']
        except Exception as e:
            print(f"failed...{e}")
            continue
        try:
            match = re.search(r"```json\s*(\{.*?\})\s*```", content, re.DOTALL)
            if not match:
                match = re.search(r"(\{.*?\})", content, re.DOTALL)
            if match:
                content = match.group(1)
            else:
                return query, None

            answer = json.loads(content)
        except Exception as e:
            print(f"json failed to parse: {e}")
            print(f"content: {content}")
            return query, None
        return query, answer

def main(args):
    # load input queries
    input_file = os.path.join(args.inputdir, "selected_query.txt")
    output_file = os.path.join(args.outputdir, "01_gemini_constraints.json")

    with open(input_file, "r", encoding="utf-8") as f:
        questions = [line.strip() for line in f if line.strip()]

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({}, f)
    constraints = {}
    main_id_counter = 0
    prompt_template = get_prompt("01 constraints", base_dir=os.path.dirname(os.path.abspath(__file__)))

    for question in tqdm(questions, desc="Processing"):
        _, answer = get_response(question, prompt_template % question)
        if answer:
            answer['main_id'] = main_id_counter
            constraints[question] = answer
            main_id_counter += 1
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(constraints, f, indent=4)
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputdir", type=str, help="input file path for sentences", default="Data_processing/Processed_data")
    parser.add_argument("--outputdir", type=str, help="output file path for sentences",default="Optimization/data_precision")
    parser.add_argument("--save-iterval", type=int, help="save to file after generating K samples", default=2)
    parser.add_argument("--worker", type=int, help="number of concurrent workers", default=1)
    args = parser.parse_args()
    main(args)