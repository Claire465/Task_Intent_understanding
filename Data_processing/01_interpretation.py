import json
import os
import requests
import re
from ..utils import get_prompt
import argparse
from tqdm import tqdm

MAX_API_RETRY = 10
Baseurl = "Baseurl"
Skey = "Skey"

def get_response(query, prompt_template):
    answer = None
    prompt = prompt_template.replace("<QUERY>", query)
    for _ in range(MAX_API_RETRY):
        try:
            payload = {
                "model": "Model_Name",
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 1024,
                "top_p": 0.8,
                "temperature": 0.7
            }
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
            print(f"API Response: {data}")

            if 'choices' in data:
                answer = data['choices'][0]['message']['content']

                match = re.search(r'(\{.*\})', answer, re.DOTALL)
                if match:
                    try:
                        response_data = json.loads(match.group(1))
                        task_type = response_data.get("task_types", "Consultation")
                        scoring_questions = response_data.get("scoring_questions", [])
                        interpretation = response_data.get("interpretation", {})

                        return query, task_type, interpretation, scoring_questions
                    except (ValueError, SyntaxError) as e:
                        print(f"Error parsing response as JSON: {e}")
                        return query, None, {}, []
                else:
                    print("Error: JSON part not found in the response.")
                    return query, None, {}, []
            else:
                    print(f"Error: No choices in the response data.")
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
    return query, None, {}, []

def save_to_json(output_file, new_entry):
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            try:
                existing_data = json.load(f)
                if not isinstance(existing_data, list):
                    existing_data = []
            except json.JSONDecodeError:
                existing_data = []
    else:
        existing_data = []

    existing_data.append(new_entry)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(existing_data, f, ensure_ascii=False, indent=4)
def main(args):
    input_file = os.path.join(args.dir, "selected_query.txt")
    output_file = os.path.join(args.dir, "01_interpretation.json")

    prompt_template = get_prompt("01 interpretation").strip().replace("\n", " ").replace("\t", " ")
    print(f"Prompt template: {prompt_template}")
    with open(input_file, "r", encoding="utf-8") as f:
        data = [line.strip() for line in f if line.strip()]

    results = []
    with tqdm(total=len(data), desc="Processing queries", unit="query") as pbar:
        for original_question in data:
            original_question, task_type, interpretation, scoring_questions = get_response(original_question,prompt_template)
            result = {
                "original_question": original_question,
                "task_type": task_type,
                "interpretation": interpretation,
                "scoring_questions": scoring_questions
            }
            results.append(result)
            save_to_json(output_file, result)
            pbar.update(1)

    print(f"{output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, help="input file path for sentences",default="Processed_data")
    parser.add_argument("--save-iterval", type=int, help="save to file after generating K samples", default=2)
    parser.add_argument("--worker", type=int, help="number of concurrent workers", default=4)
    args = parser.parse_args()
    main(args)
