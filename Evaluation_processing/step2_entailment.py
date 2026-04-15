import json
import os
import requests
from tqdm import tqdm
import argparse

MAX_API_RETRY = 30
BASE_URL = "baseurl"
API_KEY = "skey"

# Prompt 模板
PROMPT_TEMPLATE = """
We are evaluating answers to the question \
"Does the model's output {output} satisfy the evaluation criterion {question}?"
Here are two possible answers:
Possible Answer 1: {text1}
Possible Answer 2: {text2}

Does Possible Answer 1 semantically entail Possible Answer 2?
Please reason step by step:
1. What is the core claim of Possible Answer 1?
2. What is the core claim of Possible Answer 2?
3. If Possible Answer 1 is true, does Possible Answer 2 necessarily follow,
   possibly follow, or is it contradicted?
4. Final judgment (one word only): entailment / neutral / contradiction
"""

# 构造 Prompt
def build_prompt(text1, text2, question, output):
    return PROMPT_TEMPLATE.format(text1=text1, text2=text2, question=question, output=output)

# API 请求函数
def get_entailment_result(prompt: str) -> str:
    for _ in range(MAX_API_RETRY):
        try:
            payload = {
                "model": "claude-3-opus-20240229",
                "messages": [
                    {"role": "user", "content": prompt}
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
            output = requests.post(f"{BASE_URL}/v1/chat/completions", headers=headers, json=payload)
            data = output.json()
            content = data["choices"][0]["message"]["content"].strip().lower()
            if "entailment" in content:
                return "entailment"
            elif "contradiction" in content:
                return "contradiction"
            elif "neutral" in content:
                return "neutral"
        except Exception as e:
            print(f"API fail: {e}")
    return "neutral"  # 默认中性

def are_semantically_equivalent(a1: str, a2: str, question: str, output:str, strict: bool = False) -> bool:
    imp1 = get_entailment_result(build_prompt(a1, a2, question, output))
    imp2 = get_entailment_result(build_prompt(a2, a1, question, output))
    if strict:
        return imp1 == "entailment" and imp2 == "entailment"
    else:
        return "contradiction" not in [imp1, imp2] and not (imp1 == "neutral" and imp2 == "neutral")

def get_semantic_ids(generations, question, output, strict=False):
    n = len(generations)
    cluster_ids = [-1] * n
    cluster_id = 0
    for i in range(n):
        if cluster_ids[i] != -1:
            continue
        cluster_ids[i] = cluster_id
        for j in range(i + 1, n):
            if cluster_ids[j] == -1 and are_semantically_equivalent(generations[i], generations[j], question, output, strict):
                cluster_ids[j] = cluster_id
        cluster_id += 1
    return cluster_ids

def load_existing_keys(output_path):
    if not os.path.exists(output_path):
        return set()
    try:
        with open(output_path, "r", encoding="utf-8") as f:
            existing_data = json.load(f)
            return set(f"{item['main_id']}#{item['point_id']}" for item in existing_data)
    except Exception:
        return set()

def append_result_to_file(result, output_path):
    if not os.path.exists(output_path):
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump([result], f, ensure_ascii=False, indent=2)
    else:
        with open(output_path, "r+", encoding="utf-8") as f:
            data = json.load(f)
            data.append(result)
            f.seek(0)
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.truncate()


def main(args):
    input_path = os.path.join(args.dir, "step1_referee_gemini_integration.json")
    output_path = os.path.join(args.output_dir, "step2_cluster_gemini_result.json")

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    existing_keys = load_existing_keys(output_path)


    for item in tqdm(data):
        main_id = item.get("main_id", "")
        point_id = item.get("point_id", "")
        key = f"{main_id}#{point_id}"

        if key in existing_keys:
            continue
        generations = item["generations"]  # e.g., ["answer1", "answer2", ..., "answer5"]
        question = item["question"]
        output = item["output"]
        p_true = item["p_true"]
        cluster_ids = get_semantic_ids(generations, question, output, strict=False)

        result = {
            "main_id": item.get("main_id", ""),
            "point_id":item.get("point_id", ""),
            "question": question,
            "output": output,
            "generations": generations,
            "p_true": p_true,
            "semantic_ids": cluster_ids
        }
        append_result_to_file(result, output_path)

    print(f"✅ : {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default="Evaluation_processing/evaluation_data")
    parser.add_argument("--output_dir", type=str, default="Evaluation_processing/evaluation_data")
    args = parser.parse_args()
    main(args)
