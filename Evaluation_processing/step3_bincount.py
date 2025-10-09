import json
import numpy as np

def cluster_assignment_entropy(semantic_ids):
    """
    Statistically analyze the frequency distribution of semantic clusters and calculate their Shannon entropy.
    """
    n = len(semantic_ids)
    counts = np.bincount(semantic_ids)
    probabilities = counts / n
    assert np.isclose(probabilities.sum(), 1.0)
    entropy = -np.sum(probabilities * np.log(probabilities + 1e-10))
    return max(0.0, entropy)

with open("evaluation_data\step2_cluster_gemini_result.json", "r", encoding="utf-8") as f:
    data = json.load(f)

for item in data:
    semantic_ids = item.get("semantic_ids", [])
    if semantic_ids:
        item["semantic_entropy"] = cluster_assignment_entropy(semantic_ids)
    else:
        item["semantic_entropy"] = None  

with open("evaluation_precision\step3_gemini_entropy.json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

