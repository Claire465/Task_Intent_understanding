import os


def get_prompt(stage, base_dir=None):
    stage = stage.replace(' ', '_')
    if base_dir is None:
        base_dir = os.getcwd()
    prompt_path = os.path.join(base_dir, "prompts", f"{stage}.txt")
    with open(prompt_path, 'r', encoding="utf-8") as f:
        return f.read()
