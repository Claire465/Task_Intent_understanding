def get_prompt(stage):
    stage = stage.replace(' ', '_')
    with open(f"prompts/{stage}.txt", 'r', encoding="utf-8") as f:
        return f.read()