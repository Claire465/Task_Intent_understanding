EVALUATION_PROMPT = """
As an objective judge, analyze the content of the model responses and answer "yes" or "no" according to the requirements of the questions.

Judging rules:

Evaluate only the problem itself: The problem is a disassembly of the input instruction, and you only need to determine whether the model response meets the requirements of the problem, not whether the entire instruction is fully satisfied.

Criteria for "Yes" : The model response must fully meet the requirements of the problem, with no omissions, ambiguities, and errors in the details. If there is any deviation, missing or partially correct, the answer should be "no".

Criteria for "No" : Answer "no" if the model response does not meet the requirements of the question, lacks relevant information, or contains errors.

Give an example

Example 1:

Enter the command: "Generate a poem by Li Bai and end with 'generate the end'."

The model replied: "The moonlight in front of the bed, I suspect it is frost on the ground." Look up at the moon, bow my head and think of home."

Question: "Is model reply born into Li Bai's poetry?"

Answer: Yes ,Li Bai's poetry is correct, even if it does not meet the requirement of "the end of generation", it does not affect the judgment of this question.

Output format, please strictly follow the following format:

Response: < Your response >
Answer: Yes/no

Evaluation information

Input instruction

{input}

Model recovery

{output}

problem

{question}
"""

