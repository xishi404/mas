"""
Short versions of high-level MMLU Pro prompts (train phase).
"""

MMLU_SOLVE_PROMPT = """
Answer the multiple choice question. Think step by step, then end with: "The answer is [X]" where X is the option letter.

"""

REFINE_ANSWER_PROMPT = """
Given the question and analysis, select the correct option letter. End with: "The answer is [X]".
"""

SOLUTION_PROMPT = """
Answer the multiple choice question. End with: "The answer is [X]" where X is the option letter.
"""

IMPROVE_CODE_PROMPT = """
The previous solution was incorrect. Fix it. Return only the correct option letter.
"""

GENERATE_CODE_PROMPT = """
Solve the question with Python. The solve() function should return the correct option letter.
"""
