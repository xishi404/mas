"""
Short versions of MMLU Pro operator prompts (train phase).
Designed to reduce token usage while preserving task semantics.
"""

SC_ENSEMBLE_PROMPT = """
Problem: {problem}
Solutions:
{solutions}

Pick the most correct and consistent solution. In "thought", briefly state why. In "solution_letter", output only the letter (A, B, C, etc.).
"""

PYTHON_CODE_VERIFIER_PROMPT = """
Write Python code to help solve this multiple choice question. Define a `solve()` function that returns the correct option letter or a value to identify it.

Question: {problem}
Analysis: {analysis}
{feedback}

Return only the `solve` function with necessary imports. Output must be a basic type (str, int, float).
"""

SELFREFINE_PROMPT = """
Refine this answer. Fix any errors and improve correctness.

Problem:
{problem}

Previous Answer:
{solution}

End with: "The answer is [X]" where X is the correct option letter.
"""

GENERATE_COT_PROMPT = """
{instruction}

Problem: {input}

Think step by step, eliminate wrong options, then end with: "The answer is [X]".
"""
