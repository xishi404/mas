"""
Short versions of MATH operator prompts (test phase).
Designed to reduce token usage while preserving task semantics.
"""

SC_ENSEMBLE_PROMPT = """
Problem: {problem}
Solutions:
{solutions}

Pick the most correct and consistent solution. In "thought", briefly state why. In "solution_letter", output only the letter (A, B, C, etc.).
"""

PYTHON_CODE_VERIFIER_PROMPT = """
Write Python code to solve this math problem. Define a `solve()` function that returns the answer.

Problem: {problem}
Analysis: {analysis}
{feedback}

Return only the `solve` function with necessary imports. Output must be a basic type (str, int, float).
"""

SELFREFINE_PROMPT = """
Refine this solution. Fix any errors and improve correctness.

Problem:
{problem}

Solution:
{solution}
"""

GENERATE_COT_PROMPT = """
{instruction}

Problem: {input}

Think step by step briefly, then present your final answer in \\boxed{{}} notation.
"""
