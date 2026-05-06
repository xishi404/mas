MMLU_SOLVE_PROMPT = """
You are an expert test-taker solving a multiple choice question. Follow these steps carefully:

1. Read the question and all options thoroughly.
2. Identify the subject area and recall relevant knowledge.
3. Eliminate options that are clearly incorrect.
4. Analyze the remaining options carefully with step-by-step reasoning.
5. If the question involves calculation, perform the computation precisely.
6. Select the single best answer from the given options.

Format your answer as follows:
- Show your reasoning process clearly.
- At the end of your response, clearly state your final answer using the format: "The answer is [X]" where X is the letter (A, B, C, etc.) of your chosen option.
- Do not include any additional text after your final answer line.

Here is the question:

"""

REFINE_ANSWER_PROMPT = """
Given the multiple choice question, the analysis, and any supporting evidence, determine the correct answer.

Follow these guidelines:
1. Review the analysis carefully.
2. Cross-check with the available options.
3. Select the single best answer.
4. State your final answer using the format: "The answer is [X]" where X is the letter of your chosen option.
"""

SOLUTION_PROMPT = """
You are an expert solving a multiple choice question. Analyze the question step by step, consider all options, and provide the correct answer.

At the end, clearly state: "The answer is [X]" where X is the letter of your chosen option.
"""

IMPROVE_CODE_PROMPT = """
The previous solution was incorrect. Fix it and handle edge cases. The function should return only the letter of the correct answer (A, B, C, etc.).
"""

GENERATE_CODE_PROMPT = """
Solve the multiple choice question. Write Python code to verify or compute the answer if applicable. The solve() function should return the letter of the correct option.
"""
