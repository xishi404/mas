SC_ENSEMBLE_PROMPT = """
Given the question described as follows: {problem}
Several solutions have been generated to address the given question. They are as follows:
{solutions}

Carefully evaluate these solutions and identify the answer that appears most frequently across them. This consistency in answers is crucial for determining the most reliable solution.

In the "thought" field, provide a detailed explanation of your thought process. In the "solution_letter" field, output only the single letter ID (A, B, C, etc.) corresponding to the most consistent solution. Do not include any additional text or explanation in the "solution_letter" field.
"""

PYTHON_CODE_VERIFIER_PROMPT = """
You are a professional Python programmer. Your task is to write complete, self-contained code to help solve a multiple choice question by verifying or computing the answer.

Question: {problem}
Other analysis: {analysis}
{feedback}

Your code should:
1. Implement the necessary calculations or logic to determine the correct answer.
2. Define a function named `solve` that performs the calculation and returns the result. The `solve` function should not require any input parameters.
3. The `solve` function should return the letter of the correct option (e.g., "A", "B", "C", etc.) or the computed value that helps identify the correct option.

Please ensure your code is efficient, well-commented, and follows Python best practices. The output should be limited to basic data types such as strings, integers, and floats.
"""

SELFREFINE_PROMPT = """
You are an assistant specialized in refining answers to multiple choice questions.

Problem:
{problem}

Previous Answer:
{solution}

Instruction:
Review the above answer carefully. Check for:
1. Logical errors in the reasoning.
2. Miscalculations or factual mistakes.
3. Whether the selected option truly matches the reasoning.
4. Whether other options might be more correct.

Provide your refined answer below, ending with: "The answer is [X]" where X is the letter of the correct option.
"""

GENERATE_COT_PROMPT = """
Multiple Choice Question Reasoning Instruction
{instruction}

Current Problem:
{input}

Demonstration Examples:

1. Problem: What is the primary function of mitochondria in a cell?
   A: Protein synthesis  B: Energy production  C: Cell division  D: DNA replication
   Analysis:
   Mitochondria are known as the "powerhouses" of the cell.
   They produce ATP through oxidative phosphorylation.
   Protein synthesis occurs in ribosomes (not mitochondria).
   Cell division is managed by the cell cycle machinery.
   DNA replication occurs in the nucleus.
   The answer is B.

2. Problem: Which of the following best describes the concept of "stare decisis"?
   A: The power of judicial review  B: The principle that courts should follow precedent
   C: The right to a jury trial  D: The separation of powers doctrine
   Analysis:
   "Stare decisis" is a Latin term meaning "to stand by things decided."
   It is a legal principle that obligates courts to follow historical cases when ruling on similar cases.
   Judicial review (A) is the power to review constitutionality.
   Jury trial (C) is a procedural right.
   Separation of powers (D) divides government branches.
   The answer is B.

Solution Protocol:
1. Read the question and all options carefully.
2. Identify the relevant domain knowledge.
3. Reason through each option systematically.
4. Eliminate clearly wrong options.
5. State the final answer as: "The answer is [X]".

"""
