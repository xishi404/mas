import asyncio
import re
import threading
import time
import torch
from typing import Any, Callable, Dict, List, Optional, Tuple, Literal

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from lamas.ext.lamas.benchmark.benchmark import BaseBenchmark
from lamas.logs import logger
from lamas.utils.sanitize import sanitize


class MMLUProBenchmark(BaseBenchmark):
    def __init__(self, name, file_path, log_path, batch_size, controller,
                 operator_embeddings, optimizer, **kwargs):
        super().__init__(name, file_path, log_path, batch_size, controller,
                         operator_embeddings, optimizer, **kwargs)

    @staticmethod
    def extract_answer(text: str) -> Optional[str]:
        """
        Extract the answer letter from model output using multiple pattern matching strategies.
        Inspired by ChatDev's check_mmlu multi-pattern approach.
        """
        if not text:
            return None

        text = text.strip()

        # Pattern 1: "The answer is [X]" or "the answer is [X]" (most explicit)
        match = re.search(r'[Tt]he answer is\s*\(?([A-J])\)?', text)
        if match:
            return match.group(1).upper()

        # Pattern 2: "answer is [X]" without "the"
        match = re.search(r'answer is\s*\(?([A-J])\)?', text, re.IGNORECASE)
        if match:
            return match.group(1).upper()

        # Pattern 3: "FINAL ANSWER: [X]" (ChatDev style)
        match = re.search(r'FINAL ANSWER:\s*\(?([A-J])\)?', text, re.IGNORECASE)
        if match:
            return match.group(1).upper()

        # Pattern 4: "Answer: [X]" at end of text
        match = re.search(r'[Aa]nswer:\s*\(?([A-J])\)?', text)
        if match:
            return match.group(1).upper()

        # Pattern 5: Standalone letter in parentheses at end, e.g., "(C)"
        match = re.search(r'\(([A-J])\)\s*\.?\s*$', text)
        if match:
            return match.group(1).upper()

        # Pattern 6: Last single uppercase letter on its own line
        lines = text.strip().split('\n')
        for line in reversed(lines):
            line = line.strip().rstrip('.')
            if len(line) == 1 and line.upper() in 'ABCDEFGHIJ':
                return line.upper()

        # Pattern 7: "is [X]" generic fallback
        match = re.search(r'\bis\s+([A-J])\b', text)
        if match:
            return match.group(1).upper()

        # Pattern 8: Last letter found in the text matching [A-J] preceded by common indicators
        match = re.search(r'(?:choose|select|pick|correct answer is|best answer is)\s*\(?([A-J])\)?', text, re.IGNORECASE)
        if match:
            return match.group(1).upper()

        return None

    @staticmethod
    def format_question(problem: dict) -> str:
        """
        Format the MMLU Pro question with options for model input.
        Follows ChatDev's format_question pattern.
        """
        import string
        question = problem.get("question", "")
        options = problem.get("options", [])
        category = problem.get("category", "")

        # Build formatted question with category context
        formatted = f"The following is a multiple choice question about {category}.\n\n"
        formatted += f"{question}\n\n"

        # Format options as "A: option_text\nB: option_text\n..."
        for i, option in enumerate(options):
            letter = string.ascii_uppercase[i]
            formatted += f"{letter}: {option}\n"

        return formatted.strip()

    def calculate_score(self, expected_output: str, prediction: str) -> Tuple[float, str]:
        """Compare predicted answer letter with expected answer letter."""
        if prediction is None:
            return 0.0, prediction
        return (1.0 if prediction.upper() == expected_output.upper() else 0.0), prediction

    async def _generate_output(self, graph, input_text):
        max_attempts = 1 if self.local_model else 20
        @retry(stop=stop_after_attempt(max_attempts), wait=wait_fixed(1), retry=retry_if_exception_type(Exception), reraise=True)
        async def _inner():
            return await asyncio.wait_for(graph(input_text, log_path=self.log_path), timeout=1500)
        return await _inner()

    async def evaluate_problem(self, problem: dict, graph: Callable):
        import time
        # Format the question with options
        input_text = self.format_question(problem)
        expected_output = problem["answer"]  # Single letter like "A", "B", etc.

        start_time = time.time()
        try:
            result = await self._generate_output(graph, input_text)
            # Unpack result: (output, cost, logprob, total_virtual_tokens, layer_operator_info)
            if len(result) == 5:
                output, cost, logprob, cp_token, layer_operator_info = result
            elif len(result) == 4:
                output, cost, logprob, layer_operator_info = result
                cp_token = 0.0
            else:
                output, cost, logprob = result[0], result[1], result[2]
                cp_token = 0.0
                layer_operator_info = None
            latency = time.time() - start_time

            if not output:
                raise ValueError("output is empty")

            # Extract answer letter from model output
            predicted_letter = self.extract_answer(output)
            score, extracted_output = self.calculate_score(expected_output, predicted_letter)

            if score == 0:
                self.log_mismatch(input_text, expected_output, output, extracted_output)

            return input_text, output, expected_output, score, cost, logprob, cp_token, latency, layer_operator_info

        except Exception as e:
            import traceback
            logger.info(f"Maximum retries reached. Skipping this sample. Error: {type(e).__name__}: {e}\n{traceback.format_exc()}")
            latency = time.time() - start_time
            return input_text, str(e), expected_output, 0.0, 0.0, torch.tensor(0.0, dtype=torch.float32, device=self.device), 0.0, latency, None

    def get_result_columns(self) -> List[str]:
        return ["question", "prediction", "expected_output", "score", "cost", "logprob", "cp_token", "latency"]
