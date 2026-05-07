import concurrent
import sys
import traceback
import json
import os
import threading
from datetime import datetime
from typing import List

from tenacity import retry, stop_after_attempt, wait_fixed

from lamas.ext.lamas.scripts.optimized.GSM8K.train.template.operator_an import *
from lamas.ext.lamas.scripts.optimized.GSM8K.train.template.op_prompt import *
from lamas.actions.action_node import ActionNode
from lamas.llm import LLM
from lamas.logs import logger
import asyncio

# Global error logger for ScEnsemble invalid answers
_error_log_lock = threading.Lock()
_error_log_path = None

def _log_sc_error(problem: str, solutions: list, prompt: str, raw_response: dict, answer: str, answer_mapping: dict):
    """Log ScEnsemble invalid answer to a dedicated JSONL file."""
    global _error_log_path
    if _error_log_path is None:
        log_dir = os.path.join(os.getcwd(), "logs")
        os.makedirs(log_dir, exist_ok=True)
        _error_log_path = os.path.join(log_dir, "sc_ensemble_errors.jsonl")
    record = {
        "ts": datetime.now().isoformat(),
        "operator": "ScEnsemble",
        "dataset": "GSM8K",
        "invalid_answer": answer,
        "valid_keys": list(answer_mapping.keys()),
        "num_solutions": len(solutions),
        "raw_response": raw_response,
        "problem": problem[:500],
        "solutions_preview": [s[:200] if isinstance(s, str) else str(s)[:200] for s in solutions[:5]],
    }
    with _error_log_lock:
        with open(_error_log_path, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")



class Operator:
    def __init__(self, llm: LLM, name: str):
        self.name = name
        self.llm = llm

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    async def _fill_node(self, op_class, prompt, mode=None, return_usage=False, **extra_kwargs):
        fill_kwargs = {"context": prompt, "llm": self.llm, "return_usage": return_usage}
        if mode:
            fill_kwargs["mode"] = mode
        fill_kwargs.update(extra_kwargs)
        node = await ActionNode.from_pydantic(op_class).fill(**fill_kwargs)
        result = node.instruct_content.model_dump()
        # Get usage tokens from node attribute (stored separately from Pydantic model)
        usage_tokens = getattr(node, '_usage_tokens', 0) if return_usage else 0
        usage_prompt_tokens = getattr(node, '_usage_prompt_tokens', 0) if return_usage else 0
        if return_usage:
            result['_usage_prompt_tokens'] = usage_prompt_tokens
        return (result, usage_tokens) if return_usage else result


class Generate(Operator):
    def __init__(self, llm: LLM, name: str = "Generate"):
        super().__init__(llm, name)

    async def __call__(self, input, instruction, return_usage=False):
        prompt = instruction + input
        response, tokens = await self._fill_node(GenerateOp, prompt, mode="single_fill", return_usage=True)
        response["_usage_tokens"] = tokens
        return response

class GenerateCoT(Operator):
    def __init__(self, llm: LLM, name: str = "GenerateCoT"):
        super().__init__(llm, name)
        self.cot_prompt = GENERATE_COT_PROMPT

    async def __call__(self, input, instruction, return_usage=False):
        prompt = self.cot_prompt.format(input=input,instruction=instruction)
        response, tokens = await self._fill_node(GenerateOp, prompt, mode="single_fill", return_usage=True)
        response["_usage_tokens"] = tokens
        return response

class MultiGenerateCoT(Operator):
    def __init__(self, llm: LLM, name: str = "MultiGenerateCoT"):
        super().__init__(llm, name)
        self.cot_prompt = GENERATE_COT_PROMPT

    async def __call__(self, input, instruction, return_usage=False):
        prompt = self.cot_prompt.format(input=input,instruction=instruction)
        results = await asyncio.gather(
            self._fill_node(GenerateOp, prompt, mode="single_fill", return_usage=True),
            self._fill_node(GenerateOp, prompt, mode="single_fill", return_usage=True),
            self._fill_node(GenerateOp, prompt, mode="single_fill", return_usage=True)
        )

        response1, tokens1 = results[0]
        response2, tokens2 = results[1]
        response3, tokens3 = results[2]
        total_tokens = tokens1 + tokens2 + tokens3

        total_prompt_tokens = response1.get('_usage_prompt_tokens', 0) + response2.get('_usage_prompt_tokens', 0) + response3.get('_usage_prompt_tokens', 0)

        return {"response": [response1, response2, response3], "_usage_tokens": total_tokens, "_usage_prompt_tokens": total_prompt_tokens}

class ScEnsemble(Operator):
    def __init__(self, llm: LLM, name: str = "ScEnsemble"):
        super().__init__(llm, name)
        self.ensemble_prompt = SC_ENSEMBLE_PROMPT

    async def __call__(self, solutions: List[str], problem: str, return_usage=False):
        if not solutions:
            logger.warning("ScEnsemble called with empty solutions list, returning empty response")
            return {"response": "", "_usage_tokens": 0, "_usage_prompt_tokens": 0}

        if len(solutions) == 1:
            return {"response": solutions[0], "_usage_tokens": 0, "_usage_prompt_tokens": 0}

        answer_mapping = {}
        solution_text = ""
        for index, solution in enumerate(solutions):
            answer_mapping[chr(65 + index)] = index
            solution_text += f"{chr(65 + index)}: \n{str(solution)}\n\n\n"

        prompt = self.ensemble_prompt.format(problem=problem, solutions=solution_text)
        response, tokens = await self._fill_node(ScEnsembleOp, prompt, mode="xml_fill", return_usage=True)

        answer = response.get("solution_letter", "")
        answer = answer.strip().upper()

        return {"response": solutions[answer_mapping[answer]], "_usage_tokens": tokens, "_usage_prompt_tokens": response.get('_usage_prompt_tokens', 0)}

def run_code(code):
    try:
        global_namespace = {}

        disallowed_imports = [
            "os", "sys", "subprocess", "multiprocessing",
            "matplotlib", "seaborn", "plotly", "bokeh", "ggplot",
            "pylab", "tkinter", "PyQt5", "wx", "pyglet"
        ]
        for lib in disallowed_imports:
            if f"import {lib}" in code or f"from {lib}" in code:
                logger.info("Detected prohibited import: %s", lib)
                return "Error", f"Prohibited import: {lib} and graphing functionalities"

        exec(code, global_namespace)
        if 'solve' in global_namespace and callable(global_namespace['solve']):
            result = global_namespace['solve']()
            return "Success", str(result)
        else:
            return "Error", "Function 'solve' not found"
    except Exception as e:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        tb_str = traceback.format_exception(exc_type, exc_value, exc_traceback)
        return "Error", f"Execution error: {str(e)}\n{''.join(tb_str)}"
    
class Programmer(Operator):
    def __init__(self, llm: LLM, name: str = "Programmer"):
        super().__init__(llm, name)
        self.code_verifier_prompt = PYTHON_CODE_VERIFIER_PROMPT

    async def exec_code(self, code, timeout=100):
        import time
        loop = asyncio.get_running_loop()
        with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
            try:
                exec_start = time.time()
                future = loop.run_in_executor(executor, run_code, code)
                result = await asyncio.wait_for(future, timeout=timeout)
                exec_time = time.time() - exec_start
                return result[0], result[1], exec_time
            except asyncio.TimeoutError:
                executor.shutdown(wait=False, cancel_futures=True)
                return "Error", "Code execution timed out", 0.0
            except Exception as e:
                return "Error", f"Unknown error: {str(e)}", 0.0

    async def code_generate(self, problem, analysis, feedback, mode):
        prompt = self.code_verifier_prompt.format(
            problem=problem,
            analysis=analysis,
            feedback=feedback
        )
        response, tokens = await self._fill_node(CodeGenerateOp, prompt, mode, function_name="solve", return_usage=True)
        return response, tokens

    @retry(stop=stop_after_attempt(5), wait=wait_fixed(2))
    async def __call__(self, problem: str, analysis: str = "None", return_usage=False):
        code = None
        output = None
        feedback = ""
        total_tool_exec_time = 0.0
        total_tokens = 0
        total_prompt_tokens = 0

        for i in range(3):
            code_response, tokens = await self.code_generate(problem, analysis, feedback, mode="code_fill")
            total_tokens += tokens
            total_prompt_tokens += code_response.get('_usage_prompt_tokens', 0)
            code = code_response.get("code")
            if not code:
                return {"code": code, "output": "No code generated", "tool_exec_time": total_tool_exec_time, "_usage_tokens": total_tokens, "_usage_prompt_tokens": total_prompt_tokens}
            status, output, exec_time = await self.exec_code(code)
            total_tool_exec_time += exec_time

            if status == "Success":
                return {"code": code, "output": output, "tool_exec_time": total_tool_exec_time, "_usage_tokens": total_tokens, "_usage_prompt_tokens": total_prompt_tokens}
            else:
                print(f"Execution error on attempt {i + 1}, error message: {output}")
                feedback = (
                    f"\nThe result of the error from the code you wrote in the previous round:\n"
                    f"Code: {code}\n\nStatus: {status}, {output}"
                )
        return {"code": code, "output": output, "tool_exec_time": total_tool_exec_time, "_usage_tokens": total_tokens, "_usage_prompt_tokens": total_prompt_tokens}

class SelfRefine(Operator):
    def __init__(self, llm: LLM, name: str = "SelfRefine"):
        super().__init__(llm, name)
        self.selfrefine_prompt = SELFREFINE_PROMPT

    async def __call__(self, problem, solution, return_usage=False):
        prompt = self.selfrefine_prompt.format(problem=problem, solution=solution)
        response, tokens = await self._fill_node(SelfRefineOp, prompt, mode="single_fill", return_usage=True)
        response["_usage_tokens"] = tokens
        return response
    
class EarlyStop(Operator):
    def __init__(self, llm: LLM, name: str = "EarlyStop"):
        super().__init__(llm, name)

    async def __call__(self):
        return NotImplementedError
