from pydantic import BaseModel, Field


class GenerateOp(BaseModel):
    response: str = Field(default="", description="Your answer for this multiple choice question")

class CodeGenerateOp(BaseModel):
    code: str = Field(default="", description="Your complete code solution for this problem")

class ScEnsembleOp(BaseModel):
    solution_letter: str = Field(default="", description="The letter of most consistent solution.")

class SelfRefineOp(BaseModel):
    response: str = Field(default="", description="Your refined answer for this multiple choice question")
