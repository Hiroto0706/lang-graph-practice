from pydantic import BaseModel, Field


class EvaluationResult(BaseModel):
    reason: str = Field(..., description="Reason for Judgement")
    is_sufficient: bool = Field(..., description="Is there enough information")
