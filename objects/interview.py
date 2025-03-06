from typing import List
from pydantic import BaseModel, Field

from objects.persona import Persona


class Interview(BaseModel):
    persona: Persona = Field(..., description="Persona Persona")
    question: str = Field(..., description="Question to ask persona")
    answer: str = Field(..., description="Persona's answer")


class InterviewResult(BaseModel):
    interviews: List[Interview] = Field(
        default_factory=List, description="List of interview"
    )
