from typing import List
from pydantic import BaseModel, Field


class Persona(BaseModel):
    name: str = Field(..., description="Persona name")
    background: str = Field(..., description="Background which Persona has")


class Personas(BaseModel):
    personas: List[Persona] = Field(
        defalut_factory=List, description="List of Persona"
    )
