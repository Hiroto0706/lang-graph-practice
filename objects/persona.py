from pydantic import BaseModel, Field


class Persona(BaseModel):
    name: str = Field(..., description="Persona name")
    background: str = Field(..., description="Background which Persona has")
