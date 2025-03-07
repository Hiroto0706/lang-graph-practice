from typing import Annotated, List
from objects.interview import Interview
from objects.persona import Persona
from pydantic import BaseModel, Field
import operator


class InterviewState(BaseModel):
    user_request: str = Field(..., description="Request from user")
    personas: Annotated[List[Persona], operator.add] = Field(
        default_factory=List, description="List of persona"
    )
    interviews: Annotated[List[Interview], operator.add] = Field(
        default_factory=List, description="List of interviews conducted"
    )
    requirements_doc: str = Field(
        default="", description="requirements documents")
    iteration: int = Field(
        default=0, description="Number of persona generation and interview iterations"
    )
    is_information_sufficient: bool = Field(
        default=False, description="Whether enough of information"
    )
