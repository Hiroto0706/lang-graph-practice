from typing import List
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from objects.evaluation import EvaluationResult
from objects.interview import Interview


class InformationEvaluator:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm.with_structured_output(EvaluationResult)

    def run(self, user_request: str, interviews: List[Interview]) -> EvaluationResult:
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You're a specialist in assessing the sufficiency of information to create a comprehensive requirements document"
                ),
                (
                    "human",
                    f"Based on the user requests and interview results below, determine if you have enough information to create a comprehensive requirements document.\n\n"
                    "user's request: {user_request}\n\n"
                    "Interview results: {interview_results}",
                ),
            ]
        )

        chain = prompt | self.llm

        return chain.invoke(
            {
                "user_request": user_request,
                "interview_results": "\n".join(
                    f"Persona: {i.persona.name} - {i.persona.background}\n"
                    f"Question: {i.question}\n Answer: {i.answer}\n"
                    for i in interviews
                )
            }
        )
