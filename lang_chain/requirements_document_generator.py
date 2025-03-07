from typing import List
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from objects.interview import Interview


class RequirementsDocumentGenerator:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm

    def run(self, user_request: str, interviews: List[Interview]) -> str:
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a specialist who creates requirements documents based on collected information.",
                ),
                (
                    "human",
                    "Based on the following user request and interview results from multiple personas, please create a requirements document.\n\n"
                    "User Request: {user_request}\n\n"
                    "Interview Results:\n{interview_results}\n"
                    "Please include the following sections in the requirements document:\n"
                    "1. Project Overview\n"
                    "2. Key Features\n"
                    "3. Non-functional Requirements\n"
                    "4. Constraints\n"
                    "5. Target Users\n"
                    "6. Priorities\n"
                    "7. Risks and Mitigation Strategies\n\n"
                    "Please ensure the output is in Japanese.\n\nRequirements Document:",
                ),
            ]
        )
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke(
            {
                "user_request": user_request,
                "interview_results": "\n".join(
                    f"Persona: {i.persona.name} - {i.persona.background}\n"
                    f"Question: {i.question}\nAnswer: {i.answer}\n"
                    for i in interviews
                ),
            }
        )
