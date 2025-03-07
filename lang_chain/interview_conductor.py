from typing import List
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

from objects.interview import Interview, InterviewResult
from objects.persona import Persona


class InterviewConductor:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm

    def run(self, user_request: str, personas: List[Persona]) -> InterviewResult:
        questions = self._generate_questions(
            user_request=user_request, personas=personas
        )

        answers = self._generate_answers(
            personas=personas, questions=questions)
        interviews = self._create_interviews(
            personas=personas, questions=questions, answers=answers
        )

        return InterviewResult(interviews=interviews)

    def _generate_questions(
            self, user_request: str, personas: List[Persona]
    ) -> List[str]:
        question_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You're a specialist who creates questions related to the user's requirements."
                ),
                (
                    "human",
                    "Please create a question following the user's request related to the persona\n\n"
                    "User's request: {use_request}\n\n"
                    "Persona: {persona_name} - {persona_background}\n\n"
                    "Your question must be specific to extract important information from the persona."
                ),
            ]
        )

        question_chain = question_prompt | self.llm | StrOutputParser()

        question_queries = [
            {
                "user_request": user_request,
                "persona_name": persona.name,
                "persona_background": persona.background
            }
            for persona in personas
        ]

        return question_chain.batch(question_queries)

    def _generate_answers(
            self, personas: List[Persona], questions: List[str]
    ) -> List[str]:
        answer_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You need to answer as following persona: {persona_name} - {persona_background}",
                ),
                (
                    "human",
                    "Question: {question}",
                )
            ]
        )

        answer_chain = answer_prompt | self.llm | StrOutputParser()

        answer_queries = [
            {
                "persona_name": persona.name,
                "persona_background": persona.background,
                "question": question
            }
            for persona, question in zip(personas, questions)
        ]

        return answer_chain.batch(answer_queries)

    def _create_interviews(
            self, personas: List[Persona], questions: List[str], answers: List[str]
    ) -> List[Interview]:
        return [
            Interview(persona=persona, question=question, answer=answer)
            for persona, question, answer in zip(personas, questions, answers)
        ]
