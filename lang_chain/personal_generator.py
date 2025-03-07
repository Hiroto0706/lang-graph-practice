from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from objects.persona import Personas


class PersonalGenerator:
    def __init__(self, llm: ChatOpenAI, k: int = 5):
        self.llm = llm.with_structed_output(Personas)
        self.k = k

    def run(self, user_request: str) -> Personas:
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a specialist who creates diverse personas for user interviews. "
                    "Generate personas that are varied in terms of age, gender, occupation, and technical domain knowledge."),
                (
                    "human",
                    f"Create {self.k} different personas based on the following user request:\n\n"
                    "User's request: {user_request}\n\n"
                    "For each persona, include a name and brief background. Ensure diversity in age, gender, occupation, and technical domain knowledge."
                ),
            ]
        )
        chain = prompt | self.llm
        return chain.invoke({"user_request": user_request})
