import operator
from typing import Annotated, Any, Optional

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field
from lang_graph.document_agent import DocumentationAgent
load_dotenv()


def main(task: str):
    print("Hello LangGraph Practicing Project")

    # import argparse

    # parser = argparse.ArgumentParser(
    #     description="Create requirements from user's request"
    # )

    # parser.add_argument(
    #     "--task",
    #     type=str,
    #     help="Please input about you want to create application"
    # )

    # args = parser.parse_args()

    llm = ChatOpenAI(model="gpt-4o", temperature=0.0)

    agent = DocumentationAgent(llm=llm, k=5)

    final_output = agent.run(user_request=task)

    print(final_output)


if __name__ == "__main__":
    task = "読書メモから選択形式の問題を作成し、ユーザーが回答したら理解度を可視化してくれるアプリを開発したい"
    main(task)
