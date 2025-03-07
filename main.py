from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from lang_graph.document_agent import DocumentationAgent
load_dotenv()


def main():
    print("Hello LangGraph Practicing Project")

    import argparse

    parser = argparse.ArgumentParser(
        description="Create requirements from user's request"
    )

    parser.add_argument(
        "--task",
        type=str,
        help="Please input about you want to create application"
    )

    args = parser.parse_args()

    llm = ChatOpenAI(model="gpt-o3", temperature=0)

    agent = DocumentationAgent(llm=llm, k=args.k)

    final_output = agent.run(user_request=args.task)

    print(final_output)


if __name__ == "__main__":
    main()
