from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from lang_graph.life_planner_agent import LifePlannerAgent

load_dotenv()


def main():
    print("Hello LangGraph Practicing Project")

    # ユーザーからのプロファイル取得用の質問リスト
    questions = [
        "あなたの性別は？",
        "あなたの年齢は？",
        "あなたの性格は？",
        "あなたのMBTIは？",
        "好きなもの・ことは？",
        "嫌いなもの・ことは？",
        "得意なことは？",
        "苦手なことは？",
        "現在の仕事は？",
        "将来どうなっていたい？",
        "どんな人に憧れる？",
        "これは長く続けられるなってことは？",
    ]

    answers = []
    for idx, question in enumerate(questions, start=1):
        ans = input(f"Q{idx}. {question} -> ")
        answers.append(f"{question} {ans}")

    # 回答を一つの文字列にまとめてユーザーリクエストとして渡す
    user_profile = "\n".join(answers)

    print("\n=== User Profile ===")
    print(user_profile)

    # LLM とエージェントの初期化
    llm = ChatOpenAI(model="gpt-4o", temperature=0.0)
    agent = LifePlannerAgent(llm=llm)

    # エージェント実行
    final_output = agent.run(user_input=user_profile)

    print("\n=== Generated Plan ===")
    print(final_output)


if __name__ == "__main__":
    main()
