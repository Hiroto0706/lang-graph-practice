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
        "好きなものは？",
        "嫌いなものは？",
        "現在の仕事は？",
        "将来どうなっていたい？",
        "起床時間は？",
        "就寝時間は？",
        "１日に確保したい自由時間（趣味・休憩など）は？",
        "集中しやすい時間帯（朝派／夜派）は？",
        "優先度の高いタスク領域（例：勉強／運動／仕事準備など）は？",
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
