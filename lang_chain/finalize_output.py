from typing import List

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


class FinalizeOutput:
    def __init__(self, llm: ChatOpenAI):
        """
        行動提案をまとめて、ユーザー向けのわかりやすいライフスタイル戦略を生成します。
        """
        self.llm = llm

    def run(self, recommendations: List[str], profile: str) -> str:
        """
        Args:
            recommendations (List[str]): 生成された行動提案のリスト

        Returns:
            str: ユーザー向けの最終プラン文章
        """
        # プロンプトテンプレート定義
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "あなたはユーザーに対して親切でわかりやすい人生戦略を作成する専門家です。",
                ),
                (
                    "human",
                    "以下の行動提案をもとに、目的別・優先度別に整理した人生戦略を作成してください。"
                    "各提案がいつ・どのように実行すべきか、ユーザー特徴から、なぜこの行動が適切かを具体的に説明してください。\n\n"
                    "また、ユーザーが行動しやすいようにするために、各行動が何歳の時点で達成されていると良いか、その行動の達成基準は何かを示してください。\n\n"
                    "提案リスト:\n{recommendations}\n"
                    "ユーザープロファイル:\n{profile}\n",
                ),
            ]
        )
        # チェーン構築
        chain = prompt | self.llm | StrOutputParser()

        # recommendations を連結して渡す
        input_data = {"recommendations": "\n".join(
            recommendations), "profile": profile}
        # 実行して最終プランを取得
        return chain.invoke(input_data)
