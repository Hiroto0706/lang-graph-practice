from typing import List, Any
import json

from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


class RecommendationsOutput(BaseModel):
    recommendations: List[str]


class GenerateRecommendations:
    def __init__(self, llm: ChatOpenAI):
        # LLMを構造化出力用にラップ
        self.llm = llm.with_structured_output(RecommendationsOutput)

    def run(self, profile: str) -> List[str]:
        """
        ユーザープロファイルに基づき、具体的かつ実行可能な行動提案を生成して返却します。

        Args:
            profile (Any): AnalysisUser によって生成された解析結果（dictなど）

        Returns:
            List[str]: 行動提案のリスト
        """
        # プロンプト設定
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "あなたはユーザーのプロファイルに基づき、具体的かつ実行可能な行動提案を生成する専門家です。",
                ),
                (
                    "human",
                    "以下のユーザープロファイルに基づき、具体的な行動提案とその行動が推奨される理由ユーザーの情報よりJSON形式で出力してください。\n\n"
                    "プロファイル: {profile}",
                ),
            ]
        )
        chain = prompt | self.llm

        output = chain.invoke({"profile": profile})

        return output.recommendations
