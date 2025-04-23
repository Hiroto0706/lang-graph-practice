import json
from typing import List, Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from objects.evaluation import EvaluationResult


class InformationEvaluator:
    def __init__(self, llm: ChatOpenAI):
        # LLMをOutputモデルEvaluationResultでラップ
        self.llm = llm.with_structured_output(EvaluationResult)

    def run(self, recommendations: List[str]) -> EvaluationResult:
        """
        提案された行動リストがユーザーのニーズに合致しているか評価し、
        is_sufficient（bool）とreason（評価理由）を返します。

        Args:
            recommendations (List[str]): 生成された行動提案のリスト

        Returns:
            EvaluationResult: is_sufficient, reason を含むオブジェクト
        """
        # プロンプトテンプレート定義
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "あなたは行動提案の評価専門家です。ユーザーのプロファイルに基づいて、"
                    "生成された行動提案が適切かどうかを判断してください。",
                ),
                (
                    "human",
                    "以下の行動提案リストについて、ユーザーにとって十分に実行可能で有用か評価してください。"
                    " true/false で is_sufficient を出力し、その理由をreason に記述してください。\n\n"
                    "提案リスト: {recommendations}",
                ),
            ]
        )
        # チェーン構築
        chain = prompt | self.llm
        # リストをJSON文字列化
        recs_json = json.dumps(recommendations, ensure_ascii=False)
        # 実行
        result: EvaluationResult = chain.invoke({"recommendations": recs_json})
        return result
