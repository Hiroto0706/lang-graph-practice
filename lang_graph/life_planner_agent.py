from typing import Any

from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from lang_chain.information_evaluator import InformationEvaluator
from lang_chain.generate_recommendations import GenerateRecommendations
from lang_chain.analysis_user import AnalysisUser
from lang_chain.finalize_output import FinalizeOutput
from lang_graph.state import ProfileState


class LifePlannerAgent:
    def __init__(self, llm: ChatOpenAI):
        # 各種ジェネレータの初期化
        self.analysis_user = AnalysisUser(llm=llm)
        self.generate_recommendations = GenerateRecommendations(llm=llm)
        self.information_evaluator = InformationEvaluator(llm=llm)
        self.finalize_output = FinalizeOutput(llm=llm)

        # グラフの作成
        self.graph = self._create_graph()

    def _create_graph(self) -> StateGraph:
        # グラフの初期化
        workflow = StateGraph(ProfileState)

        # 各ノードの追加
        workflow.add_node("analysis_user", self._analysis_user)
        workflow.add_node("generate_recommendations", self._generate_recommendations)
        workflow.add_node("evaluate_recommendations", self._evaluate_recommendations)
        workflow.add_node("finalize_output", self._finalize_output)

        # エントリーポイント設定
        workflow.set_entry_point("analysis_user")

        # 通常エッジ
        workflow.add_edge("analysis_user", "generate_recommendations")
        workflow.add_edge("generate_recommendations", "evaluate_recommendations")

        # 条件付きエッジの追加
        workflow.add_conditional_edges(
            "evaluate_recommendations",
            lambda state: not state.is_sufficient and state.iteration < 5,
            {True: "analysis_user", False: "finalize_output"},
        )
        workflow.add_edge("finalize_output", END)

        # グラフのコンパイル
        return workflow.compile()

    def _analysis_user(self, state: ProfileState) -> dict[str, Any]:
        # ユーザープロファイルを解析
        profile = self.analysis_user.run(state.user_input)
        return {
            "profile": profile,
            "iteration": state.iteration + 1,
        }

    def _generate_recommendations(self, state: ProfileState) -> dict[str, Any]:
        # プロファイルに基づき行動提案を生成
        recommendations: list[str] = self.generate_recommendations.run(state.profile)
        return {"recommendations": recommendations}

    def _evaluate_recommendations(self, state: ProfileState) -> dict[str, Any]:
        # 提案が適切か評価
        result = self.information_evaluator.run(state.recommendations)
        return {
            "is_sufficient": result.is_sufficient,
            "reason": result.reason,
        }

    def _finalize_output(self, state: ProfileState) -> dict[str, Any]:
        # 最終プランを生成・出力
        plan: str = self.finalize_output.run(state.recommendations, state.profile)
        return {"final_output": plan}

    def run(self, user_input: str) -> str:
        # 初期状態設定＆グラフ走査
        initial_state = ProfileState(user_input=user_input)
        final_state = self.graph.invoke(initial_state)
        return final_state["final_output"]
