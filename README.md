# lang-graph-practice

This is a repository for practicing LangGraph.

# What is this?

This repo is references following book.

https://www.amazon.co.jp/LangChain%E3%81%A8LangGraph%E3%81%AB%E3%82%88%E3%82%8BRAG%E3%83%BBAI%E3%82%A8%E3%83%BC%E3%82%B8%E3%82%A7%E3%83%B3%E3%83%88%EF%BC%BB%E5%AE%9F%E8%B7%B5%EF%BC%BD%E5%85%A5%E9%96%80-%E3%82%A8%E3%83%B3%E3%82%B8%E3%83%8B%E3%82%A2%E9%81%B8%E6%9B%B8-%E8%A5%BF%E8%A6%8B-%E5%85%AC%E5%AE%8F/dp/4297145308

# My Motivation

I want to learn LangGraph because I hope to develop applications using RAG and AI Agents.

So I learn this book about LangChain, LangGraph and AI Agents.

# What I learned this project

I learned about how to use LangGraph on this project.

Important things are State, Node, and Edge.

State is memory for AI Agent.
State make you hold a lot of information for you needed.

You write what you want to keep information on State.

```python
class InterviewState(BaseModel):
    user_request: str = Field(..., description="ユーザーからのリクエスト")
    personas: Annotated[list[Persona], operator.add] = Field(
        default_factory=list, description="生成されたペルソナのリスト"
    )
    interviews: Annotated[list[Interview], operator.add] = Field(
        default_factory=list, description="実施されたインタビューのリスト"
    )
    requirements_doc: str = Field(default="", description="生成された要件定義")
    iteration: int = Field(
        default=0, description="ペルソナ生成とインタビューの反復回数"
    )
    is_information_sufficient: bool = Field(
        default=False, description="情報が十分かどうか"
    )
```

For example, request, answer history, and does needed to rerun llm etc...

Node is a component for LangGraph.
LangGraph run from node to node.

You can add node to workflow using `add_node` method.

You can do that like below code.

```python
        # 各ノードの追加
        workflow.add_node("generate_personas", self._generate_personas)
        workflow.add_node("conduct_interviews", self._conduct_interviews)
        workflow.add_node("evaluate_information", self._evaluate_information)
        workflow.add_node("generate_requirements", self._generate_requirements)
```

Edge is a connection component between nodes.

You can add edge below code.

```python
        # エントリーポイントの設定
        workflow.set_entry_point("generate_personas")

        # ノード間のエッジの追加
        workflow.add_edge("generate_personas", "conduct_interviews")
        workflow.add_edge("conduct_interviews", "evaluate_information")

        # 条件付きエッジの追加
        workflow.add_conditional_edges(
            "evaluate_information",
            lambda state: not state.is_information_sufficient and state.iteration < 5,
            {True: "generate_personas", False: "generate_requirements"},
        )
        workflow.add_edge("generate_requirements", END)
```

`set_entry_point` method can make you set start node.
This code, you set generate_personas as start node.

`add_edge` is normal edge, it can connect between nodes.

`conditional_edges` make condition for Node.
1st argument is source node, 2nd argument is condition for node.
3rd argument is conditional branch destination. True section is destination that if 2nd argument condition is true, False is False ver.

 `add_edge` make you set last node.
You can set final node using `add_edge`.


Finally, you invoked `workflow.compile()` method, you are able to create compiled workflow.
That object based on Runnable Class, so you can use invoke method and batch method.

```python
        return workflow.compile()
```

If you run LangGraph, you write code like below.

```python
    def run(self, user_request: str) -> str:
        # 初期状態の設定
        initial_state = InterviewState(user_request=user_request)
        # グラフの実行
        final_state = self.graph.invoke(initial_state)
        # 最終的な要件定義書の取得
        return final_state["requirements_doc"]
```

First, you create initial_state.
Second, you invoke lang graph using invoke method.
Finally, you return data you need.