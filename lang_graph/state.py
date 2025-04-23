import operator
from typing import Annotated

from pydantic import BaseModel, Field


class ProfileState(BaseModel):
    user_input: str = Field(..., description="ユーザーの質問の回答結果")
    profile: str = Field(default="", description="ユーザープロファイル情報")
    recommendations: Annotated[list[str], operator.add] = Field(
        default_factory=list, description="推奨されたアクティビティのリスト"
    )
    schedule: str = Field(default="", description="生成されたスケジュール")
    is_sufficient: bool = Field(default=False, description="情報が十分かどうか")
    iteration: int = Field(default=0, description="プロファイル生成と推奨の反復回数")
    final_output: str = Field(
        default="", description="最終的な成果物。ユーザーに推奨する行動が書かれてある"
    )
