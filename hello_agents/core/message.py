"""统一消息格式，内部元数据与 API 消息分开。"""

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

# user：用户输入；assistant：模型回复；system：角色/规则；tool：程序执行工具后的结果。
# Literal 把 role 的取值限制在这四个字符串内，避免消息角色写错。
MessageRole = Literal["user", "assistant", "system", "tool"]


class Message(BaseModel):
    """一条对话消息；时间和元数据供程序记录，API 只接收协议需要的字段。"""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    content: str
    role: MessageRole
    # default_factory 在每次创建消息时调用，得到各自的时间和字典，避免共享可变默认值。
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = Field(default_factory=dict)
    tool_call_id: str | None = None  # 工具结果用它指明自己对应哪一次调用。
    tool_calls: list[dict[str, Any]] | None = None  # 模型请求执行的工具列表。

    def __init__(self, content: str, role: MessageRole, **kwargs: Any):
        """支持教程里的 Message("你好", "user") 写法，其余字段用关键字传入。"""
        super().__init__(content=content, role=role, **kwargs)

    def to_dict(self) -> dict[str, Any]:
        """转换成模型 API 消息；本地 timestamp、metadata 不随请求发送。"""
        result: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.role == "tool":
            # 一轮可调用多个工具，只有携带调用 ID，模型才能把结果和请求对上。
            if not self.tool_call_id:
                raise ValueError("tool 消息必须携带 tool_call_id")
            result["tool_call_id"] = self.tool_call_id
        if self.role == "assistant" and self.tool_calls:
            result["tool_calls"] = self.tool_calls
        return result

    def __str__(self) -> str:
        """print(message) 时显示便于阅读的“[角色] 内容”。"""
        return f"[{self.role}] {self.content}"
