"""可注入的脚本模型：离线学习编排行为，不模拟真实模型的智能。"""

from collections import deque
from collections.abc import Iterator
from copy import deepcopy
from typing import Any

from openai.types.chat import ChatCompletionMessage

from .core import LLMError


class ScriptedLLM:
    """按顺序消费预设回复，替代真实模型驱动同一套 Agent 代码。

    例：ScriptedLLM(["[TOOL_CALL:calculator:2+2]", "答案是4"])。
    第一次调用返回工具请求，第二次返回答案；工具仍由程序实际执行。
    requests 保存发给“模型”的消息，测试可检查程序是否把工具结果送回去了。
    """

    provider = "scripted"
    model = "offline-script"

    def __init__(self, responses: list[str | dict[str, Any]]):
        """文本回复写字符串；原生工具回复写包含 tool_calls 的消息字典。"""
        # deque 是双端队列，popleft 可依次取走队头；深拷贝避免改动外部脚本。
        self.responses = deque(deepcopy(responses))
        self.requests: list[dict[str, Any]] = []

    def _next(self, messages: list[dict[str, Any]], kwargs: dict[str, Any]) -> str | dict[str, Any]:
        """记录本次调用快照，再消费一个响应；脚本用完时明确报错。"""
        self.requests.append(deepcopy({"messages": messages, "options": kwargs}))
        if not self.responses:
            raise LLMError("离线脚本响应已用完；请为每次模型调用提供一个响应")
        return self.responses.popleft()

    def invoke(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        """实现文本模型接口；不分析问题内容，只返回下一个预设字符串。"""
        response = self._next(messages, kwargs)
        if not isinstance(response, str):
            raise LLMError("文本接口需要脚本中的字符串响应")
        return response

    def stream(self, messages: list[dict[str, Any]], **kwargs: Any) -> Iterator[str]:
        """yield from 逐个产出字符串的字符，用于演示流式消费，不模拟真实网络分块。"""
        yield from self.invoke(messages, **kwargs)

    def complete(self, messages: list[dict[str, Any]], **kwargs: Any) -> ChatCompletionMessage:
        """把预设字典转换成 SDK 消息对象，使原生工具 Agent 使用相同的读取方式。"""
        response = self._next(messages, kwargs)
        if isinstance(response, str):
            # 简单文本也可作为原生流程的最后一条回复，包装成 assistant 消息即可。
            response = {"role": "assistant", "content": response}
        return ChatCompletionMessage.model_validate(response)
