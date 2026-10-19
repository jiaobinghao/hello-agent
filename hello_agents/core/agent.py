"""所有范式共享的历史、调用参数和执行轨迹。"""

from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any

from .config import Config
from .llm import ChatModel
from .message import Message


class Agent(ABC):
    """所有 Agent 的公共底座，具体的行动循环由子类实现。

    可以把 Agent 理解成“拿着模型和工具完成任务的程序”。这里保存它的
    姓名、模型、角色设定与历史，不规定它采用对话、反思还是规划策略。
    ABC + abstractmethod 要求子类实现 run()，因此不能直接实例化 Agent。
    """

    def __init__(
        self,
        name: str,
        llm: ChatModel,
        system_prompt: str | None = None,
        config: Config | None = None,
    ):
        """llm 负责问模型；system_prompt 设定角色；config 控制调用和历史上限。"""
        self.name = name
        self.llm = llm
        self.system_prompt = system_prompt or "你是一个认真、清晰的中文助手。"
        # 优先使用显式配置，其次沿用模型配置；脚本模型没有 config 时采用默认值。
        # getattr(..., None) 允许模型没有该属性；深拷贝避免多个 Agent 共用可变配置。
        self.config = (config or getattr(llm, "config", None) or Config()).model_copy(deep=True)
        # _history 跨多轮保留问答；steps 只记录本次 run() 的中间过程。
        self._history: list[Message] = []
        self.steps: list[dict[str, Any]] = []

    @abstractmethod
    def run(self, input_text: str, **kwargs: Any) -> str:
        """完成一次任务并返回最终文本；失败时抛出框架异常。"""

    def add_message(self, message: Message) -> None:
        """保存一条消息，并把历史长度限制在配置的上限内。"""
        self._history.append(message.model_copy(deep=True))
        # [-N:] 表示只取列表最后 N 条；旧消息被丢弃，防止历史无限增长。
        self._history = self._history[-self.config.max_history_length:]

    def get_history(self) -> list[Message]:
        """返回历史副本，外部修改副本不会影响 Agent 下一轮使用的上下文。"""
        return [m.model_copy(deep=True) for m in self._history]

    def clear_history(self) -> None:
        """重置已保存的问答和当前执行轨迹，用于开始一段新对话。"""
        self._history.clear()
        self.steps.clear()

    def get_trace(self) -> list[dict[str, Any]]:
        """返回本轮执行轨迹的副本，例如调用了什么工具、得到了什么结果。"""
        return deepcopy(self.steps)

    def _start(self, input_text: str, system_prompt: str | None = None) -> list[dict[str, Any]]:
        """开始一轮任务：清空旧轨迹，组装“角色设定 + 旧对话 + 新问题”。"""
        self.steps.clear()
        if not isinstance(input_text, str) or not input_text.strip():
            raise ValueError("任务文本不能为空")
        return [
            {"role": "system", "content": system_prompt or self.system_prompt},
            # * 将生成器里的历史消息展开，逐条放入外层列表。
            *(m.to_dict() for m in self._history),
            {"role": "user", "content": input_text},
        ]

    def _options(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """合并模型调用参数；本次 run() 传入的参数可覆盖配置默认值。"""
        options: dict[str, Any] = {}
        if self.config.temperature is not None:
            options["temperature"] = self.config.temperature
        if self.config.max_tokens is not None:
            options["max_tokens"] = self.config.max_tokens
        return options | kwargs  # 字典 | 表示合并，同名键采用右边的值。

    def _remember(self, input_text: str, answer: str) -> str:
        """任务成功才保存一对问答，避免下次把半完成的流程当成已完成对话。"""
        # 只持久化成功的完整问答；工具过程可通过 get_trace() 查看。
        self.add_message(Message(input_text, "user"))
        self.add_message(Message(answer, "assistant"))
        # 奇数长度上限可能截断一对问答，移除开头孤立的回答以保留完整轮次。
        if self._history and self._history[0].role == "assistant":
            self._history.pop(0)
        return answer
