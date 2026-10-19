"""原生 Chat Completions 工具调用，完整保留 assistant/tool 消息配对。"""

import json
from typing import Any

from ..core import Agent, AgentParseError, Config, MaxStepsExceeded, ToolError
from ..core.llm import ChatModel
from ..tools import ToolRegistry


class FunctionCallAgent(Agent):
    """使用模型 API 的结构化工具调用，而非在回答文字中寻找工具标记。

    一次往返：发送工具定义 → 收到 assistant.tool_calls → 本地执行函数 →
    把每个结果写成 tool 消息送回 → 模型继续调用工具或生成最终答案。
    """

    def __init__(
        self, name: str, llm: ChatModel, tool_registry: ToolRegistry | None = None,
        system_prompt: str | None = None, config: Config | None = None,
        max_steps: int | None = None,
    ):
        """tool_registry 提供工具及其参数定义；max_steps 是整轮任务的模型调用上限。"""
        super().__init__(name, llm, system_prompt, config)
        self.tool_registry = tool_registry if tool_registry is not None else ToolRegistry()
        self.max_steps = self.config.max_steps if max_steps is None else max_steps
        if self.max_steps < 1:
            raise ValueError("max_steps 必须大于 0")

    def run(self, input_text: str, **kwargs: Any) -> str:
        """保留每个调用 ID，保证模型能够把工具结果与自己的请求一一对应。"""
        messages = self._start(input_text)
        options = self._options(kwargs)
        if {"tools", "stream"} & options.keys():
            raise ValueError("FunctionCallAgent 的 tools 和 stream 参数由框架管理")
        # 1. JSON Schema 描述工具名称、用途和参数结构；它只是说明书，不执行函数。
        schemas = self.tool_registry.to_openai_schema()
        if schemas:
            options["tools"] = schemas
        for _ in range(self.max_steps):
            # 2. complete 返回消息对象，才能同时读取 content 与 tool_calls。
            message = self.llm.complete(messages, **options)
            calls = message.tool_calls or []
            if not calls:
                # 3a. 没有新的工具请求且有非空文字，才视为最终答案。
                if not message.content or not message.content.strip():
                    raise AgentParseError("模型没有返回工具调用或最终答案")
                return self._remember(input_text, message.content)
            # 3b. 同一轮可能请求多个工具，每次调用都必须有独立 ID。
            ids = [call.id for call in calls]
            if any(not call_id for call_id in ids) or len(set(ids)) != len(ids):
                raise AgentParseError("工具调用 ID 缺失或重复")
            # SDK 中的其他字段（例如兼容服务的 reasoning_content）也应传回。
            messages.append(message.model_dump(exclude_none=True))
            # model_dump 将 SDK 消息对象转为字典；exclude_none 省略空字段。
            for call in calls:
                try:
                    # 4. arguments 在协议里是 JSON 字符串，先解码成字典，再验证和执行。
                    arguments = json.loads(call.function.arguments)
                    if not isinstance(arguments, dict):
                        raise ToolError("原生函数调用参数必须是 JSON 对象")
                    result = self.tool_registry.execute_tool(call.function.name, arguments)
                except (ValueError, ToolError) as exc:
                    # 参数无效也要回填对应结果，不能让某个调用缺少响应。
                    result = f"工具错误：{exc}"
                # 5. 每个 tool 消息必须引用原调用 ID；下一轮模型才能使用真实结果。
                messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
                self.steps.append({"tool": call.function.name, "tool_call_id": call.id,
                                   "arguments": call.function.arguments, "observation": result})
        raise MaxStepsExceeded(f"{self.name} 在 {self.max_steps} 次模型调用内未完成")
