"""基础对话与教程中的文本工具标记。"""

from collections.abc import Iterator
from typing import Any

from ._parsing import TextToolCall, parse_tool_markers
from ..core import Agent, AgentParseError, Config, MaxStepsExceeded, ToolError
from ..core.llm import ChatModel
from ..tools import Tool, ToolRegistry


class SimpleAgent(Agent):
    """最适合入门的 Agent：普通对话加一层可选的工具执行循环。

    没有工具：问题 → 模型 → 答案。
    有工具：问题 → 模型要求调用工具 → 程序执行 → 结果送回模型 → 答案。
    工具请求以普通文本中的 [TOOL_CALL:名称:参数] 标记表达。
    """

    def __init__(
        self, name: str, llm: ChatModel, system_prompt: str | None = None,
        config: Config | None = None, tool_registry: ToolRegistry | None = None,
        enable_tool_calling: bool = True,
    ):
        """tool_registry 保存可用工具；enable_tool_calling 可暂时禁用工具能力。"""
        super().__init__(name, llm, system_prompt, config)
        self.tool_registry = tool_registry
        self.enable_tool_calling = enable_tool_calling

    def has_tools(self) -> bool:
        """只有开关启用、存在注册表且注册表非空时，才进入工具调用模式。"""
        return bool(self.enable_tool_calling and self.tool_registry and self.tool_registry.list_tools())

    def add_tool(self, tool: Tool) -> None:
        """运行前或两轮对话之间都可添加工具，并自动启用工具模式。"""
        if self.tool_registry is None:
            self.tool_registry = ToolRegistry()
        self.tool_registry.register_tool(tool)
        self.enable_tool_calling = True

    def remove_tool(self, name: str) -> bool:
        """按名称移除工具，返回这次是否真的移除了一个工具。"""
        return self.tool_registry.unregister(name) if self.tool_registry else False

    def list_tools(self) -> list[str]:
        """返回已注册工具的名称，便于检查 Agent 当前具备哪些能力。"""
        return self.tool_registry.list_tools() if self.tool_registry else []

    def _prompt(self) -> str:
        """给角色设定补上工具说明和输出格式，让模型知道怎样请求执行工具。"""
        if not self.has_tools():
            return self.system_prompt
        return (
            self.system_prompt + "\n可用工具：\n" + self.tool_registry.get_tools_description()
            + '\n需要工具时输出 [TOOL_CALL:工具名:JSON对象]，例如 '
            + '[TOOL_CALL:calculator:{"expression":"2+2"}]。'
            + "工具由程序执行；拿到真实结果后再回答，不要编造结果。无需工具时直接回答。"
        )

    def _observe(self, messages: list[dict[str, Any]], response: str, calls: list[TextToolCall]) -> None:
        """执行本轮请求的工具，把真实结果追加到本次任务的消息列表中。"""
        # 先保留模型刚才发出的工具请求，再追加结果，模型才知道结果从何而来。
        messages.append({"role": "assistant", "content": response})
        observations = []
        for call in calls:
            try:
                # 模型只写了工具名称和参数；真正的 Python 函数在注册表里执行。
                result = self.tool_registry.execute_tool(call.name, call.arguments)
            except ToolError as exc:
                # 错误也作为观察结果反馈，模型下一轮有机会换参数或换工具。
                result = f"工具错误：{exc}"
            self.steps.append({"tool": call.name, "arguments": call.arguments, "observation": result})
            observations.append(f"{call.name}: {result}")
        # 这里使用的是文本协议，结果作为补充 user 消息发送。
        # 原生 FunctionCallAgent 才使用带 tool_call_id 的 tool 消息。
        messages.append({"role": "user", "content": "工具执行结果：\n" + "\n".join(observations)})

    def run(self, input_text: str, max_tool_iterations: int | None = None, **kwargs: Any) -> str:
        """完成一轮任务；可单独限制工具轮数，总模型调用次数受 config.max_steps 限制。"""
        # 1. 组装角色设定、历史和本次问题，这个列表会随工具执行逐步增长。
        messages = self._start(input_text, self._prompt())
        if max_tool_iterations is not None and max_tool_iterations < 0:
            raise ValueError("max_tool_iterations 不能为负数")
        tool_rounds = 0
        for _ in range(self.config.max_steps):
            # 2. 让模型决定直接回答，还是先请求工具。一次 invoke 就是一次模型调用。
            response = self.llm.invoke(messages, **self._options(kwargs))
            # 3. 先解析整轮文本；解析出错时还没执行工具，可避免执行半个请求。
            calls = parse_tool_markers(response) if self.has_tools() else []
            if not calls:
                # 4a. 没有工具标记，说明这是最终回答，保存问答并返回。
                return self._remember(input_text, response)
            if max_tool_iterations is not None and tool_rounds >= max_tool_iterations:
                raise MaxStepsExceeded("已达到文本工具调用轮数上限")
            # 4b. 有工具标记，执行并追加结果；下一次循环会把这些结果交给模型。
            self._observe(messages, response, calls)
            tool_rounds += 1
        # 达到上限仍然没有最终回答，明确报错，不把工具请求当成任务完成。
        raise MaxStepsExceeded(f"{self.name} 在 {self.config.max_steps} 次模型调用内未完成")

    def stream_run(self, input_text: str, **kwargs: Any) -> Iterator[str]:
        """以生成器形式返回回答，调用方通过 for 循环读取输出。

        无工具时实时产出每个片段；有工具时先收齐整轮文本，再判断是否需要
        执行工具，最终回答才产出给用户，因此工具模式不是逐片段实时输出。
        """
        messages = self._start(input_text, self._prompt())
        use_tools = self.has_tools()
        for _ in range(self.config.max_steps):
            chunks = []
            for chunk in self.llm.stream(messages, **self._options(kwargs)):
                chunks.append(chunk)
                if not use_tools:
                    # yield 暂停本函数，把片段交给调用方；下次迭代从这里继续。
                    yield chunk
            response = "".join(chunks)
            if not response.strip():
                raise AgentParseError("流式响应为空")
            calls = parse_tool_markers(response) if use_tools else []
            if not calls:
                if use_tools:
                    yield response
                # 调用方完整读完这一轮后才记录历史，提前关闭生成器不保存半轮回答。
                self._remember(input_text, response)
                return
            self._observe(messages, response, calls)
        raise MaxStepsExceeded(f"{self.name} 在 {self.config.max_steps} 次模型调用内未完成")
