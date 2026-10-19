"""ReAct：生成行动、执行工具、将观察结果送回模型。"""

from typing import Any

from ._parsing import parse_react_action
from ..core import Agent, AgentParseError, Config, MaxStepsExceeded, ToolError
from ..core.llm import ChatModel
from ..tools import ToolRegistry

# Prompt 是发给模型的指令模板；运行时填入可用工具、当前问题和本轮观察记录。
# Thought 表达简短的行动说明，Action 是程序实际解析并执行的部分。
DEFAULT_PROMPT = """解决任务时，每轮只决定一个行动。
可用工具及参数：
{tools}
先输出 Thought: 简短说明下一步，再输出 Action: 工具名[参数]。
参数使用 JSON 对象；单参数工具也可用原始文本。
完成任务时输出 Action: Finish[最终答案]。
只使用程序返回的 Observation，不自行编造工具执行结果。
任务：{question}
本次执行记录：
{history}
"""


class ReActAgent(Agent):
    """通过“决定行动 → 执行 → 观察”循环求解，Finish 动作表示完成任务。"""

    def __init__(
        self, name: str, llm: ChatModel, tool_registry: ToolRegistry,
        system_prompt: str | None = None, config: Config | None = None,
        max_steps: int | None = None, custom_prompt: str | None = None,
    ):
        """max_steps 限制模型调用次数；custom_prompt 可替换默认指令模板。"""
        super().__init__(name, llm, system_prompt, config)
        self.tool_registry = tool_registry
        self.max_steps = self.config.max_steps if max_steps is None else max_steps
        if self.max_steps < 1:
            raise ValueError("max_steps 必须大于 0")
        self.prompt_template = custom_prompt or DEFAULT_PROMPT
        # 与父类跨轮问答历史不同，这里只记录当前任务的动作和 Observation。
        self.current_history: list[str] = []

    def run(self, input_text: str, **kwargs: Any) -> str:
        """每轮最多处理一个动作，直到 Finish 或用完模型调用次数。"""
        messages = self._start(input_text)
        self.current_history = []
        for _ in range(self.max_steps):
            # 1. 把前面已经尝试的动作和观察结果写回提示词，供模型决定下一步。
            prompt = self.prompt_template.format(
                tools=self.tool_registry.get_tools_description(), question=input_text,
                history="\n".join(self.current_history),
            )
            # 完整运行记录放在本轮提示词，旧对话仍来自统一历史。
            response = self.llm.invoke(messages[:-1] + [{"role": "user", "content": prompt}], **self._options(kwargs))
            # messages[:-1] 保留系统设定和旧问答，替换末尾问题以免重复发送。
            self.current_history.append(response)
            try:
                # 2. 程序只执行明确的 Action 字段，不执行 Thought 里的自然语言。
                action = parse_react_action(response)
            except AgentParseError as exc:
                # 格式不对也计为一次模型调用；把错误加入观察记录，让下一轮修正。
                observation = f"格式错误：{exc}。请按协议重试。"
                self.steps.append({"response": response, "error": str(exc)})
                self.current_history.append("Observation: " + observation)
                continue
            if action.name == "Finish":
                # 3a. 完成信号不属于注册表工具，由 Agent 自己识别并返回最终答案。
                if not action.arguments:
                    raise AgentParseError("Finish 的最终答案不能为空")
                return self._remember(input_text, action.arguments)
            try:
                # 3b. 普通动作交给工具注册表。工具失败也变成观察，而不是伪造成功。
                observation = self.tool_registry.execute_tool(action.name, action.arguments)
            except ToolError as exc:
                observation = f"工具错误：{exc}"
            self.steps.append({"response": response, "tool": action.name, "arguments": action.arguments, "observation": observation})
            self.current_history.append("Observation: " + observation)
        # 模型始终不发 Finish 时由程序强制结束，防止行动循环无限继续。
        raise MaxStepsExceeded(f"{self.name} 在 {self.max_steps} 次模型调用内未返回 Finish")
