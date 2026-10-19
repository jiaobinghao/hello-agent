"""工具按顺序执行，上一步结果可供后一步模板引用。"""

from dataclasses import dataclass
from typing import Any

from .registry import ToolRegistry
from ..core.exceptions import ToolError


@dataclass(frozen=True)
class ChainStep:
    """工具链的一步：调用哪个工具、如何生成输入、把结果存在哪个变量下。"""

    tool_name: str
    input_template: str | dict[str, Any]
    output_key: str


class ToolChain:
    """预先排好的工具执行顺序，后一步可引用前一步的真实结果。

    例如第一步算 {input} 并保存为 subtotal，第二步算 {subtotal} * 2。
    和 Agent 不同，工具链不询问模型如何决策，步骤由开发者事先确定。
    """

    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self.steps: list[ChainStep] = []

    def add_step(self, tool_name: str, input_template: str | dict[str, Any], output_key: str | None = None) -> "ToolChain":
        """追加步骤；input_template 支持 {变量名}，返回 self 以支持连续调用。"""
        key = output_key or f"step_{len(self.steps)}_result"
        # input 留给初始输入；结果名也不能重复，否则后面的步骤会拿错数据。
        if key == "input" or any(step.output_key == key for step in self.steps):
            raise ToolError(f"工具链输出变量保留或重复：{key}")
        self.steps.append(ChainStep(tool_name, input_template, key))
        return self

    def execute(self, registry: ToolRegistry, initial_input: str, context: dict[str, Any] | None = None) -> str:
        """顺序执行全部步骤，返回最后一步的结果；任何一步失败都立即停止。"""
        if not self.steps:
            raise ToolError("工具链至少需要一个步骤")
        # 复制上下文以免改动调用方的字典，初始输入固定放到 input 变量里。
        values = dict(context or {}) | {"input": initial_input}
        for step in self.steps:
            try:
                # 1. 执行前才替换模板，因此能读到前一步刚写入的结果。
                data = self._render(step.input_template, values)
            except (KeyError, ValueError, IndexError, AttributeError) as exc:
                raise ToolError(f"工具链模板无法解析：{exc}") from exc
            # 2. 执行后按结果名保存，下一步即可通过 {结果名} 引用。
            values[step.output_key] = registry.execute_tool(step.tool_name, data)
        return values[self.steps[-1].output_key]

    @classmethod
    def _render(cls, value: Any, context: dict[str, Any]) -> Any:
        """递归替换模板里的字符串；字典/列表只遍历其中的值，数字等原样保留。"""
        if isinstance(value, str):
            # 例如 "{subtotal} * 2" 在 subtotal=5 时会变成 "5 * 2"。
            return value.format_map(context)
        if isinstance(value, dict):
            return {key: cls._render(item, context) for key, item in value.items()}
        if isinstance(value, list):
            return [cls._render(item, context) for item in value]
        return value


class ToolChainManager:
    """按名称保存多条工具链，所有链共用一个工具注册表。"""

    def __init__(self, registry: ToolRegistry):
        self.registry = registry
        self.chains: dict[str, ToolChain] = {}

    def register_chain(self, chain: ToolChain) -> None:
        """登记一条链并拒绝同名覆盖，避免误用另一套执行步骤。"""
        if chain.name in self.chains:
            raise ToolError(f"工具链已注册：{chain.name}")
        self.chains[chain.name] = chain

    def execute_chain(self, name: str, initial_input: str, context: dict[str, Any] | None = None) -> str:
        """查找指定链，并将初始输入与额外变量交给它执行。"""
        if name not in self.chains:
            raise ToolError(f"未找到工具链：{name}")
        return self.chains[name].execute(self.registry, initial_input, context)
