"""工具注册、发现、输入归一化与执行。"""

import json
from typing import Any, Callable

from .base import FunctionTool, Tool
from ..core.exceptions import ToolError


class ToolRegistry:
    """工具名到工具对象的映射，也是 Agent 执行工具的统一入口。

    模型只提交“用哪个工具、传什么参数”；这里查找对象、校验参数，再执行代码。
    Agent 无需知道每个工具到底是类工具，还是由函数包装而成。
    """

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register_tool(self, tool: Tool, *, replace: bool = False) -> None:
        """注册类工具，默认拒绝重名；有意替换时需显式传 replace=True。"""
        if tool.name in self._tools and not replace:
            raise ToolError(f"工具已注册：{tool.name}")
        self._tools[tool.name] = tool

    def register_function(self, name: str, description: str, func: Callable[..., Any], *, replace: bool = False) -> None:
        """先把普通函数包装成 FunctionTool，再复用同一套注册与执行流程。"""
        self.register_tool(FunctionTool(name, description, func), replace=replace)

    def get_tool(self, name: str) -> Tool | None:
        """只查找，不执行；没有这个名字时返回 None。"""
        return self._tools.get(name)

    def unregister(self, name: str) -> bool:
        """删除指定工具，返回它此前是否存在。"""
        return self._tools.pop(name, None) is not None

    def list_tools(self) -> list[str]:
        """列出工具名，保留注册顺序。"""
        return list(self._tools)

    def get_tools_description(self) -> str:
        """生成给文本型 Agent 使用的说明，包含工具用途和参数 JSON Schema。"""
        return "\n".join(
            f"- {tool.name}: {tool.description}\n  参数 JSON Schema: "
            + json.dumps(tool.parameter_model.model_json_schema(), ensure_ascii=False)
            for tool in self._tools.values()
        ) or "暂无可用工具"

    def to_openai_schema(self) -> list[dict[str, Any]]:
        """生成全部原生工具定义，FunctionCallAgent 将它放进请求的 tools 字段。"""
        return [tool.to_openai_schema() for tool in self._tools.values()]

    def execute_tool(self, name: str, input_data: str | dict[str, Any]) -> str:
        """查找、解析、验证并执行工具，成功返回文本，失败统一抛 ToolError。

        多参数工具传字典或 JSON 对象字符串；只有一个参数时，还可直接传
        原始文字，例如 execute_tool("calculator", "2 + 2")。
        """
        # 1. 名称查找：工具不存在就立即报错，不尝试猜测另一个工具。
        tool = self.get_tool(name)
        if tool is None:
            raise ToolError(f"未找到工具：{name}")
        try:
            parameters: Any = input_data
            if isinstance(input_data, str):
                # 2. 文本以 { 开头时按 JSON 对象解码；其他文本根据参数数量处理。
                if input_data.lstrip().startswith("{"):
                    parameters = json.loads(input_data)
                else:
                    fields = tool.parameter_model.model_fields
                    if not fields and not input_data.strip():
                        parameters = {}
                    elif len(fields) == 1:
                        # 单参数可推断唯一字段名，例如 "2+2" → {"expression": "2+2"}。
                        parameters = {next(iter(fields)): input_data}
                    else:
                        raise ToolError("多参数工具必须传入 JSON 对象")
            if not isinstance(parameters, dict):
                raise ToolError("工具参数必须是 JSON 对象，单参数工具也可传入原始文本")
            # 3. 类型转换、缺省值、未知字段等都先通过 Pydantic 检查。
            arguments = tool.validate_parameters(parameters)
            # 4. 真正执行工具函数；结果统一转成文字，便于回传模型。
            return str(tool.run(arguments))
        except ToolError:
            # 已经是框架工具异常时保留原类型与说明。
            raise
        except Exception as exc:
            # 包装 JSON 解码、参数校验、原始函数异常，让 Agent 统一捕获和反馈。
            raise ToolError(f"工具 {name} 执行失败：{exc}") from exc
