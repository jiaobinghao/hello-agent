"""工具的参数验证和自描述接口。"""

import inspect
import re
from abc import ABC, abstractmethod
from functools import cached_property
from typing import Any, Callable, Literal, get_type_hints

from pydantic import BaseModel, ConfigDict, Field, create_model

from ..core.exceptions import ToolError


class ToolParameter(BaseModel):
    """工具参数的说明，用于生成校验规则和提供给模型的“参数说明书”。

    例如 expression 是参数名，string 是类型，required 表示是否必填。
    required=False 时，调用方不传该字段就使用 default。
    """

    name: str
    type: Literal["string", "integer", "number", "boolean", "array", "object"] = "string"
    description: str = ""
    required: bool = True
    default: Any = None


class Tool(ABC):
    """所有工具的共同约定：描述需要什么参数，并实现怎样执行。

    新增类工具时只需实现 get_parameters() 和 run()；注册表负责统一调用。
    模型能看到的 Schema 来自参数定义，真正的操作发生在 run() 中。
    """

    def __init__(self, name: str, description: str):
        """name 是模型用于选择工具的标识，description 说明用途。"""
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_-]{0,63}", name):
            raise ToolError("工具名称必须是最多 64 位的字母、数字、下划线或连字符，以字母或下划线开头")
        self.name = name
        self.description = description

    @abstractmethod
    def run(self, parameters: dict[str, Any]) -> str:
        """执行已经验证的参数，返回文本。"""

    @abstractmethod
    def get_parameters(self) -> list[ToolParameter]:
        """描述此工具的全部参数。"""

    @cached_property
    def parameter_model(self) -> type[BaseModel]:
        """按工具参数动态生成 Pydantic 校验类，首次访问后缓存结果。

        返回的是“类”而不是已填好值的对象；之后用这个类校验每次工具输入。
        """
        # JSON Schema 的类型名称需要先映射到 Python 类型，Pydantic 才能校验。
        types = {
            "string": str, "integer": int, "number": float,
            "boolean": bool, "array": list[Any], "object": dict[str, Any],
        }
        fields = {}
        for parameter in self.get_parameters():
            # create_model 要求每项为 (字段类型, 默认值/字段配置)。
            # ... 在这里表示必填；Field 还携带可供模型阅读的参数说明。
            fields[parameter.name] = (
                types[parameter.type],
                Field(default=... if parameter.required else parameter.default,
                      description=parameter.description),
            )
        return create_model(
            f"{self.name}Parameters", __config__=ConfigDict(extra="forbid"), **fields,
        )

    def validate_parameters(self, parameters: dict[str, Any]) -> dict[str, Any]:
        """校验输入并填入默认值，返回归一化字典，例如将整数参数 "3" 转成 3。"""
        return self.parameter_model.model_validate(parameters).model_dump()

    def to_openai_schema(self) -> dict[str, Any]:
        """生成原生工具调用所需的工具定义，告诉模型工具叫什么、参数怎么填。"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameter_model.model_json_schema(),
            },
        }


class FunctionTool(Tool):
    """把普通 Python 函数包装为工具，无需为每个简单函数编写 Tool 子类。

    例如 def multiply(value: int, factor: int = 2) 会生成两个整数参数，
    value 必填、factor 默认 2；后续由 Pydantic 校验，再调用原函数。
    """

    def __init__(self, name: str, description: str, func: Callable[..., Any]):
        """检查函数签名，用类型标注和默认值构造参数模型。"""
        super().__init__(name, description)
        if inspect.iscoroutinefunction(func):
            raise ToolError("函数工具请使用同步函数；并发执行由 AsyncToolExecutor 调度")
        self.func = func
        signature = inspect.signature(func)  # 读取参数名、参数种类和默认值。
        annotations = get_type_hints(func)  # 读取并解析 int、list[str] 等类型标注。
        fields = {}
        for parameter in signature.parameters.values():
            # 执行时统一用 func(**arguments)，因此只支持可按名称传入的参数。
            if parameter.kind not in {
                inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY,
            }:
                raise ToolError("函数工具只支持具名参数，不支持位置专用参数、*args 或 **kwargs")
            annotation = annotations.get(parameter.name, str)  # 未写类型时按字符串处理。
            default = ... if parameter.default is inspect.Parameter.empty else parameter.default
            fields[parameter.name] = (annotation, default)
        # 函数签名可表达更丰富的类型，所以直接生成模型，覆盖父类的按需生成逻辑。
        self.parameter_model = create_model(
            f"{name}Parameters", __config__=ConfigDict(extra="forbid"), **fields,
        )

    def get_parameters(self) -> list[ToolParameter]:
        """把函数参数模型转换成通用工具参数说明，原生 Schema 仍直接使用完整模型。"""
        schema = self.parameter_model.model_json_schema()
        parameters = []
        for name, info in schema.get("properties", {}).items():
            # 可选参数的 Schema 可能用 anyOf 表示“某类型或 null”，这里提取非空类型。
            types = info.get("anyOf", [info])
            kind = next((t.get("type") for t in types if t.get("type") != "null"), "string")
            parameters.append(ToolParameter(
                name=name,
                type=kind if kind in {"string", "integer", "number", "boolean", "array", "object"} else "object",
                description=info.get("description", ""),
                required=name in schema.get("required", []),
                default=info.get("default"),
            ))
        return parameters

    def run(self, parameters: dict[str, Any]) -> str:
        """把校验后的参数按名称传给原函数，并将返回值转换成工具结果文本。"""
        # model_validate 保留嵌套的 Pydantic 模型等 Python 对象传给原始函数。
        arguments = self.parameter_model.model_validate(parameters)
        # 从模型对象取字段值，避免 model_dump 把嵌套模型也变成字典后改变函数入参类型。
        result = self.func(**{name: getattr(arguments, name) for name in type(arguments).model_fields})
        if inspect.isawaitable(result):
            # 防止工具意外返回尚未执行的异步对象；注册表约定这里应已得到实际结果。
            if inspect.iscoroutine(result):
                result.close()
            raise ToolError("工具返回了 awaitable，请注册同步函数")
        return str(result)
