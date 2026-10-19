"""框架异常：调用方可以区分配置、模型、协议和工具失败。"""


class HelloAgentsError(Exception):
    """所有框架异常的基类。"""


class ConfigurationError(HelloAgentsError):
    """缺失或无效的运行配置。"""


class LLMError(HelloAgentsError):
    """模型请求失败，或者响应无法使用。"""


class AgentParseError(HelloAgentsError):
    """模型输出不符合当前 Agent 的文本协议。"""


class MaxStepsExceeded(HelloAgentsError):
    """已经用完本次运行允许的模型调用次数。"""


class ToolError(HelloAgentsError):
    """工具不存在、参数错误或执行失败。"""
