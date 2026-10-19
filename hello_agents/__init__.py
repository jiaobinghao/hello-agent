"""Hello-Agents 第七章的本地独立实现。"""

# 把常用类统一导出，应用可直接 from hello_agents import ReActAgent 等，
# 而不必依赖内部目录结构；具体实现仍分别位于 agents、core、tools 中。
from .agents import FunctionCallAgent, PlanAndSolveAgent, ReActAgent, ReflectionAgent, SimpleAgent
from .core import (
    Agent, AgentParseError, ChatModel, Config, ConfigurationError, HelloAgentsError,
    HelloAgentsLLM, LLMError, MaxStepsExceeded, Message, ToolError,
)
from .tools import (
    AsyncToolExecutor, CalculatorTool, FunctionTool, SearchTool, Tool, ToolChain,
    ToolChainManager, ToolParameter, ToolRegistry,
)

__version__ = "0.1.0"
# __all__ 列出本包公开的名称，from hello_agents import * 时按此列表导入。
__all__ = [
    "Agent", "Config", "Message", "ChatModel", "HelloAgentsLLM", "HelloAgentsError",
    "ConfigurationError", "LLMError", "AgentParseError", "MaxStepsExceeded", "ToolError",
    "SimpleAgent", "ReActAgent", "ReflectionAgent", "PlanAndSolveAgent", "FunctionCallAgent",
    "Tool", "ToolParameter", "FunctionTool", "ToolRegistry", "CalculatorTool", "SearchTool",
    "ToolChain", "ToolChainManager", "AsyncToolExecutor",
]
