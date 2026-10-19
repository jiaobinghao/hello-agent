"""五种 Agent 共用 Agent 基类，但用各自的 run() 实现不同求解策略。"""

from .function_call_agent import FunctionCallAgent
from .plan_solve_agent import PlanAndSolveAgent
from .react_agent import ReActAgent
from .reflection_agent import ReflectionAgent
from .simple_agent import SimpleAgent

__all__ = ["SimpleAgent", "ReActAgent", "ReflectionAgent", "PlanAndSolveAgent", "FunctionCallAgent"]
