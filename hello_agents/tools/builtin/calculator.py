"""对 AST 白名单求值，不执行模型提供的 Python 代码。"""

import ast
import math
import operator
from typing import Any

from ..base import Tool, ToolParameter
from ...core.exceptions import ToolError

# ast.parse 会把 2 + 3 变成“加法节点，左边是 2，右边是 3”的树。
# 这里只允许映射表中的运算，遇到其他语法就拒绝，不用 eval 执行模型文字。
OPERATORS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod, ast.Pow: operator.pow,
}
FUNCTIONS = {"sqrt": math.sqrt, "abs": abs, "round": round}
CONSTANTS = {"pi": math.pi, "e": math.e}


class CalculatorTool(Tool):
    """计算受限的实数表达式，展示怎样实现一个无网络依赖的类工具。"""

    def __init__(self):
        super().__init__("calculator", "计算数学表达式；支持 + - * / // % **、sqrt、abs、round、pi 和 e")

    def get_parameters(self) -> list[ToolParameter]:
        """声明模型只需传一个 expression 字符串，注册表据此生成参数规则。"""
        return [ToolParameter(name="expression", description="数学表达式，例如 sqrt(16) + 2 * 3")]

    def run(self, parameters: dict[str, Any]) -> str:
        """校验参数 → 解析语法树 → 限制复杂度 → 递归求值 → 返回结果文本。"""
        expression = self.validate_parameters(parameters)["expression"]
        if not expression.strip() or len(expression) > 1000:
            raise ToolError("表达式不能为空，且不能超过 1000 个字符")
        try:
            # mode="eval" 只解析单个表达式，不接收赋值、循环等语句；此处尚未执行。
            tree = ast.parse(expression, mode="eval")
            if sum(1 for _ in ast.walk(tree)) > 100:
                raise ToolError("表达式过于复杂")
            return str(self._evaluate(tree.body))
        except ToolError:
            raise
        except (SyntaxError, ValueError, TypeError, ArithmeticError, RecursionError) as exc:
            raise ToolError(f"计算失败：{exc}") from exc

    def _evaluate(self, node: ast.AST) -> int | float:
        """根据节点类型逐层计算，每次递归只处理一个受允许的语法节点。"""
        if isinstance(node, ast.Constant) and type(node.value) in {int, float}:
            # 数字字面量；用 type 精确判断，防止布尔值被当成 int 接收。
            result = node.value
        elif isinstance(node, ast.Name) and node.id in CONSTANTS:
            # 变量名只允许预先定义的 pi、e，不能任意读取程序变量。
            result = CONSTANTS[node.id]
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            # 处理 +3、-3 等只有一个操作数的运算。
            result = self._evaluate(node.operand) * (-1 if isinstance(node.op, ast.USub) else 1)
        elif isinstance(node, ast.BinOp) and type(node.op) in OPERATORS:
            # 双操作数运算先递归求左右子树，例如 2 + 3 * 4 会先算右边的 3 * 4。
            left, right = self._evaluate(node.left), self._evaluate(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 100:
                # 限制指数，避免极端幂运算消耗过多资源。
                raise ToolError("幂指数的绝对值不能超过 100")
            result = OPERATORS[type(node.op)](left, right)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in FUNCTIONS:
            # 函数调用只允许白名单中的数学函数，参数也必须递归通过同样的检查。
            if node.keywords or len(node.args) > 2:
                raise ToolError("数学函数仅接受至多两个位置参数")
            result = FUNCTIONS[node.func.id](*(self._evaluate(arg) for arg in node.args))
        else:
            # 属性访问、导入、字符串、集合等都不属于本计算器的数学语法。
            raise ToolError("表达式包含不支持的语法")
        # 每个中间结果都检查，拒绝复数、无穷大、NaN 和过大的数值。
        if type(result) not in {int, float} or not math.isfinite(result) or abs(result) > 1e100:
            raise ToolError("计算结果必须是绝对值不超过 1e100 的有限实数")
        return result
