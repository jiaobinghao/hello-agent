"""文本工具协议的解析，支持嵌套 JSON 数组以及字符串中的右方括号。"""

import re
from dataclasses import dataclass

from ..core.exceptions import AgentParseError


@dataclass(frozen=True)
class TextToolCall:
    """解析出的一次行动；这里只保存名称与原始参数文本，还没有执行工具。

    dataclass 自动生成构造方法；frozen=True 防止解析后被意外修改。
    """

    name: str
    arguments: str


def bracket_end(text: str, start: int) -> int:
    """从左括号之后开始扫描，返回与它匹配的右括号下标。

    不能直接找第一个 ]：参数可能包含数组 [1, 2]，也可能包含字符串 "a]b"。
    因此同时跟踪方括号嵌套层数、JSON 双引号状态和转义字符。
    """
    # 调用时已经越过了外层左括号，所以从深度 1 开始。
    depth = 1
    quoted = escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if quoted:
            # JSON 字符串内的括号只是文字；转义的引号也不能结束字符串。
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
        elif char == '"':
            quoted = True
        elif char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                # 内层方括号已全部闭合，这个 ] 才对应最外层的工具调用标记。
                return index
    raise AgentParseError("工具调用缺少闭合的 ]，或 JSON 字符串没有闭合")


def parse_tool_markers(text: str) -> list[TextToolCall]:
    """解析 SimpleAgent 的全部 [TOOL_CALL:名称:参数]，返回调用列表。

    先解析完整段回复再交给 Agent 执行，避免前一个工具已执行、后一个标记才报错。
    """
    calls = []
    position = 0
    # := 同时赋值和判断；find 没找到会返回 -1，找到则得到标记的起始下标。
    while (start := text.find("[TOOL_CALL:", position)) != -1:
        # 正则只识别工具名称部分，复杂参数交给 bracket_end 处理。
        header = re.match(r"\[TOOL_CALL:\s*([A-Za-z_][A-Za-z0-9_-]*)\s*:", text[start:])
        if header is None:
            raise AgentParseError("工具调用格式应为 [TOOL_CALL:工具名:参数]")
        argument_start = start + header.end()  # 把局部匹配位置换成整段文本中的位置。
        end = bracket_end(text, argument_start)
        calls.append(TextToolCall(header.group(1), text[argument_start:end].strip()))
        position = end + 1  # 从本次调用的结束位置后继续找，支持一轮多个工具。
    return calls


def parse_react_action(text: str) -> TextToolCall:
    """解析 ReAct 的单个 Action；Finish 也按同一格式解析，由 Agent 区分。"""
    # MULTILINE 使 ^ 匹配每行开头，只接收独立 Action 行，避免误读叙述中的例子。
    actions = list(re.finditer(r"^\s*Action:\s*([A-Za-z_][A-Za-z0-9_-]*)\[", text, re.MULTILINE))
    if len(actions) != 1:
        # 一轮发出多个动作会让执行顺序和结束信号含糊，因此要求模型重新输出。
        raise AgentParseError("每轮必须且只能输出一个 Action: 工具名[参数] 或 Action: Finish[答案]")
    action = actions[0]
    end = bracket_end(text, action.end())
    if text[end + 1:].strip():
        raise AgentParseError("Action 后不允许再输出其他内容或执行结果")
    return TextToolCall(action.group(1), text[action.end():end].strip())
