"""无密钥、无网络的五种范式与工具组合演示。"""

import asyncio

from .agents import FunctionCallAgent, PlanAndSolveAgent, ReActAgent, ReflectionAgent, SimpleAgent
from .testing import ScriptedLLM
from .tools import AsyncToolExecutor, CalculatorTool, ToolChain, ToolRegistry


def run_demo() -> None:
    """先观察各类 Agent 怎么运行，再观察串行工具链与并发工具的区别。"""
    print("离线演示：模型响应来自预设脚本，工具和 Agent 执行真实代码。")
    registry = ToolRegistry()
    registry.register_tool(CalculatorTool())
    question = "计算 15 * 8 + 32"
    # 每项是一对 (Agent 对象, 任务)。列表里的模型响应数对应这次任务需要的请求数。
    # 回复来自脚本，因此这里展示的是程序流程，而非模型的实际求解能力。
    cases = [
        # 一次请求：模型直接回答，无工具执行。
        (SimpleAgent("基础对话", ScriptedLLM(["你好，我可以帮助你学习 Agent 框架。"])), "你好"),
        # 两次请求：先发工具标记，程序执行计算器，模型再读结果给出答案。
        (SimpleAgent("文本工具", ScriptedLLM([
            '[TOOL_CALL:calculator:{"expression":"15 * 8 + 32"}]', "计算结果为 152。",
        ]), tool_registry=registry), question),
        # 两次请求：第一次选 calculator 动作，第二次用 Finish 明确结束。
        (ReActAgent("ReAct", ScriptedLLM([
            "Thought: 使用计算器。\nAction: calculator[15 * 8 + 32]",
            "Thought: 已拿到结果。\nAction: Finish[计算结果为 152。]",
        ]), registry), question),
        # 四次请求：初稿 → 审查意见 → 修订稿 → “无需改进”停止。
        (ReflectionAgent("Reflection", ScriptedLLM([
            "Agent 就是模型。", "需要补充工具、状态和行动循环。",
            "Agent 使用模型决定行动，调用工具并维护状态，通过反馈循环完成任务。", "无需改进",
        ])), "用一句话解释 Agent"),
        # 三次请求：一个含两步的计划，再分别执行第一步和第二步。
        (PlanAndSolveAgent("Plan-and-Solve", ScriptedLLM([
            '["求周二和周三的销量", "汇总三天销量并回答"]',
            "周二 30 个，周三 25 个。", "15 + 30 + 25 = 70，三天共卖出 70 个苹果。",
        ])), "周一卖15个苹果，周二是两倍，周三比周二少5个，一共卖多少？"),
        # 两次请求：结构化 tool_calls 请求计算器，拿到对应 ID 的工具结果后回答。
        (FunctionCallAgent("FunctionCall", ScriptedLLM([
            {"role": "assistant", "content": None, "tool_calls": [{
                "id": "demo_calculator", "type": "function",
                "function": {"name": "calculator", "arguments": '{"expression":"15 * 8 + 32"}'},
            }]}, "计算结果为 152。",
        ]), registry), question),
    ]
    for agent, task in cases:
        # 相同的 run() 接口，背后采用不同策略；轨迹能看到程序实际做过什么。
        print(f"\n[{agent.name}] {agent.run(task)}")
        print(f"历史 {len(agent.get_history())} 条，执行轨迹 {len(agent.get_trace())} 条")
        for step in agent.get_trace():
            if "tool" in step:
                print(f"  {step['tool']} → {step['observation']}")

    # 工具链由开发者固定顺序：第一步算 2+3，存入 subtotal；第二步引用它再乘 2。
    chain = ToolChain("两步计算").add_step("calculator", "{input}", "subtotal")
    chain.add_step("calculator", "{subtotal} * 2")
    print(f"\n[工具链] (2 + 3) * 2 = {chain.execute(registry, '2 + 3')}")

    async def parallel() -> list[str]:
        """两个表达式没有依赖，可以同时提交给执行器；退出 async with 时关闭线程池。"""
        async with AsyncToolExecutor(registry, max_workers=2) as executor:
            return await executor.execute_tools_parallel([
                {"tool_name": "calculator", "input_data": "sqrt(16)"},
                {"tool_name": "calculator", "input_data": "2 ** 8"},
            ])

    # asyncio.run 创建事件循环、运行协程到完成，再关闭事件循环。
    print(f"[并发工具] {asyncio.run(parallel())}")
