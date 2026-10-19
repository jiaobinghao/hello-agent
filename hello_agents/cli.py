"""python -m hello_agents 与 hello-agent 的共享入口。"""

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from pydantic import ValidationError

from . import (
    CalculatorTool, Config, FunctionCallAgent, HelloAgentsError, HelloAgentsLLM,
    PlanAndSolveAgent, ReActAgent, ReflectionAgent, SearchTool, SimpleAgent, ToolRegistry,
)
from .demo import run_demo


def main(argv: list[str] | None = None) -> int:
    """命令行主流程：读参数 → 准备模型/工具 → 创建 Agent → 执行任务。

    返回值是进程退出码，0 表示成功，1 表示运行错误，130 表示用户中断。
    argv=None 时读取真实命令行；测试可直接传入字符串列表模拟命令行参数。
    """
    # 1. 声明可用参数。nargs="?" 允许 task 省略，store_true 表示出现即启用的开关。
    parser = argparse.ArgumentParser(description="Hello-Agents 第七章本地框架")
    parser.add_argument("task", nargs="?", help="任务文本；省略时提示输入")
    parser.add_argument("--demo", action="store_true", help="运行完全离线的全部范式演示")
    parser.add_argument("--agent", choices=["simple", "react", "reflection", "plan", "function"], default="react")
    parser.add_argument("--env-file", type=Path, default=Path.cwd() / ".env")
    parser.add_argument("--model")
    parser.add_argument("--provider")
    parser.add_argument("--base-url")
    parser.add_argument("--max-steps", type=int, help="simple/react/function 的模型调用次数上限")
    parser.add_argument("--stream", action="store_true", help="SimpleAgent 流式输出")
    parser.add_argument("--chat", action="store_true", help="保持同一 Agent 的多轮对话，/clear 清空历史，/exit 退出")
    parser.add_argument("--no-tools", action="store_true", help="不注册计算器")
    parser.add_argument("--search", action="store_true", help="注册 Tavily 搜索工具")
    parser.add_argument("--trace", action="store_true", help="将本轮执行轨迹输出至 stderr")
    args = parser.parse_args(argv)
    # 参数组合在创建客户端之前就检查，错误时打印用法而不是发出无效请求。
    if args.stream and args.agent != "simple":
        parser.error("--stream 需要 --agent simple")
    if args.max_steps is not None and args.max_steps < 1:
        parser.error("--max-steps 必须大于 0")
    try:
        if args.demo:
            # 离线分支直接使用脚本模型，不要求先配置密钥或模型 ID。
            run_demo()
            return 0
        # 2. 把 .env 读入进程环境；override=False 保留用户已设置的环境变量。
        load_dotenv(args.env_file, override=False)
        config = Config.from_env()
        if args.max_steps is not None:
            config.max_steps = args.max_steps
        # 3. 注册希望交给 Agent 的工具；注册只是保存对象，此时不执行工具。
        registry = ToolRegistry()
        if not args.no_tools:
            registry.register_tool(CalculatorTool())
        if args.search:
            if not os.getenv("TAVILY_API_KEY"):
                parser.error("--search 需要设置 TAVILY_API_KEY")
            registry.register_tool(SearchTool())
        # 4. 创建模型客户端。with 保证对话结束或运行出错时关闭连接。
        with HelloAgentsLLM(
            model=args.model, provider=args.provider, base_url=args.base_url, config=config,
        ) as llm:
            common = {"name": args.agent, "llm": llm, "config": config}
            # **common 展开三个共享参数；只有支持工具的策略接收注册表。
            if args.agent == "simple":
                agent = SimpleAgent(**common, tool_registry=registry)
            elif args.agent == "react":
                agent = ReActAgent(**common, tool_registry=registry)
            elif args.agent == "function":
                agent = FunctionCallAgent(**common, tool_registry=registry)
            elif args.agent == "reflection":
                agent = ReflectionAgent(**common)
            else:
                agent = PlanAndSolveAgent(**common)
            task = args.task
            while True:
                # 5. 一次循环处理一轮输入；--chat 下复用同一个 Agent，历史才会保留。
                if task is None:
                    try:
                        task = input("你 > ").strip()
                    except EOFError:
                        # 输入流结束（如终端 Ctrl+D）时退出交互。
                        break
                if args.chat and task == "/exit":
                    break
                if args.chat and task == "/clear":
                    agent.clear_history()
                    task = None
                    continue
                # 6. 流式模式逐段展示，普通模式等整轮结束后一次展示最终答案。
                if args.stream:
                    for chunk in agent.stream_run(task):
                        print(chunk, end="", flush=True)
                    print()
                else:
                    print(agent.run(task))
                if args.trace:
                    # stdout 放答案，stderr 放过程，便于重定向结果而不混入调试信息。
                    print(json.dumps(agent.get_trace(), ensure_ascii=False, indent=2), file=sys.stderr)
                if not args.chat:
                    # 没有 --chat 时只做一个任务，不继续等待下一轮输入。
                    break
                task = None
        return 0
    except (HelloAgentsError, ValidationError, ValueError) as exc:
        # 把常见配置/运行错误转成可读提示；其他未预期异常仍保留 traceback 便于调试。
        print(f"错误：{exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        # 用户按 Ctrl+C 中断，使用终端惯用的退出码。
        print("\n已退出。", file=sys.stderr)
        return 130
