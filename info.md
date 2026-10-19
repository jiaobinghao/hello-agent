# 项目上下文

这是 `/home/binghao/dev-agent/hello-agent` 下独立的 Python 教学框架，依据 Datawhale Hello-Agents 第七章从零实现，不依赖上游 `hello-agents` 包。发行名 `hello-agent-local`，导入名 `hello_agents`。Python >=3.10，使用 Pydantic v2、OpenAI SDK、python-dotenv。

- `hello_agents/core/`：统一消息、独立实例配置、异常、ChatModel Protocol 和 Agent 抽象基类；模型连接采用 Chat Completions 协议。
- `hello_agents/agents/`：Simple、ReAct、Reflection、PlanAndSolve、FunctionCall。统一入口 `run(input_text) -> str`，历史只记录成功问答，过程通过 `get_trace()` 查看。
- `hello_agents/tools/`：Tool / FunctionTool、注册表、AST 计算器、可选 Tavily 搜索、工具链、基于线程池的并发执行器。
- `hello_agents/testing.py`：ScriptedLLM，便于离线注入回复；`demo.py` 和 `cli.py` 提供演示与命令行。
- `docs/architecture.md`：章节映射、运行上限和扩展边界；`README.md`：安装、配置、运行和验证。

开发时保留清晰的分层；不将密钥写入源码、示例、文档或测试。库导入时不隐式加载 `.env`，不打印输出或执行网络请求。模型和工具失败使用框架异常；达到循环上限明确报错。不要执行模型提供的 Python 表达式或计划代码。

用户希望直接阅读中文注释学习框架。新增或修改源码时，用中文说明类/方法职责、参数含义、数据流和关键分支；Agent 执行循环按步骤解释，并说明较难的 Python 写法。测试用例也提供中文说明，避免只看英文函数名才能理解验证目的。

本地虚拟环境 `.venv/` 已创建。离线验证：`.venv/bin/python -m unittest discover -s tests -v`；端到端演示：`.venv/bin/python -m hello_agents --demo`。模型协议测试使用真实 SDK 与内存 HTTP transport，不访问在线服务。沙箱若禁止本地 socketpair 写入，需在允许 asyncio 线程唤醒的终端运行并发部分。没有真实模型或搜索服务的在线验证记录。
