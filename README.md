# Hello-Agent：从零构建 Agent 框架

根据 [Hello-Agents 第七章：构建你的智能体框架](https://datawhalechina.github.io/hello-agents/#/./chapter7/第七章%20构建你的Agent框架) 实现的独立 Python 项目。核心代码都在本目录的 `hello_agents/` 中，可直接阅读、调试和修改。需要 Python 3.10+；本地已创建 `.venv`，使用 Python 3.12 验证。

覆盖第七章的统一模型接口、消息与配置、四种基础 Agent、工具系统，以及当前章节补充的原生 `FunctionCallAgent`。具体执行行为和边界见 [架构说明](docs/architecture.md)。

## 先运行离线演示

当前工作区已装好独立环境，可直接执行：

```bash
cd /home/binghao/dev-agent/hello-agent
.venv/bin/python -m hello_agents --demo
```

演示依次运行 Simple、文本工具调用、ReAct、Reflection、Plan-and-Solve、FunctionCall、工具链和并发工具。模型回复使用 `ScriptedLLM` 预设脚本，计算器、参数验证和执行循环运行真实代码；不需要 API Key，不访问网络，也不产生模型费用。

换一台机器时先安装：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install -e .
.venv/bin/python -m hello_agents --demo
```

`requirements.txt` 固定本次验证的直接依赖版本；`pyproject.toml` 声明兼容的依赖范围和 `hello-agent` 命令。pip 会安装构建依赖 `setuptools>=68`；已有该构建依赖的离线环境可用 `--no-build-isolation` 安装。

## 连接真实模型

```bash
cp .env.example .env
```

编辑 `.env`，填写服务实际提供的模型 ID 和密钥。例如使用 DeepSeek：

```dotenv
LLM_PROVIDER=deepseek
LLM_MODEL_ID=你的模型ID
LLM_BASE_URL=https://api.deepseek.com
LLM_API_KEY=你的密钥
```

从项目目录运行：

```bash
# 教程中的 Thought / Action / Observation 循环
.venv/bin/python -m hello_agents --agent react --trace '请用计算器计算 15 * 8 + 32'

# 原生工具调用：模型必须支持 Chat Completions function calling
.venv/bin/python -m hello_agents --agent function --trace '请用工具计算 sqrt(16) + 2 ** 8'

# 保留当前进程中的多轮历史；/clear 清空，/exit 退出
.venv/bin/python -m hello_agents --agent simple --no-tools --chat

# 无工具时按文本片段实时输出
.venv/bin/python -m hello_agents --agent simple --no-tools --stream '解释 Agent 的行动循环'

.venv/bin/python -m hello_agents --agent reflection '用三句话解释 Agent 与普通聊天模型的区别'
.venv/bin/python -m hello_agents --agent plan '周一卖15个苹果，周二是两倍，周三比周二少5个，三天共卖多少？'
```

省略任务时提示输入一条任务；`--chat` 才会持续读取下一轮。`--max-steps 8` 调整 Simple/ReAct/FunctionCall 的调用上限；`--trace` 将本轮工具观察或生成阶段输出至 stderr，最终答案输出至 stdout。也可使用安装后的 `.venv/bin/hello-agent` 命令。

交互终端会在 Python 提供 `readline` 时启用行编辑，正确处理中文退格和光标移动；通过管道或重定向输入时按普通文本读取。

CLI 默认加载**当前目录**的 `.env`，可用 `--env-file /path/to/.env` 指定；已有环境变量优先。库不在导入时加载 `.env`，写 Python 脚本时需要显式调用 `load_dotenv()`。

配置项：

| 环境变量 | 含义 |
| --- | --- |
| `LLM_PROVIDER` | `auto`、`openai`、`deepseek`、`modelscope`、`zhipu`、`ollama`、`vllm`、`local`、`custom` |
| `LLM_MODEL_ID` | 必填，服务实际提供的模型 ID |
| `LLM_BASE_URL` | API 根地址，SDK 追加 `/chat/completions` |
| `LLM_API_KEY` | 通用密钥；也可使用对应服务的 `*_API_KEY` |
| `LLM_TEMPERATURE` | 默认 0.7；还兼容教程的 `TEMPERATURE` |
| `LLM_MAX_TOKENS` | 可选；还兼容 `MAX_TOKENS` |
| `LLM_TIMEOUT` | 单次请求超时秒数，默认 60 |
| `LLM_MAX_STEPS` | Simple/ReAct/FunctionCall 的最多模型调用次数，默认 5 |
| `LLM_MAX_PLAN_STEPS` | Plan-and-Solve 最多计划步数，默认 8 |
| `LLM_MAX_HISTORY_LENGTH` | 保存的历史消息数上限，默认 100，至少 2 |
| `TAVILY_API_KEY` | 可选，使用 `--search` 时需要 |

模型接口使用兼容 Chat Completions 的协议；需要 Responses API 的模型不在本次实现范围内。请求不自动重试。不同模型可能不接受 `temperature` 或 `max_tokens`，这时可在 Python 中使用 `Config(temperature=None)`，并通过运行参数指定服务需要的字段。

Ollama / vLLM 配置示例，需要自行启动推理服务：

```dotenv
LLM_PROVIDER=auto
LLM_MODEL_ID=已部署模型的ID
LLM_BASE_URL=http://localhost:8000/v1
```

`auto` 优先按 URL 检测服务，避免被其他服务的环境密钥影响。localhost 的 8000 端口识别为 vLLM，11434 为 Ollama；这些服务未提供密钥时使用 `local` 占位值。启用鉴权的服务仍需填写实际密钥。显式传入 `provider/model/base_url/api_key` 优先于对应环境设置。

## 在 Python 中使用

```python
from dotenv import load_dotenv
from hello_agents import HelloAgentsLLM, ReActAgent, ToolRegistry, CalculatorTool

load_dotenv()
registry = ToolRegistry()
registry.register_tool(CalculatorTool())

with HelloAgentsLLM() as llm:
    agent = ReActAgent("数学助手", llm, registry, max_steps=5)
    print(agent.run("计算 15 * 8 + 32"))
    print(agent.get_trace())
    print(agent.get_history())
```

普通函数也能注册为工具：

```python
def multiply(value: int, factor: int = 2) -> str:
    return str(value * factor)

registry.register_function("multiply", "将整数乘以指定倍数", multiply)
print(registry.execute_tool("multiply", {"value": 3}))  # 6
```

工具参数由类型标注生成 Pydantic 模型及 JSON Schema，校验必填项、类型和未知字段。单参数工具可传原始字符串，多参数工具必须传字典或 JSON 对象。另有 [自定义工具示例](examples/custom_tool.py)，展示长度单位换算和 FunctionCallAgent 集成。

工具链用模板引用结果，并发执行器按输入顺序返回：

```python
import asyncio
from hello_agents import ToolChain, AsyncToolExecutor

chain = ToolChain("计算后翻倍")
chain.add_step("calculator", "{input}", "subtotal")
chain.add_step("calculator", "{subtotal} * 2")
print(chain.execute(registry, "2 + 3"))  # 10

async def main():
    async with AsyncToolExecutor(registry, max_workers=2) as executor:
        return await executor.execute_tools_parallel([
            {"tool_name": "calculator", "input_data": "sqrt(16)"},
            {"tool_name": "calculator", "input_data": "2 ** 8"},
        ])

print(asyncio.run(main()))  # ['4.0', '256']
```

函数工具使用同步函数，并发由线程池负责；共享有状态的工具需由实现方保证线程安全。取消异步等待不会强制中止已经运行的同步工具，外部请求应设置自己的超时。线程池可通过 `with` / `async with` 或 `close()` 关闭。

## 目录与阅读顺序

```text
hello-agent/
├── hello_agents/
│   ├── core/            # Config、Message、LLM、Agent 抽象基类、异常
│   ├── agents/          # 五种 Agent 实现、文本协议解析
│   ├── tools/           # 工具基类、注册表、工具链、并发执行器
│   │   └── builtin/     # AST 计算器、可选 Tavily 搜索
│   ├── testing.py       # 可注入的 ScriptedLLM
│   ├── demo.py          # 离线端到端演示
│   └── cli.py           # 命令行与多轮对话入口
├── examples/            # 离线演示、自定义工具
├── tests/               # unittest，无真实服务调用
├── docs/architecture.md # 章节对应、执行流程和实现边界
├── info.md              # 项目背景与开发约定
├── .env.example         # 不含密钥的配置模板
└── pyproject.toml       # 包定义、依赖与命令入口
```

建议按 `core/message.py` → `core/config.py` → `core/llm.py` → `core/agent.py` → `tools/base.py` / `registry.py` → `agents/simple_agent.py` → `agents/react_agent.py` 阅读，再看反思、规划与原生工具调用。框架不隐式打印模型输出；应用负责展示，执行细节可通过 `get_trace()` 查看。

## 验证

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m hello_agents --demo
```

测试覆盖参数和配置校验、会话裁剪、AST 限制、文本工具标记、工具错误恢复、循环上限、反思停止条件、计划解析、工具链及真实并发。模型协议测试使用真实 OpenAI SDK 和内存 HTTP transport，验证 assistant/tool 消息配对、多工具调用、无效 JSON 参数、HTTP 错误和 SSE 流式输出。

本次 42 项离线测试全部通过。尚未向真实模型或 Tavily 发起在线请求；使用前需完成自己的模型、密钥及网络配置。若执行环境禁止本地 socketpair 写入，`asyncio` 的线程唤醒会受限，并发示例需在允许此操作的本地终端运行。

参考：[第七章源码与文档](https://github.com/datawhalechina/hello-agents/tree/main/code/chapter7)、[OpenAI function calling](https://developers.openai.com/api/docs/guides/function-calling)、[Tavily Search API](https://docs.tavily.com/documentation/api-reference/endpoint/search)。
