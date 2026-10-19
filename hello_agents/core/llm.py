"""模型通信层：把统一的消息列表发给在线或本地模型服务。

阅读主线：__init__ 解析连接配置 → _request 发送请求 → complete 取出回复。
上层有三种取结果的方式：invoke 返回完整文本，stream 逐段返回文本，
complete 保留包含 tool_calls 的消息对象，供原生工具调用使用。
"""

import os
from collections.abc import Iterator, Sequence
from typing import Any, Protocol
from urllib.parse import urlparse

from openai import OpenAI, OpenAIError

from .config import Config
from .exceptions import ConfigurationError, LLMError
from .message import Message


class ChatModel(Protocol):
    """约定模型应提供的方法，Agent 按这个约定使用模型。

    Protocol 只描述“能做什么”：其他类实现相同方法即可作为模型传入，
    不必继承 HelloAgentsLLM；因此离线脚本也能驱动同一套 Agent 循环。
    下方 ... 表示这里只声明方法，由具体模型类实现。
    """

    # 普通对话：发出一次请求，拿到一段完整的回答。
    def invoke(self, messages: list[dict[str, Any]], **kwargs: Any) -> str: ...

    # 流式对话：返回迭代器，调用方每次取到新生成的一段文字。
    def stream(self, messages: list[dict[str, Any]], **kwargs: Any) -> Iterator[str]: ...

    # 原生工具调用：返回消息对象，除文字外还可包含工具名、参数和调用 ID。
    def complete(self, messages: list[dict[str, Any]], **kwargs: Any) -> Any: ...


# 映射关系：服务名 →（专用密钥环境变量名，默认 API 根地址）。
# local/custom 没有预设地址，需调用方配置；具体模型 ID 始终由用户指定。
PROVIDERS = {
    "openai": ("OPENAI_API_KEY", "https://api.openai.com/v1"),
    "deepseek": ("DEEPSEEK_API_KEY", "https://api.deepseek.com"),
    "modelscope": ("MODELSCOPE_API_KEY", "https://api-inference.modelscope.cn/v1"),
    "zhipu": ("ZHIPU_API_KEY", "https://open.bigmodel.cn/api/paas/v4"),
    "ollama": (None, "http://localhost:11434/v1"),
    "vllm": (None, "http://localhost:8000/v1"),
    "local": (None, None),
    "custom": (None, None),
}

#  Python 的 Protocol 支持结构化类型检查：
#  只要一个类提供了约定的方法，并且参数、返回值类型兼容，就可以满足这个协议，不必显式继承它。
class HelloAgentsLLM:
    """用同一套接口访问兼容 Chat Completions 的服务，连接细节集中在这里。"""

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        provider: str | None = None,
        config: Config | None = None,
        client: Any = None,
        **kwargs: Any,
    ):
        """建立客户端；此时不发送模型请求。

        model 是服务上的模型 ID，base_url 是服务根地址，api_key 是凭证。
        provider 选择服务默认配置；client 可注入外部客户端，便于离线测试。
        **kwargs 接收额外关键字，这里仅允许 temperature/max_tokens/timeout。
        """
        # 1. 先取得配置副本，再用显式参数覆盖；重新构造 Config 可校验覆盖值。
        self.config = (config or Config.from_env()).model_copy(deep=True)
        allowed = {"temperature", "max_tokens", "timeout"}
        if kwargs.keys() - allowed:
            raise ConfigurationError(f"未知模型配置：{sorted(kwargs.keys() - allowed)}")
        self.config = Config(**(self.config.to_dict() | kwargs))

        # 2. 连接地址优先取函数参数，其次环境变量；服务名可显式指定或自动推断。
        actual_url = base_url or os.getenv("LLM_BASE_URL")
        self.provider = provider or self.config.default_provider
        if self.provider == "auto":
            self.provider = self._auto_detect_provider(actual_url)
        if self.provider not in PROVIDERS:
            raise ConfigurationError(f"不支持的 provider：{self.provider}")
        key_env, default_url = PROVIDERS[self.provider]

        # 3. 选定服务后补全默认地址，解析 URL 以检查协议和主机名。
        self.base_url = actual_url or default_url
        # 将url拆分为scheme:https hostname:api.deepseek.com path:/v1
        parsed = urlparse(self.base_url or "")
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ConfigurationError("请设置有效的 LLM_BASE_URL（包含 http:// 或 https://）")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ConfigurationError("base_url 不应包含用户名、密码、查询参数或片段")
        self.model = model or self.config.default_model
        if not self.model or not self.model.strip():
            raise ConfigurationError("请通过 model 参数或 LLM_MODEL_ID 设置模型 ID")

        # 4. 密钥优先级：显式 api_key → 通用 LLM_API_KEY → 服务专用变量。
        resolved_key = api_key or os.getenv("LLM_API_KEY")
        if not resolved_key and key_env:
            # provider对应的key_env环境变量字符串，再通过这个环境变量getenv到对应的api_key
            resolved_key = os.getenv(key_env)
        if not resolved_key and self.provider in {"ollama", "vllm", "local"}:
            # SDK 要求非空密钥，未配置鉴权的本地服务可用占位值。
            resolved_key = "local"
        if not resolved_key and client is None:
            raise ConfigurationError(f"请设置 LLM_API_KEY 或 {key_env or 'api_key 参数'}")
        self.temperature = self.config.temperature
        self.max_tokens = self.config.max_tokens
        self.timeout = self.config.timeout

        # 5. 记录客户端的归属；只有本类创建的连接才由本类关闭。
        self._owns_client = client is None # 如果当前client没传，那就是True，说明是本类创建的客户端，需要自己处理
        # 归根结底还是openai创建client，client是openai的OpenAI类实例，里面有chat属性，chat属性里面有completions属性，completions属性里面有create方法
        self._client = client if client is not None else OpenAI(
            api_key=resolved_key,
            base_url=self.base_url,
            timeout=self.timeout,
            max_retries=0,  # 不隐式重试，避免一次 Agent 步骤背后出现额外模型请求。
        )

    # 当provider为auto时，自动检测服务提供方
    @staticmethod
    def _auto_detect_provider(base_url: str | None) -> str:
        """有 URL 时按主机/端口判断；没有 URL 时按已配置的服务密钥判断。"""
        if base_url:
            parsed = urlparse(base_url)
            host = parsed.hostname or ""
            for name, (_, url) in PROVIDERS.items():
                if url and host == urlparse(url).hostname and name not in {"ollama", "vllm"}:
                    return name
            if host in {"localhost", "127.0.0.1", "::1"}:
                # 相同本地主机可运行不同服务，这里用常见端口区分。
                try:
                    return {11434: "ollama", 8000: "vllm"}.get(parsed.port, "local")
                except ValueError as exc:
                    raise ConfigurationError("base_url 端口无效") from exc
            return "custom"  # 未识别的 URL 保留为兼容服务，不强行猜测提供方。
        for name, (key_env, _) in PROVIDERS.items():
            if key_env and os.getenv(key_env):
                return name
        return "openai"

    def _request(self, messages: Sequence[Message | dict[str, Any]], stream: bool, **kwargs: Any):
        """所有请求共用的出口：合并生成参数、转换消息、调用 SDK。"""
        # & 求集合交集：调用参数不允许占用这些由本方法管理的关键字。
        if {"model", "messages", "stream"} & kwargs.keys():
            raise ConfigurationError("model/messages/stream 由模型接口管理，不能在调用参数中覆盖")
        options: dict[str, Any] = {}
        if self.temperature is not None:
            options["temperature"] = self.temperature
        if self.max_tokens is not None:
            options["max_tokens"] = self.max_tokens
        options.update(kwargs)
        # 此行才真正发起模型请求。**options 将参数字典展开为关键字参数。
        # 兼容 Message 对象与普通字典，便于内部协议消息直接送回模型。
        return self._client.chat.completions.create(
            model=self.model,
            messages=[m.to_dict() if isinstance(m, Message) else m for m in messages],
            stream=stream,
            **options,
        )

    def complete(self, messages: Sequence[Message | dict[str, Any]], **kwargs: Any) -> Any:
        """发送一次非流式请求，返回 SDK 的 assistant 消息，保留工具调用信息。"""
        try:
            response = self._request(messages, stream=False, **kwargs)
            if not response.choices:
                raise LLMError("模型未返回 choices")
            choice = response.choices[0]  # choices 是候选回复列表，本框架使用第一项。
            # length 表示超出生成长度，content_filter 表示被过滤，均不能当作完整回复。
            if choice.finish_reason in {"length", "content_filter"}:
                raise LLMError(f"模型响应未完整生成：{choice.finish_reason}")
            return choice.message
        except OpenAIError as exc:
            # 向上统一抛出 LLMError；from exc 保留原始异常，调试时仍可追溯原因。
            raise LLMError(f"模型请求失败：{exc}") from exc

    def invoke(self, messages: Sequence[Message | dict[str, Any]], **kwargs: Any) -> str:
        """普通文本接口：从完整消息中取出 content，确保调用方得到非空文字。"""
        message = self.complete(messages, **kwargs)
        if message.tool_calls:
            # 原生调用还要执行工具、反馈结果，不能当成普通文本提前结束。
            raise LLMError("收到原生工具调用，请使用 FunctionCallAgent 或 complete()")
        content = message.content
        if not content or not content.strip():
            raise LLMError("模型未返回文本内容")
        return content

    def stream(self, messages: Sequence[Message | dict[str, Any]], **kwargs: Any) -> Iterator[str]:
        """流式文本接口：每收到一段增量文本就 yield 给调用方，而不自行打印。"""
        response = None
        has_content = False
        try:
            response = self._request(messages, stream=True, **kwargs)
            for chunk in response:
                if not chunk.choices:
                    continue  # SSE 是流式事件；有的事件只携带用量统计，没有候选回复。
                choice = chunk.choices[0]
                if choice.finish_reason in {"length", "content_filter"}:
                    raise LLMError(f"模型响应未完整生成：{choice.finish_reason}")
                if choice.delta.tool_calls:
                    raise LLMError("stream() 仅支持文本；原生工具调用使用 complete()")
                if choice.delta.content:
                    # delta 仅包含这次新增的部分；调用方按顺序拼接即可得到完整回答。
                    has_content = True
                    yield choice.delta.content
            if not has_content:
                raise LLMError("模型未返回文本内容")
        except OpenAIError as exc:
            raise LLMError(f"模型流式请求失败：{exc}") from exc
        finally:
            # 正常读完、发生异常、调用方提前关闭生成器，都需要释放响应连接。
            if response is not None:
                response.close()

    def think(self, messages: Sequence[Message | dict[str, Any]], **kwargs: Any) -> Iterator[str]:
        """教程中的流式调用名称，行为与 stream 相同，并非额外启动一轮推理。"""
        return self.stream(messages, **kwargs)

    def close(self) -> None:
        """关闭本类创建的客户端；外部注入的客户端仍由调用方负责。"""
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> "HelloAgentsLLM":
        """进入 with HelloAgentsLLM() as llm 时，把当前对象交给 llm。"""
        return self

    def __exit__(self, *_: Any) -> None:
        """退出 with 块时自动清理连接，块内发生异常也会执行。"""
        self.close()
