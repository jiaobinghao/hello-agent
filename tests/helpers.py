"""用真实 SDK + 内存 HTTP transport 验证请求与响应，不访问网络。"""

import json
from copy import deepcopy

try:
    import httpx2 as httpx  # OpenAI SDK 3.x
except ImportError:
    import httpx  # OpenAI SDK 1.x/2.x
from openai import OpenAI

from hello_agents import Config, HelloAgentsLLM


def make_llm(test_case, replies, **kwargs):
    """返回一个走真实 SDK 的模型和请求记录列表；底层 HTTP 由内存函数替代。

    replies 按请求顺序提供回复，支持普通文字、原生消息、完整响应字典，
    或 (HTTP状态码, 错误响应体)，用来测试成功与失败场景。
    """
    requests = []
    pending = list(deepcopy(replies))

    def handler(request):
        """拦截 SDK 发出的请求，记录消息后返回下一个预设的 HTTP 响应。"""
        requests.append(json.loads(request.content))
        if not pending:
            raise AssertionError("出现未预期的额外模型请求")
        reply = pending.pop(0)
        if isinstance(reply, tuple):
            # 例如 (401, {...}) 表示模拟服务拒绝凭证的情况。
            return httpx.Response(reply[0], json=reply[1])
        if isinstance(reply, str):
            reply = {"role": "assistant", "content": reply}
        if "choices" not in reply:
            # 把简短的 assistant 消息补齐成 SDK 期望的完整 Chat Completions 响应。
            reply = {
                "id": "test-completion", "object": "chat.completion", "created": 0,
                "model": "test-model", "choices": [{"index": 0, "message": reply,
                "finish_reason": "tool_calls" if reply.get("tool_calls") else "stop"}],
            }
        return httpx.Response(200, json=reply)

    # MockTransport 在进程内调用 handler，不建立真实网络连接，也不使用环境代理。
    http_client = httpx.Client(transport=httpx.MockTransport(handler), trust_env=False)
    client = OpenAI(api_key="offline-test-key", base_url="https://example.test/v1", http_client=http_client, max_retries=0)
    test_case.addCleanup(client.close)  # 无论测试成功失败，结束后都关闭客户端。
    return HelloAgentsLLM(model="test-model", provider="custom", base_url="https://example.test/v1", client=client, config=Config(), **kwargs), requests


def native_call(*calls, content=None, **extra):
    """把多组 (调用ID, 工具名, JSON参数文本) 转成原生工具调用回复。"""
    return {
        "role": "assistant", "content": content, "tool_calls": [
            {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}
            for call_id, name, arguments in calls
        ], **extra,
    }
