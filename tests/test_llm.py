"""模型协议测试：真实 SDK 解析内存 HTTP 响应，验证本框架的请求与错误处理。"""

import json
import os
import unittest
from unittest.mock import MagicMock, patch

from openai import OpenAI

from hello_agents import Config, ConfigurationError, HelloAgentsLLM, LLMError, Message
from tests.helpers import httpx, make_llm


class LLMTests(unittest.TestCase):
    def test_sdk_request_options_message_conversion_and_text(self):
        """模型参数和 Message 正确转换成请求，完整文字从 SDK 消息中取出。"""
        llm, requests = make_llm(self, ["你好"], temperature=0, max_tokens=200)
        self.assertEqual(llm.invoke([Message("你好", "user")]), "你好")
        self.assertEqual(requests[0]["messages"], [{"role": "user", "content": "你好"}])
        self.assertEqual(requests[0]["temperature"], 0)
        self.assertEqual(requests[0]["max_tokens"], 200)
        self.assertFalse(requests[0]["stream"])

    def test_api_failure_is_wrapped_without_retries(self):
        """HTTP 401 统一转为 LLMError，并且只发送一次请求。"""
        llm, requests = make_llm(self, [(401, {"error": {"message": "Invalid key", "type": "authentication_error"}})])
        with self.assertRaises(LLMError):
            llm.invoke([{"role": "user", "content": "你好"}])
        self.assertEqual(len(requests), 1)

    def test_empty_and_truncated_responses_are_errors(self):
        """无候选、生成被截断、无文本都不能伪装成成功回答。"""
        replies = [
            {"choices": []},
            {"choices": [{"index": 0, "finish_reason": "length", "message": {"role": "assistant", "content": "半句话"}}]},
            {"role": "assistant", "content": None},
        ]
        for reply in replies:
            with self.subTest(reply=reply):
                llm, _ = make_llm(self, [reply])
                with self.assertRaises(LLMError):
                    llm.invoke([{"role": "user", "content": "任务"}])

    def test_provider_url_precedence_and_local_key(self):
        """本地 URL 不受其他服务密钥干扰，显式传入的地址优先于环境配置。"""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "other-provider-key", "LLM_BASE_URL": "http://localhost:11434/v1"}, clear=True):
            with HelloAgentsLLM(model="local-model") as llm:
                self.assertEqual(llm.provider, "ollama")
                self.assertEqual(llm._client.api_key, "local")
            llm = HelloAgentsLLM(model="custom", base_url="https://example.test/v1", api_key="explicit", client=MagicMock())
            self.assertEqual(llm.provider, "custom")
            self.assertEqual(llm.base_url, "https://example.test/v1")

    def test_provider_specific_env_and_explicit_overrides(self):
        """可从专用密钥识别服务，显式 provider 和 model 可以覆盖环境默认值。"""
        env = {"DEEPSEEK_API_KEY": "offline-key", "LLM_MODEL_ID": "env-model"}
        with patch.dict(os.environ, env, clear=True):
            with HelloAgentsLLM() as llm:
                self.assertEqual(llm.provider, "deepseek")
                self.assertEqual(llm.model, "env-model")
            with HelloAgentsLLM(model="override", provider="vllm", base_url="http://127.0.0.1:8123/v1") as llm:
                self.assertEqual(llm.provider, "vllm")
                self.assertEqual(llm.model, "override")

    def test_invalid_configuration_fails_before_network(self):
        """缺少模型、服务名无效或地址不合规时，应在发送请求之前失败。"""
        cases = [
            {}, {"model": "m"}, {"model": "m", "provider": "unsupported"},
            {"model": "m", "base_url": "not-a-url"},
            {"model": "m", "base_url": "https://user:pass@example.test/v1"},
        ]
        for kwargs in cases:
            with self.subTest(kwargs=kwargs), patch.dict(os.environ, {}, clear=True):
                with self.assertRaises(ConfigurationError):
                    HelloAgentsLLM(**kwargs)

    def test_injected_client_lifetime_belongs_to_caller(self):
        """外部注入的客户端由外部管理，退出模型的 with 块不能擅自关闭它。"""
        client = MagicMock()
        with HelloAgentsLLM(model="m", client=client):
            pass
        client.close.assert_not_called()

    def test_sse_stream_with_usage_event_and_early_close(self):
        """SSE 跳过纯统计事件、拼接文本片段，并在提前关闭生成器时释放响应。"""
        requests = []
        chunks = [
            {"choices": []},
            {"choices": [{"index": 0, "delta": {"content": "你"}, "finish_reason": None}]},
            {"choices": [{"index": 0, "delta": {"content": "好"}, "finish_reason": None}]},
            {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        ]
        content = "".join("data: " + json.dumps(chunk) + "\n\n" for chunk in chunks) + "data: [DONE]\n\n"
        responses = []

        def handler(request):
            requests.append(json.loads(request.content))
            response = httpx.Response(200, content=content.encode(), headers={"content-type": "text/event-stream"})
            responses.append(response)
            return response

        client = OpenAI(api_key="offline", base_url="https://example.test/v1", max_retries=0,
                        http_client=httpx.Client(transport=httpx.MockTransport(handler), trust_env=False))
        self.addCleanup(client.close)
        llm = HelloAgentsLLM(model="test", client=client, config=Config(temperature=None))
        self.assertEqual(list(llm.think([Message("问候", "user")])), ["你", "好"])
        self.assertTrue(requests[0]["stream"])
        self.assertNotIn("temperature", requests[0])
        stream = llm.stream([Message("问候", "user")])
        self.assertEqual(next(stream), "你")
        stream.close()
        self.assertTrue(responses[-1].is_closed)
