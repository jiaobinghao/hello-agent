"""工具测试：实际执行计算、参数转换、链式传递与线程并发，搜索用内存响应。"""

import asyncio
import io
import json
import os
import threading
import unittest
from unittest.mock import patch

from hello_agents import (
    AsyncToolExecutor, CalculatorTool, SearchTool, ToolChain, ToolChainManager,
    ToolError, ToolRegistry,
)


class ToolTests(unittest.TestCase):
    def setUp(self):
        self.registry = ToolRegistry()
        self.registry.register_tool(CalculatorTool())

    def test_calculator_supported_math(self):
        """白名单中的运算、函数和常量都能通过注册表正确求值。"""
        cases = {"15 * 8 + 32": "152", "sqrt(16) + abs(-3)": "7.0", "2 ** 8": "256", "round(pi, 2)": "3.14"}
        for expression, expected in cases.items():
            with self.subTest(expression=expression):
                self.assertEqual(self.registry.execute_tool("calculator", expression), expected)

    def test_calculator_rejects_code_and_unbounded_math(self):
        """代码访问、集合、除零和极端数值不能绕过受限计算器。"""
        for expression in ["", "1/0", "__import__('os').getcwd()", "(1).__class__", "[1] * 10", "True + 1", "10 ** 10000", "(-1) ** 0.5", "1e300"]:
            with self.subTest(expression=expression), self.assertRaises(ToolError):
                self.registry.execute_tool("calculator", expression)

    def test_function_registration_schema_validation_and_defaults(self):
        """函数签名生成正确的参数说明，并检查类型转换、默认值和额外字段。"""
        def multiply(value: int, factor: int = 2) -> str:
            return str(value * factor)

        self.registry.register_function("multiply", "放大整数", multiply)
        self.assertEqual(self.registry.execute_tool("multiply", '{"value":"3"}'), "6")
        self.assertEqual(self.registry.execute_tool("multiply", {"value": 4, "factor": 3}), "12")
        schema = self.registry.get_tool("multiply").to_openai_schema()["function"]["parameters"]
        self.assertEqual(schema["properties"]["value"]["type"], "integer")
        self.assertEqual(schema["required"], ["value"])
        self.assertFalse(schema["additionalProperties"])
        for arguments in [{}, {"value": "oops"}, {"value": 2, "extra": 1}, "value=3", '{"value":']:
            with self.subTest(arguments=arguments), self.assertRaises(ToolError):
                self.registry.execute_tool("multiply", arguments)

    def test_missing_duplicate_and_unregister(self):
        """未知工具和重名注册应报错，移除时应准确报告工具是否存在。"""
        with self.assertRaises(ToolError):
            self.registry.execute_tool("missing", "x")
        with self.assertRaises(ToolError):
            self.registry.register_tool(CalculatorTool())
        self.assertTrue(self.registry.unregister("calculator"))
        self.assertFalse(self.registry.unregister("calculator"))

    def test_chain_variable_flow_and_context_isolation(self):
        """前一步结果可进入后一步模板，但执行不能改动外部传入的上下文字典。"""
        chain = ToolChain("math").add_step("calculator", "{input}", "subtotal")
        chain.add_step("calculator", {"expression": "{subtotal} * {factor}"})
        context = {"factor": 3}
        manager = ToolChainManager(self.registry)
        manager.register_chain(chain)
        self.assertEqual(manager.execute_chain("math", "2 + 3", context), "15")
        self.assertEqual(context, {"factor": 3})

    def test_chain_stops_after_error(self):
        """任何步骤失败后不执行后续步骤，空链和缺少模板变量也必须报错。"""
        calls = []
        self.registry.register_function("record", "记录", lambda text: calls.append(text))
        chain = ToolChain("bad").add_step("calculator", "1 / 0", "a").add_step("record", "{a}")
        with self.assertRaises(ToolError):
            chain.execute(self.registry, "")
        self.assertEqual(calls, [])
        with self.assertRaises(ToolError):
            ToolChain("empty").execute(self.registry, "")
        with self.assertRaises(ToolError):
            ToolChain("missing").add_step("calculator", "{unknown}").execute(self.registry, "")

    def test_async_executor_is_concurrent_preserves_order_and_handles_errors(self):
        """线程屏障验证真正并发，结果保持输入顺序，还可按项收集异常。"""
        barrier = threading.Barrier(2, timeout=3)

        def rendezvous(value: int) -> str:
            barrier.wait()
            return str(value)

        self.registry.register_function("rendezvous", "并发测试", rendezvous)

        async def run():
            async with AsyncToolExecutor(self.registry, max_workers=2) as executor:
                results = await executor.execute_tools_parallel([
                    {"tool_name": "rendezvous", "input_data": {"value": i}} for i in [2, 1]
                ])
                failures = await executor.execute_tools_parallel([
                    {"tool_name": "calculator", "input_data": "2+2"},
                    {"tool_name": "missing", "input_data": "x"},
                ], return_exceptions=True)
            self.assertTrue(executor._closed)
            return results, failures

        results, failures = asyncio.run(run())
        self.assertEqual(results, ["2", "1"])
        self.assertEqual(failures[0], "4")
        self.assertIsInstance(failures[1], ToolError)

    def test_search_missing_key_and_http_adapter(self):
        """搜索缺密钥应提示，提供密钥后以模拟 HTTP 校验请求与结果转换。"""
        with patch.dict(os.environ, {}, clear=True), self.assertRaises(ToolError):
            SearchTool().run({"query": "test"})
        captured = []

        def open_search(request, timeout):
            captured.append((request, timeout))
            return io.BytesIO(json.dumps({"results": [{"title": "文档", "url": "https://example.test", "content": "摘要"}]}).encode())

        with patch("hello_agents.tools.builtin.search.urlopen", side_effect=open_search):
            result = SearchTool(api_key="offline-key").run({"query": "Agent框架"})
        self.assertEqual(json.loads(result)[0]["title"], "文档")
        self.assertEqual(captured[0][0].full_url, "https://api.tavily.com/search")
        self.assertEqual(json.loads(captured[0][0].data)["query"], "Agent框架")
