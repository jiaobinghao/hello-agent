"""通过预设模型回复观察各类 Agent 的决策循环；工具运行真实代码。"""

import unittest

from hello_agents import (
    AgentParseError, CalculatorTool, Config, FunctionCallAgent, MaxStepsExceeded,
    PlanAndSolveAgent, ReActAgent, ReflectionAgent, SimpleAgent, ToolRegistry,
)
from hello_agents.testing import ScriptedLLM
from tests.helpers import make_llm, native_call


class AgentTests(unittest.TestCase):
    def setUp(self):
        self.registry = ToolRegistry()
        self.registry.register_tool(CalculatorTool())

    def test_simple_executes_real_tool_and_supplies_observation(self):
        """模型要调用计算器时，程序真实计算，并把 152 送回下一次模型请求。"""
        llm = ScriptedLLM(['[TOOL_CALL:calculator:{"expression":"15*8+32"}]', "152"])
        agent = SimpleAgent("a", llm, tool_registry=self.registry)
        self.assertEqual(agent.run("计算"), "152")
        self.assertIn("calculator: 152", llm.requests[1]["messages"][-1]["content"])
        self.assertEqual(agent.get_trace()[0]["observation"], "152")
        self.assertEqual(len(agent.get_history()), 2)

    def test_simple_handles_nested_arrays_brackets_in_strings_and_multiple_calls(self):
        """工具标记能包含数组、文字中的右括号，一条回复还可请求多个工具。"""
        def join(parts: list[str]) -> str:
            return "|".join(parts)

        self.registry.register_function("join", "拼接", join)
        llm = ScriptedLLM([
            '[TOOL_CALL:join:{"parts":["a]b","c:d"]}]\n[TOOL_CALL:calculator:2+2]', "完成",
        ])
        agent = SimpleAgent("a", llm, tool_registry=self.registry)
        agent.run("拼接并计算")
        self.assertEqual([s["observation"] for s in agent.get_trace()], ["a]b|c:d", "4"])

    def test_simple_tool_error_is_feedback_for_correction(self):
        """除零错误回传模型，模型修改表达式后仍可完成本次任务。"""
        llm = ScriptedLLM(["[TOOL_CALL:calculator:1/0]", "[TOOL_CALL:calculator:4/2]", "2"])
        agent = SimpleAgent("a", llm, tool_registry=self.registry)
        self.assertEqual(agent.run("计算"), "2")
        self.assertIn("工具错误", llm.requests[1]["messages"][-1]["content"])

    def test_malformed_simple_marker_does_not_execute_partial_calls(self):
        """后面的标记未闭合时，前面的工具也不能先执行，避免产生半轮副作用。"""
        llm = ScriptedLLM(["[TOOL_CALL:calculator:2+2] [TOOL_CALL:calculator:3+3"])
        agent = SimpleAgent("a", llm, tool_registry=self.registry)
        with self.assertRaises(AgentParseError):
            agent.run("计算")
        self.assertEqual(agent.get_trace(), [])
        self.assertEqual(agent.get_history(), [])

    def test_simple_limit_counts_model_requests_without_extra_fallback(self):
        """到达调用次数上限就报错，不额外问一次模型，也不把未完成任务记入历史。"""
        llm = ScriptedLLM(["[TOOL_CALL:calculator:2+2]"] * 3)
        agent = SimpleAgent("a", llm, tool_registry=self.registry, config=Config(max_steps=2))
        with self.assertRaises(MaxStepsExceeded):
            agent.run("一直计算")
        self.assertEqual(len(llm.requests), 2)
        self.assertEqual(agent.get_history(), [])

    def test_simple_tool_round_limit(self):
        """工具轮数上限为 0 时，即使模型请求调用工具，也不能执行它。"""
        llm = ScriptedLLM(["[TOOL_CALL:calculator:2+2]"])
        agent = SimpleAgent("a", llm, tool_registry=self.registry)
        with self.assertRaises(MaxStepsExceeded):
            agent.run("计算", max_tool_iterations=0)
        self.assertEqual(agent.get_trace(), [])

    def test_plain_stream_emits_chunks_and_only_commits_on_completion(self):
        """普通流式逐片段返回；提前关闭流时不保存半截回答。"""
        agent = SimpleAgent("a", ScriptedLLM(["你好"]), enable_tool_calling=False)
        stream = agent.stream_run("你好")
        self.assertEqual(next(stream), "你")
        stream.close()
        self.assertEqual(agent.get_history(), [])
        agent = SimpleAgent("b", ScriptedLLM(["你好"]))
        self.assertEqual(list(agent.stream_run("你好")), ["你", "好"])
        self.assertEqual(agent.get_history()[-1].content, "你好")

    def test_stream_with_tools_hides_markers_and_records_result(self):
        """流式工具模式只向调用方输出最终答案，工具标记留在内部处理。"""
        llm = ScriptedLLM(["[TOOL_CALL:calculator:2+2]", "结果是4"])
        agent = SimpleAgent("a", llm, tool_registry=self.registry)
        self.assertEqual(list(agent.stream_run("计算")), ["结果是4"])
        self.assertEqual(agent.get_trace()[0]["observation"], "4")

    def test_react_retries_format_and_unknown_tool_with_observations(self):
        """ReAct 可从格式错误、未知工具逐步修正，直到真实执行并返回 Finish。"""
        llm = ScriptedLLM([
            "我会计算", "Thought: 调用。\nAction: missing[2+2]",
            "Thought: 换用计算器。\nAction: calculator[2+2]",
            "Thought: 完成。\nAction: Finish[答案是4]",
        ])
        agent = ReActAgent("a", llm, self.registry)
        self.assertEqual(agent.run("计算"), "答案是4")
        self.assertIn("格式错误", llm.requests[1]["messages"][-1]["content"])
        self.assertIn("未找到工具", llm.requests[2]["messages"][-1]["content"])
        self.assertIn("Observation: 4", llm.requests[3]["messages"][-1]["content"])

    def test_react_rejects_multiple_actions_without_side_effects(self):
        """一轮出现两个 Action 时应拒绝整轮，不能先执行其中一个。"""
        llm = ScriptedLLM(["Action: calculator[2+2]\nAction: Finish[4]"])
        agent = ReActAgent("a", llm, self.registry, max_steps=1)
        with self.assertRaises(MaxStepsExceeded):
            agent.run("计算")
        self.assertNotIn("tool", agent.get_trace()[0])
        self.assertEqual(agent.get_history(), [])

    def test_reflection_refines_then_stops_and_supports_custom_prompts(self):
        """反思先审查再修订，拿到停止标记后结束，同时验证自定义模板。"""
        llm = ScriptedLLM(["初稿", "补充例子", "改进稿", "无需改进。"])
        agent = ReflectionAgent("a", llm, custom_prompts={"initial": "定制任务:{task}"})
        self.assertEqual(agent.run("解释工具"), "改进稿")
        self.assertEqual(llm.requests[0]["messages"][-1]["content"], "定制任务:解释工具")
        self.assertIn("补充例子", llm.requests[2]["messages"][-1]["content"])
        self.assertEqual([s["phase"] for s in agent.get_trace()], ["initial", "reflect", "refine", "reflect"])

    def test_reflection_stop_is_not_substring_match(self):
        """“并非无需改进”包含停止词但含义相反，不能提前结束修订流程。"""
        llm = ScriptedLLM(["初稿", "并非无需改进，请补充数据", "新答案"])
        self.assertEqual(ReflectionAgent("a", llm, max_iterations=1).run("任务"), "新答案")

    def test_plan_parses_fenced_python_list_and_propagates_results(self):
        """解析代码围栏中的计划，第二个步骤必须能看到第一个步骤的结果。"""
        llm = ScriptedLLM(["```python\n['算周二', '汇总三天']\n```", "30", "70"])
        agent = PlanAndSolveAgent("a", llm)
        self.assertEqual(agent.run("苹果销量"), "70")
        self.assertEqual(agent.plan, ["算周二", "汇总三天"])
        self.assertIn("结果：30", llm.requests[2]["messages"][-1]["content"])
        self.assertEqual(agent.get_history()[-1].content, "70")

    def test_plan_rejects_invalid_empty_code_and_excessive_steps(self):
        """空计划、代码和过长计划必须在执行第一步之前报错。"""
        for response in ["[]", '[" "]', '["一步", 2]', "{'a': 1}", "__import__('os').getcwd()", '["一","二","三"]']:
            with self.subTest(response=response):
                llm = ScriptedLLM([response])
                agent = PlanAndSolveAgent("a", llm, config=Config(max_plan_steps=2))
                with self.assertRaises(AgentParseError):
                    agent.run("任务")
                self.assertEqual(len(llm.requests), 1)
                self.assertEqual(agent.get_history(), [])

    def test_native_calls_real_sdk_multi_call_pairing_and_reasoning_preservation(self):
        """一轮原生多工具调用按 ID 配对结果，并保留兼容服务返回的附加字段。"""
        reply = native_call(
            ("c1", "calculator", '{"expression":"2+2"}'),
            ("c2", "calculator", '{"expression":"3*3"}'),
            reasoning_content="选择两个计算工具调用",
        )
        llm, requests = make_llm(self, [reply, "4 和 9"])
        agent = FunctionCallAgent("a", llm, self.registry)
        self.assertEqual(agent.run("计算两个表达式"), "4 和 9")
        messages = requests[1]["messages"]
        self.assertEqual(messages[-3]["tool_calls"], reply["tool_calls"])
        self.assertEqual(messages[-3]["reasoning_content"], reply["reasoning_content"])
        self.assertEqual([(m["tool_call_id"], m["content"]) for m in messages[-2:]], [("c1", "4"), ("c2", "9")])
        self.assertEqual(requests[0]["tools"][0]["function"]["name"], "calculator")

    def test_native_bad_json_and_wrong_parameter_shape_are_reported_to_model(self):
        """无效 JSON 或非对象参数应变成对应工具错误，模型可重新传参修正。"""
        llm, requests = make_llm(self, [
            native_call(("c1", "calculator", "broken-json"), ("c2", "calculator", "[]")),
            native_call(("c3", "calculator", '{"expression":"2+2"}')),
            "4",
        ])
        agent = FunctionCallAgent("a", llm, self.registry)
        self.assertEqual(agent.run("计算"), "4")
        self.assertTrue(all("工具错误" in m["content"] for m in requests[1]["messages"][-2:]))
        self.assertEqual(requests[2]["messages"][-1]["tool_call_id"], "c3")

    def test_native_limit_does_not_report_tool_request_as_final_answer(self):
        """原生调用达到上限仍未生成答案时应报错，不能把工具请求当答案返回。"""
        llm, requests = make_llm(self, [native_call(("c1", "calculator", '{"expression":"2+2"}'))])
        agent = FunctionCallAgent("a", llm, self.registry, max_steps=1)
        with self.assertRaises(MaxStepsExceeded):
            agent.run("计算")
        self.assertEqual(len(requests), 1)
        self.assertEqual(agent.get_history(), [])

    def test_native_without_tools_omits_empty_tools_field(self):
        """没有注册工具时省略 tools 字段，保持普通对话请求兼容。"""
        llm, requests = make_llm(self, ["你好"])
        self.assertEqual(FunctionCallAgent("a", llm).run("你好"), "你好")
        self.assertNotIn("tools", requests[0])
