"""核心接口测试：环境配置、消息格式以及多轮历史怎样保留和隔离。"""

import os
import unittest
from unittest.mock import patch

from pydantic import ValidationError

from hello_agents import Config, ConfigurationError, Message, SimpleAgent
from hello_agents.testing import ScriptedLLM


class CoreTests(unittest.TestCase):
    def test_config_env_and_instance_isolation(self):
        """环境字符串能转成正确类型，修改外部配置不会串改 Agent 内部配置。"""
        with patch.dict(os.environ, {"LLM_MAX_STEPS": "3", "LLM_TEMPERATURE": "0", "DEBUG": "false"}, clear=True):
            config = Config.from_env()
        self.assertEqual(config.max_steps, 3)
        self.assertEqual(config.temperature, 0)
        self.assertFalse(config.debug)
        agent = SimpleAgent("a", ScriptedLLM(["答复"]), config=config)
        config.max_steps = 7
        self.assertEqual(agent.config.max_steps, 3)

    def test_invalid_env_is_configuration_error(self):
        """无效的步数环境变量应在启动时被拒绝，而不是等运行循环时才失败。"""
        for value in ["zero", "0", "-1"]:
            with self.subTest(value=value), patch.dict(os.environ, {"LLM_MAX_STEPS": value}, clear=True):
                with self.assertRaises(ConfigurationError):
                    Config.from_env()

    def test_message_validation_metadata_and_protocol_fields(self):
        """每条消息拥有独立元数据，角色必须有效，tool 结果必须关联调用 ID。"""
        first, second = Message("你好", "user"), Message("你好", "user")
        first.metadata["source"] = "local"
        self.assertEqual(second.metadata, {})
        self.assertEqual(first.to_dict(), {"role": "user", "content": "你好"})
        self.assertIsNotNone(first.timestamp.tzinfo)
        with self.assertRaises(ValidationError):
            Message("x", "unknown")
        with self.assertRaises(ValueError):
            Message("result", "tool").to_dict()
        self.assertEqual(Message("4", "tool", tool_call_id="call1").to_dict()["tool_call_id"], "call1")

    def test_history_is_bounded_complete_turns_and_copied(self):
        """历史裁剪保留完整问答，getter 返回副本，下一轮只使用仍保留的上下文。"""
        llm = ScriptedLLM(["A1", "A2", "A3"])
        agent = SimpleAgent("a", llm, config=Config(max_history_length=3))
        agent.run("Q1")
        agent.run("Q2")
        self.assertEqual([(m.role, m.content) for m in agent.get_history()], [("user", "Q2"), ("assistant", "A2")])
        history = agent.get_history()
        history[0].content = "changed"
        self.assertEqual(agent.get_history()[0].content, "Q2")
        agent.run("Q3")
        self.assertIn({"role": "user", "content": "Q2"}, llm.requests[-1]["messages"])
        self.assertNotIn({"role": "user", "content": "Q1"}, llm.requests[-1]["messages"])
        agent.clear_history()
        self.assertEqual(agent.get_history(), [])

    def test_empty_task_never_calls_model(self):
        """空任务在本地校验阶段就失败，不消耗模型调用。"""
        llm = ScriptedLLM([])
        with self.assertRaises(ValueError):
            SimpleAgent("a", llm).run(" ")
        self.assertEqual(llm.requests, [])
