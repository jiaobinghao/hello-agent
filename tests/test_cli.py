"""命令行测试：直接调用 main 并捕获输出，不启动真实模型服务。"""

import contextlib
import io
import os
import unittest
from unittest.mock import patch

from hello_agents.cli import main


class CLITests(unittest.TestCase):
    def test_demo_runs_all_paradigms_without_credentials(self):
        """清空环境凭证后，离线演示仍应完成全部 Agent 和工具组合。"""
        output = io.StringIO()
        with patch.dict(os.environ, {}, clear=True), contextlib.redirect_stdout(output):
            self.assertEqual(main(["--demo"]), 0)
        text = output.getvalue()
        for label in ["基础对话", "ReAct", "Reflection", "Plan-and-Solve", "FunctionCall", "工具链", "并发工具"]:
            self.assertIn(label, text)
        self.assertIn("calculator → 152", text)

    def test_missing_model_is_actionable_error(self):
        """未配置模型时返回失败退出码，并提示具体需要配置的变量名。"""
        error = io.StringIO()
        with patch.dict(os.environ, {}, clear=True), contextlib.redirect_stderr(error):
            self.assertEqual(main(["--env-file", "/tmp/nonexistent-hello-agent-test.env", "你好"]), 1)
        self.assertIn("LLM_MODEL_ID", error.getvalue())

    def test_invalid_stream_combination_is_rejected(self):
        """不支持的策略与流式选项组合，由命令行解析阶段直接拒绝。"""
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as context:
            main(["--agent", "react", "--stream", "你好"])
        self.assertEqual(context.exception.code, 2)
