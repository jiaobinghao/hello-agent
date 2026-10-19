"""先生成答案，再通过独立审查和修订迭代改善。"""

from typing import Any

from ..core import Agent, Config
from ..core.llm import ChatModel

# 三个阶段分别扮演写作者、审查者、修改者；每一阶段都需要一次模型请求。
DEFAULT_PROMPTS = {
    "initial": "完成下面的任务，给出可直接使用的答案：\n{task}",
    "reflect": (
        "审查答案是否准确、完整并满足任务。指出可操作的改进；"
        "如果没有需要修改的地方，只输出：无需改进。\n任务：{task}\n答案：{content}"
    ),
    "refine": "按审查意见重新完成任务。\n任务：{task}\n原答案：{last_attempt}\n意见：{feedback}",
}


class ReflectionAgent(Agent):
    """先出初稿，再让模型审查并修订；这里没有工具执行环节。"""

    def __init__(
        self, name: str, llm: ChatModel, system_prompt: str | None = None,
        config: Config | None = None, max_iterations: int = 2,
        custom_prompts: dict[str, str] | None = None,
    ):
        """max_iterations 限制审查/修订轮数，0 表示只生成初稿。"""
        super().__init__(name, llm, system_prompt, config)
        if max_iterations < 0:
            raise ValueError("max_iterations 不能为负数")
        self.max_iterations = max_iterations
        # 用户可只覆盖一个阶段，未提供的阶段继续使用默认提示词。
        self.prompts = DEFAULT_PROMPTS | (custom_prompts or {})
        if self.prompts.keys() != DEFAULT_PROMPTS.keys():
            raise ValueError("反思提示词仅支持 initial、reflect、refine")

    def run(self, input_text: str, **kwargs: Any) -> str:
        """最多请求模型 1 + 2 * max_iterations 次，不使用 Config.max_steps。"""
        # 旧对话作为共有上下文；当前任务由各阶段提示词带入，所以去掉最后一条。
        context = self._start(input_text)[:-1]

        def ask(phase: str, **values: str) -> str:
            """共享的阶段调用：填模板、询问模型、把阶段输出记入轨迹。"""
            prompt = self.prompts[phase].format(task=input_text, **values)
            response = self.llm.invoke(context + [{"role": "user", "content": prompt}], **self._options(kwargs))
            self.steps.append({"phase": phase, "response": response})
            return response

        # 1. 先产生可供审查的初稿。
        answer = ask("initial")
        for _ in range(self.max_iterations):
            # 2. 审查的是当前答案；修订后下一轮就会审查新的版本。
            feedback = ask("reflect", content=answer)
            # 必须是完整的停止标记，避免“并非无需改进”等包含关系误判。
            if feedback.strip().rstrip("。.!！") == "无需改进":
                break
            # 3. 将原稿与反馈一起交给模型，要求返回完整修订稿。
            answer = ask("refine", last_attempt=answer, content=answer, feedback=feedback)
        # 仅将最后版本存入会话历史；初稿和反馈仍可在执行轨迹中查看。
        return self._remember(input_text, answer)
