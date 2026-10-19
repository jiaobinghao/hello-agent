"""计划用 JSON/Python 字符串列表表达，执行时逐步携带已完成结果。"""

import ast
import json
import re
from typing import Any

from ..core import Agent, AgentParseError, Config
from ..core.llm import ChatModel

# planner 负责“先做什么、后做什么”；executor 只完成计划中当前的一步。
# 最后一步必须回答原问题，这样无需额外再请求一次模型来汇总答案。
DEFAULT_PROMPTS = {
    "planner": (
        "将问题拆成按依赖顺序排列的可执行步骤，最后一步必须直接回答原问题。"
        "仅输出非空字符串组成的列表，使用 JSON 或 Python 列表格式。\n问题：{question}"
    ),
    "executor": (
        "执行当前步骤并输出这一步的结果。可参考之前的结果；"
        "若当前是最后一步，给出原问题的完整答案。\n问题：{question}\n"
        "计划：{plan}\n已完成：{history}\n当前步骤：{current_step}"
    ),
}


class PlanAndSolveAgent(Agent):
    """先拆解完整计划，再按计划逐步执行，每一步都能看到此前结果。"""

    def __init__(
        self, name: str, llm: ChatModel, system_prompt: str | None = None,
        config: Config | None = None, custom_prompts: dict[str, str] | None = None,
    ):
        """custom_prompts 可覆盖 planner 或 executor；计划长度由 config 限制。"""
        super().__init__(name, llm, system_prompt, config)
        self.prompts = DEFAULT_PROMPTS | (custom_prompts or {})
        if self.prompts.keys() != DEFAULT_PROMPTS.keys():
            raise ValueError("规划提示词仅支持 planner、executor")
        self.plan: list[str] = []  # 保存本轮解析后的计划，便于外部检查。

    def _parse_plan(self, text: str) -> list[str]:
        """把模型生成的列表文本转成 Python 列表，只接受非空的字符串步骤。"""
        if len(text) > 20000:
            raise AgentParseError("计划输出过长")
        # 模型常把列表包在 Markdown 代码围栏中；先提取围栏内的文字。
        # fullmatch 要求整段输出符合格式；DOTALL 让 . 也能匹配换行。
        fenced = re.fullmatch(r"\s*```(?:json|python)?\s*\n?(.*?)\n?```\s*", text, re.DOTALL)
        content = fenced.group(1).strip() if fenced else text.strip()
        try:
            # 首选标准 JSON，例如 ["求单价", "计算总价"]。
            plan = json.loads(content)
        except ValueError:
            try:
                # 兼容 Python 单引号列表。literal_eval 只解析字面量，不执行任意代码。
                plan = ast.literal_eval(content)
            except (ValueError, SyntaxError, RecursionError) as exc:
                raise AgentParseError("计划必须是 JSON/Python 字符串列表，不能是代码或自由文本") from exc
        # 解析成功还不够：必须是非空列表，且每项都是有内容的字符串。
        if not isinstance(plan, list) or not plan or any(not isinstance(step, str) or not step.strip() for step in plan):
            raise AgentParseError("计划必须是非空字符串列表")
        if len(plan) > self.config.max_plan_steps:
            raise AgentParseError(f"计划不能超过 {self.config.max_plan_steps} 步")
        return [step.strip() for step in plan]

    def run(self, input_text: str, **kwargs: Any) -> str:
        """一次规划 + 每个步骤一次执行，调用上限为 1 + config.max_plan_steps。"""
        context = self._start(input_text)[:-1]
        self.plan = []
        # 1. 先问模型要计划，拿到并校验计划后才开始执行，避免执行无效计划。
        prompt = self.prompts["planner"].format(question=input_text)
        response = self.llm.invoke(context + [{"role": "user", "content": prompt}], **self._options(kwargs))
        self.steps.append({"phase": "planner", "response": response})
        self.plan = self._parse_plan(response)
        completed = []
        for index, step in enumerate(self.plan, 1):
            # 2. enumerate(..., 1) 从第 1 步编号；把前面的真实结果传给当前步骤。
            prompt = self.prompts["executor"].format(
                question=input_text, plan=json.dumps(self.plan, ensure_ascii=False),
                history="\n".join(completed) or "暂无", current_step=step,
            )
            answer = self.llm.invoke(context + [{"role": "user", "content": prompt}], **self._options(kwargs))
            completed.append(f"{index}. {step}\n结果：{answer}")
            self.steps.append({"phase": "executor", "step": step, "response": answer})
        # 3. 计划已保证非空，循环结束后的 answer 就是最后一步对原问题的回答。
        return self._remember(input_text, answer)
