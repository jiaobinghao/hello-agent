"""每个实例独立的配置，避免多个 Agent 相互修改全局状态。"""

import os
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .exceptions import ConfigurationError


class Config(BaseModel):
    """集中管理运行参数，Pydantic 在创建和赋值时负责类型转换与范围检查。

    例如环境变量里的 "5" 会转成整数 5；max_steps=0 会被拒绝。
    这里不保存密钥，密钥由模型客户端读取，避免输出配置时带出凭证。
    """

    # 禁止拼错的额外配置字段；对象创建后修改字段时也要重新校验。
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    default_model: str | None = None  # 具体模型 ID；None 表示尚未配置。
    default_provider: str = "auto"  # 服务提供方；auto 尝试按 URL 或密钥变量判断。
    temperature: float | None = Field(default=0.7, ge=0, le=2)  # 控制采样随机性；None 表示不发送此参数。
    max_tokens: int | None = Field(default=None, gt=0)  # 单次生成的 token 上限，非字符数。
    timeout: float = Field(default=60, gt=0)  # 单次网络请求的超时秒数。
    max_steps: int = Field(default=5, gt=0)  # Simple/ReAct/FunctionCall 的模型调用次数上限。
    max_plan_steps: int = Field(default=8, gt=0)  # PlanAndSolve 最多拆出多少个步骤。
    max_history_length: int = Field(default=100, ge=2)  # 最多保存多少条历史消息。
    # 留给应用层使用的调试设置；库本身不自动设置全局日志或打印调试输出。
    debug: bool = False
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> "Config":
        """从进程环境变量构造配置；读取 .env 文件由 CLI 或调用方负责。"""
        # 左边是 Python 字段名，右边是按优先级排列的环境变量名。
        fields = {
            "default_model": ("LLM_MODEL_ID",),
            "default_provider": ("LLM_PROVIDER",),
            "temperature": ("LLM_TEMPERATURE", "TEMPERATURE"),
            "max_tokens": ("LLM_MAX_TOKENS", "MAX_TOKENS"),
            "timeout": ("LLM_TIMEOUT",),
            "max_steps": ("LLM_MAX_STEPS",),
            "max_plan_steps": ("LLM_MAX_PLAN_STEPS",),
            "max_history_length": ("LLM_MAX_HISTORY_LENGTH",),
            "debug": ("LLM_DEBUG", "DEBUG"),
            "log_level": ("LOG_LEVEL",),
        }
        values = {}
        for field, names in fields.items():
            for name in names:
                if os.getenv(name, "").strip():# 这里真正读取环境变量值
                    values[field] = os.environ[name]
                    if name == "DEBUG":
                        # 通用 DEBUG 可能被宿主设置为 release 等非布尔值。
                        values[field] = os.environ[name].strip().lower() in {"true", "1", "yes", "on"}
                    break  # 命中第一个非空变量后停止，不再使用后面的兼容名称。
        try:
            # **values 把字典展开为关键字参数；cls(...) 触发 Pydantic 校验。
            return cls(**values)
        except ValidationError as exc:
            raise ConfigurationError(f"环境变量配置无效：{exc}") from exc

    def to_dict(self) -> dict[str, Any]:
        """把配置对象转成普通字典，供参数合并或展示使用。"""
        return self.model_dump()
