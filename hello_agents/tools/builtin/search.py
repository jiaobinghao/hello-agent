"""可选的 Tavily 搜索适配；不安装额外搜索 SDK。"""

import json
import os
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from ..base import Tool, ToolParameter
from ...core.exceptions import ToolError


class SearchTool(Tool):
    """把 Tavily HTTP 搜索接口包装为 Tool，模型只需要提供 query 搜索词。"""

    def __init__(self, api_key: str | None = None, max_results: int = 3, timeout: float = 20):
        """密钥可显式传入或从环境读取；创建工具本身不发起搜索请求。"""
        super().__init__("search", "搜索互联网资料，返回标题、链接和内容摘要；需要 Tavily API Key")
        if not 1 <= max_results <= 10 or timeout <= 0:
            raise ValueError("max_results 必须为 1～10，timeout 必须大于 0")
        self._api_key = api_key or os.getenv("TAVILY_API_KEY")
        self.max_results = max_results
        self.timeout = timeout

    def get_parameters(self) -> list[ToolParameter]:
        """只把搜索词暴露给模型；密钥、超时和结果数量由程序配置。"""
        return [ToolParameter(name="query", description="要搜索的关键词或问题")]

    def run(self, parameters: dict[str, Any]) -> str:
        """校验搜索词，发送一次 HTTP 请求，并将结果列表编码为文本供模型阅读。"""
        query = self.validate_parameters(parameters)["query"]
        if not self._api_key:
            raise ToolError("搜索需要配置 TAVILY_API_KEY")
        if not query.strip():
            raise ToolError("搜索词不能为空")
        # Python 字典 → JSON 字符串 → UTF-8 字节，成为 POST 请求体。
        # Authorization 请求头携带密钥，参数说明和工具结果里都不需要出现密钥。
        request = Request(
            "https://api.tavily.com/search",
            data=json.dumps({"query": query, "max_results": self.max_results}).encode(),
            headers={"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            # with 确保响应连接及时关闭；timeout 限制等待搜索服务的时间。
            with urlopen(request, timeout=self.timeout) as response:
                payload = json.load(response)
            # 工具统一返回 str；ensure_ascii=False 保持中文可读，切片限制返回条数。
            return json.dumps(payload.get("results", [])[:self.max_results], ensure_ascii=False)
        except HTTPError as exc:
            # 状态错误与网络/格式错误统一包装，Agent 可把它们反馈给模型。
            raise ToolError(f"搜索服务返回 HTTP {exc.code}") from exc
        except (URLError, TimeoutError, ValueError, TypeError, AttributeError) as exc:
            raise ToolError("搜索请求失败或响应格式无效") from exc
