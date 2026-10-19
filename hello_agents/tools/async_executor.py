"""有并发上限的同步工具线程执行器，按输入顺序返回结果。"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any

from .registry import ToolRegistry
from ..core.exceptions import ToolError


class AsyncToolExecutor:
    """把耗时的同步工具交给线程池，使多个独立工具可以同时等待外部结果。

    async/await 让调用方在等待时继续运行其他任务；同步工具本身仍在工作线程
    中执行，不会自动变成协程。带依赖的步骤应使用 ToolChain 顺序执行。
    """

    def __init__(self, registry: ToolRegistry, max_workers: int = 4):
        """max_workers 限制同时执行的工作线程数，多出的任务在线程池中排队。"""
        self.registry = registry
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="hello-tool")
        self._closed = False

    async def execute_tool_async(self, tool_name: str, input_data: str | dict[str, Any]) -> str:
        """提交一个工具任务，等待其执行结果；等待期间不占用事件循环线程。"""
        if self._closed:
            raise ToolError("执行器已关闭")
        loop = asyncio.get_running_loop()
        # partial 预先绑定工具名与参数，得到一个无需参数即可执行的函数。
        # run_in_executor 把这个函数放进线程池，await 等它执行完后再取得结果。
        return await loop.run_in_executor(
            self.executor, partial(self.registry.execute_tool, tool_name, input_data),
        )

    async def execute_tools_parallel(self, tasks: list[dict[str, Any]], *, return_exceptions: bool = False) -> list[Any]:
        """并发执行任务列表，结果位置与输入位置一致，不按完成时间重新排序。

        return_exceptions=False 时向调用方抛出错误；True 时将异常对象放在对应
        结果位置。已经在线程中运行的同步工具不会因等待被取消而强制停止。
        """
        # * 将多个协程展开为 gather 的参数；gather 负责调度并收集所有结果。
        return await asyncio.gather(
            *(self.execute_tool_async(task["tool_name"], task["input_data"]) for task in tasks),
            return_exceptions=return_exceptions,
        )

    def close(self) -> None:
        """关闭执行器：取消尚未开始的排队任务，等待已开始的工具结束。"""
        self._closed = True
        self.executor.shutdown(wait=True, cancel_futures=True)

    def __enter__(self) -> "AsyncToolExecutor":
        """支持 with，用完后自动关闭线程池。"""
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    async def __aenter__(self) -> "AsyncToolExecutor":
        """支持 async with，适合在异步程序中管理执行器的生命周期。"""
        return self

    async def __aexit__(self, *_: Any) -> None:
        # close 会等待工作线程退出，因此也把它放在线程中，避免阻塞事件循环。
        await asyncio.to_thread(self.close)
