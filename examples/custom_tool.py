"""函数注册、参数推断和真实模型接入。"""

from dotenv import load_dotenv

from hello_agents import FunctionCallAgent, HelloAgentsLLM, ToolRegistry


def convert_length(value: float, from_unit: str = "m", to_unit: str = "cm") -> str:
    """换算 m、cm、mm 长度单位，例如 convert_length(1.5) 返回 "150 cm"。

    函数签名也是工具的参数说明：value 是必填数字，两个单位参数有默认值。
    注册后，框架会读取这些类型标注和默认值，自动生成参数校验模型。
    """
    # 每种单位换算成米的倍率：1 cm = 0.01 m，1 mm = 0.001 m。
    scales = {"m": 1, "cm": 0.01, "mm": 0.001}
    if from_unit not in scales or to_unit not in scales:
        raise ValueError("只支持 m、cm、mm")
    # 先换算成米，再除以目标单位的倍率；:g 去掉数值末尾不必要的小数零。
    return f"{value * scales[from_unit] / scales[to_unit]:g} {to_unit}"


def main() -> None:
    """加载连接配置，注册普通函数，让模型通过原生调用使用它。"""
    # 1. 脚本主动加载 .env；导入框架本身不会隐式读取环境文件。
    load_dotenv()
    registry = ToolRegistry()
    # 2. 函数只是登记到注册表中，这一步不会真的换算任何数值。
    registry.register_function("convert_length", "换算 m、cm、mm 长度单位", convert_length)
    with HelloAgentsLLM() as llm:
        # 3. run() 内把函数定义发送给模型；收到调用请求后程序才会执行 convert_length。
        agent = FunctionCallAgent("单位换算助手", llm, registry)
        print(agent.run("请用工具把 1.5 米换成厘米"))
        print(agent.get_trace())  # 检查模型实际传了什么参数、工具返回了什么。


if __name__ == "__main__":
    # 直接执行文件时启动示例；被其他脚本 import 时只定义函数，不发模型请求。
    main()
