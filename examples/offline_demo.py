"""先在项目目录安装 pip install -e .，再运行此文件。"""

from hello_agents.demo import run_demo

if __name__ == "__main__":
    # 复用包内演示，脚本模型不需要 API Key，计算器与工具执行流程仍运行真实代码。
    run_demo()
