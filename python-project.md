一般初始化 Python 仓库，会准备好 **Git、虚拟环境、目录结构和依赖配置**。以一个简单的 Python 包为例：

1. **创建目录和 Git 仓库**

   ```bash
   mkdir my-project
   cd my-project
   git init -b main
   ```

2. **创建独立的 Python 环境**

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

   激活后，`python` 和 `pip` 都使用这个项目的环境，安装依赖不会影响其他项目。

3. **准备基本目录**

   ```text
   my-project/
   ├── pyproject.toml      # 包信息、依赖、构建配置
   ├── README.md           # 项目介绍、安装与运行方法
   ├── .gitignore          # Git 忽略规则
   ├── my_package/         # Python 源码
   │   └── __init__.py
   └── tests/              # 测试
       └── test_example.py
   ```

   `my-project` 是项目名称；`my_package` 是代码中使用的导入名称：

   ```python
   import my_package
   ```

4. **用 `pyproject.toml` 声明项目**

   最小示例：

   ```toml
   [build-system]
   requires = ["setuptools"]
   build-backend = "setuptools.build_meta"

   [project]
   name = "my-project"
   version = "0.1.0"
   description = "一个 Python 项目"
   requires-python = ">=3.10"
   dependencies = []

   [tool.setuptools.packages.find]
   include = ["my_package*"]
   ```

   如果项目需要第三方库，就在 `dependencies` 中声明，例如：

   ```toml
   dependencies = [
       "pydantic>=2,<3",
       "python-dotenv>=1,<2",
   ]
   ```

   然后安装项目：

   ```bash
   python -m pip install -e .
   ```

   `-e` 表示**以开发模式安装**，修改源码后会直接生效，不必反复安装。

5. **配置忽略规则，再做首次提交**

   `.gitignore` 可以先写：

   ```gitignore
   .venv/
   __pycache__/
   *.py[cod]
   *.egg-info/
   build/
   dist/
   .env
   .env.*
   !.env.example
   .pytest_cache/
   .ruff_cache/
   ```

   `.env.example` 可以保存配置示例，真实密钥放在被忽略的 `.env` 中。

   检查文件后提交：

   ```bash
   git status
   git add .
   git diff --cached
   git commit -m "初始化 Python 项目"
   ```
