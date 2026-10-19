"""python -m hello_agents 的入口，把参数处理与任务执行统一交给 cli.main。"""

from .cli import main

# SystemExit 把 main 返回的整数用作进程退出码，终端可以据此判断是否成功。
raise SystemExit(main())
