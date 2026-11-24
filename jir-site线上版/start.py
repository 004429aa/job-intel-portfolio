# start.py —— 线上启动入口
# 为什么不用 uvicorn --port $PORT：沙箱的启动命令不经shell展开$PORT会变空参数
# 在Python进程内读PORT环境变量最稳（托管平台注入的就是这个变量）
import os
import uvicorn

port = int(os.environ.get("PORT", 3000))   # 平台注入PORT，兜底3000
uvicorn.run("jir_site_api:app", host="0.0.0.0", port=port)
