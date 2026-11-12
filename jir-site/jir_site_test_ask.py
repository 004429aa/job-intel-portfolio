# jir_site_test_ask.py —— U6 RAG问答接口测试
# 运行：pytest jir_site_test_ask.py -v
import os
import pytest
from fastapi.testclient import TestClient

import jir_site_api                                        # 被测模块（import会触发模型加载，稍慢）
from jir_site_api import app

client = TestClient(app)


# 用例1：不设key → 必须 503（可离线测，不消耗API额度）
# 关键：用 monkeypatch 改"模块级 DEEPSEEK_KEY"，而不是改环境变量——
# 因为模块在 import 时就把 key 读进常量了，事后改 env 不会生效
def test_ask_no_key_returns_503(monkeypatch):
    monkeypatch.setattr(jir_site_api, "DEEPSEEK_KEY", None)
    resp = client.post("/jobs/ask",
                       json={"question": "有哪些和机器视觉相关的岗位", "top_k": 3})
    assert resp.status_code == 503
    assert "DEEPSEEK_API_KEY" in resp.json()["message"]


# 用例2：真实调用（仅当设置了key才跑）
# skipif 写法：没有key就跳过不报红 —— 本单元的新知识点
# 注意：判断的是"环境变量"，不是模块常量，所以在命令行设了key就生效
@pytest.mark.skipif(not os.environ.get("DEEPSEEK_API_KEY"),
                    reason="未设置DEEPSEEK_API_KEY，跳过真实调用")
def test_ask_real_call_returns_answer():
    resp = client.post("/jobs/ask",
                       json={"question": "有哪些和机器视觉相关的岗位", "top_k": 3})
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"]                                # 有生成答案
    assert isinstance(body["sources"], list)             # 有可查证的来源
    assert "[" in body["answer"]                         # 答案里岗位带[编号]，可对照sources查证
