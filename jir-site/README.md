# Jir-Site 岗位情报站后端（FastAPI + RAG）

30条广深AI交叉岗JD 入库 SQLite → FastAPI 五类端点 → BGE 语义检索 → DeepSeek RAG 问答（带[编号]引用防幻觉），20项 pytest 全绿，已双线部署上线。

## 做了什么

底层用 Pandas 和 SQLite 搭建全量数据导入管道；中间层利用 BGE 模型（bge-small-zh-v1.5）将岗位 JD 向量化存入本地 NPZ 库，实现毫秒级语义检索（打破关键词死板匹配的局限）；上层接入 DeepSeek API 实现 RAG 问答。核心亮点：严格的 Prompt 防幻觉机制（带编号溯源、拒答无资料问题）、A/B 控制变量测试（`run_abtest.bat`）、边界防崩溃的全套 Pytest 单元测试。

## 文件清单

| 文件 | 作用 |
|---|---|
| `jir_site_db.py` | 建库+导入30条JD（DROP TABLE IF EXISTS 重建） |
| `jir_site_embed.py` | BGE 向量化 → jd_vectors.npz |
| `jir_site_api.py` | API主体：GET /jobs（四参过滤）、/jobs/{id}、/jobs/semantic、POST /jobs/ask、DELETE、根路由 |
| `jir_site_test_api.py` | 18项接口测试 |
| `jir_site_test_ask.py` | RAG问答测试（含无key 503降级、真实LLM调用skipif） |
| `run_abtest.bat` | A/B 控制变量调优入口（结果写入 RAG评估.md） |
| `RAG评估.md` | A/B 测试结果与评估记录 |
| `AI交叉岗位情报站.xlsx` | 原始数据源（30条JD） |

## 本地运行

```bash
pip install fastapi uvicorn pandas openpyxl numpy sentence-transformers openai pytest
python jir_site_db.py        # 建库（路径配置在文件顶部，需按本机修改）
python jir_site_embed.py     # 向量化
python jir_site_api.py       # 启动，浏览器开 http://127.0.0.1:8000/docs
pytest jir_site_test_api.py jir_site_test_ask.py
```

RAG 问答需设置 `DEEPSEEK_API_KEY` 环境变量，未设置时 /ask 返回 503（有测试覆盖）。

## 部署

已双线部署：云托管平台（可直接访问，链接见仓库主 README）+ 自有腾讯云服务器（nginx + systemd + SSH 密钥）。完整部署实录与踩坑见 `jir-site线上版/README` 和仓库主 README 部署段。

## AI 参与程度（如实标注）

约 70% 代码由 AI 生成或基于参考样本：AI 负责基础 CRUD 路由、Pydantic 校验模型拼装、SQL/向量检索底层实现、pytest 用例编写。我负责整体架构设计、业务逻辑串联、Prompt 防幻觉调优、Windows 环境踩坑修复、控制变量测试方案设计，以及云服务器部署的全部命令实操与排障。
