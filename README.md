# Job Intel Portfolio

这个作品集包含岗位情报分析、深度学习、岗位情报站后端三个实践项目。岗位情报站采集了30条广深 AI 交叉岗位 JD，使用 DeepSeek 批量解析，并通过 pandas 完成统计和图表报告；NEU 钢面缺陷分类项目使用 PyTorch 从零训练 CNN，对六类缺陷进行分类。将输入分辨率从64提高到128、训练轮数从10增加到15后，准确率从60%多提升至约82%。

## 在线体验

Jir-Site 已部署上线，可直接访问：

- 首页（API 欢迎页）：https://job-intel-rag.app.workbuddy.host/?sharecode=PP7HFLNZRJ
- 交互式文档（Swagger UI，可在网页上直接调试每个接口）：https://job-intel-rag.app.workbuddy.host/docs?sharecode=PP7HFLNZRJ

线上可用接口：`GET /jobs`（四参过滤）、`GET /jobs/{id}`、`GET /jobs/semantic`（语义检索）、`POST /jobs/ask`（RAG 问答）。

## 三个项目一览

| 项目 | 目录 | 技术栈 | 做了什么 |
|---|---|---|---|
| Jir-Py 岗位数据分析 | `jir-py/` | requests / pandas / matplotlib | 30条JD采集→DeepSeek解析→SQLite→统计图表→HTML报告 |
| Jir-DL 缺陷分类 | `jir-dl/` | PyTorch / CNN | NEU钢面缺陷6分类，U1-U10单元制学习，准确率约82% |
| Jir-Site 岗位情报API | `jir-site/` | FastAPI / BGE / DeepSeek | 四参过滤API→语义检索→RAG问答→pytest→云端部署 |

**Jir-Site 岗位情报站后端（FastAPI + RAG）**：开发了一套全链路的岗位情报 API 服务。底层用 Pandas 和 SQLite 搭建全量数据导入管道；中间层利用 BGE 模型将岗位 JD 向量化并存入本地 NPZ 库，实现毫秒级的语义检索（打破了传统关键词死板匹配的局限）；上层接入 DeepSeek API 实现 RAG 问答。核心亮点：设计了严格的 Prompt 防幻觉机制（带编号溯源、拒答无资料问题），并实现了 A/B 控制变量测试与边界防崩溃的全套 Pytest 单元测试。

部署是这个项目的最后一环，踩坑也最密集：云沙箱启动命令不展开 `$PORT`（改用 Python 进程内读环境变量）；torch 在弱配沙箱加载超过健康检查60秒窗口（改为懒加载仍不行，最终把 BGE 模型导出为 ONNX，用 onnxruntime 推理，导入约1秒，与原模型输出对齐验证余弦相似度=1.000000）；数据库改用相对路径避免线上找不到文件。完整踩坑记录见 `jir-site线上版/README`。

## 本地运行 Jir-Site

```bash
cd jir-site
pip install fastapi uvicorn numpy openai sentence-transformers pytest pandas openpyxl

# 1. 首次运行：建库+导入30条JD，再生成向量库
#    注意：jir_site_db.py 顶部配置区是本机绝对路径，需先改成你的目录
python jir_site_db.py             # 读 xlsx → SQLite（需 pandas）
python jir_site_embed.py          # BGE向量化 → jd_vectors.npz

# 2. 启动 API
python jir_site_api.py            # 默认 127.0.0.1:8000，浏览器开 /docs

# 3. 跑测试（20项，含真实LLM调用用例默认跳过）
pytest jir_site_test_api.py jir_site_test_ask.py

# 4. RAG 问答需要设置 DeepSeek key
setx DEEPSEEK_API_KEY <你的key>
```

注：向量库 `jd_vectors.npz` 和数据库 `情报站.db` 未随仓库提交（见 .gitignore），需先运行 `jir_site_db.py`（建库）和 `jir_site_embed.py`（向量化）生成。`jir-site线上版/` 是部署用的最小运行集，含模型本地化导出脚本与沙箱适配版代码（ONNX 推理版 API）。

## AI 参与程度（如实标注）

我主要参与岗位数据采集、脚本理解与调试，以及图表和报告核对；岗位项目的主体代码由 AI agent 编写。NEU 项目基于 agent 模板完成训练，我亲自调整参数并记录结果，网络结构并非原创设计。Jir-Site 项目约70%代码由 AI 生成，AI 负责基础 CRUD 路由、Pydantic 校验模型的拼装、SQL/向量检索的底层实现；我负责整体架构设计、业务逻辑串联、Prompt 的防幻觉调优、Windows 环境（国内镜像/文件占用）的踩坑与修复，以及控制变量测试方案的设计。部署环节的 ONNX 迁移方案由 AI 提出，我完成了导出脚本的编写、对齐验证和线上部署实操。目前岗位分析只有30条样本，适合展示流程和观察趋势；NEU 项目尚未使用预训练模型或数据增强，仍存在过拟合风险，结果还有改进空间。
