# jir_site_api.py —— U3规范基底 + U5语义检索 + U6 RAG问答（ask端点已接入）
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, model_validator
from typing import Optional
import sqlite3
import os
import numpy as np
from datetime import datetime

# ========== 【U6新增】LLM密钥：参数是环境变量名，绝不写明文 ==========
DEEPSEEK_KEY = os.environ.get("DEEPSEEK_API_KEY")
from openai import OpenAI

# 镜像前置，必须在导入sentence_transformers之前
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from sentence_transformers import SentenceTransformer#转语义向量embedding，专门做语义检索

DB_PATH = r"E:\Aaaaaaa.Item\Item--Job Information Report\Jir-Site\情报站.db"
VEC_PATH = r"E:\Aaaaaaa.Item\Item--Job Information Report\Jir-Site\jd_vectors.npz"

app = FastAPI(title="岗位情报站API", version="0.3")

# ========== 【U5新增：全局加载Embedding模型，只加载一次】 ==========
model = SentenceTransformer("BAAI/bge-small-zh-v1.5")

# ===== 数据模型：POST请求体（入） =====
class JobIn(BaseModel):
    """POST/PUT 请求体。Field约束自动校验+生成/docs文档。"""
    title: str = Field(..., min_length=2, description="岗位名称，至少2字")
    company: str
    city: str
    source: str = "手动录入"
    industry: Optional[str] = None
    experience: Optional[str] = None
    education: Optional[str] = None
    salary_min: float = Field(..., ge=0, description="月薪下限k，非负")
    salary_max: float = Field(..., ge=0)
    skills: Optional[str] = None
    notes: Optional[str] = None

    @model_validator(mode="after")
    def check_salary_range(self):
        if self.salary_min > self.salary_max:
            raise ValueError("薪资下限不能大于上限")
        return self

# ===== GET 普通响应模型 =====
class JobOut(BaseModel):
    id: int
    title: str
    company: str
    city: str
    source: str
    industry: Optional[str]
    experience: Optional[str]
    education: Optional[str]
    salary_min: float
    salary_max: float
    skills: Optional[str]
    notes: Optional[str]

class JobListResponse(BaseModel):
    count: int
    jobs: list[JobOut]

class JobCreateResponse(BaseModel):
    id: int
    message: str
    job: JobIn

# ========== 【U5新增：语义检索返回模型】 ==========
class JobSemanticOut(BaseModel):
    id: int
    title: str
    company: str
    city: str
    score: float

class JobSemanticResponse(BaseModel):
    count: int
    jobs: list[JobSemanticOut]

# ========== 【U6新增：RAG问答请求体模型】 ==========
class AskIn(BaseModel):
    question: str = Field(..., min_length=2, description="用户问题，如：有哪些和机器视觉相关的岗位")
    top_k: int = Field(5, ge=1, le=10, description="检索多少条JD作为参考资料")
    temperature: float = Field(0.3, ge=0.0, le=2.0,
                               description="生成温度：低=老实转述资料，高=自由发挥（改动测试项）")

# ========== 【U6新增：RAG问答返回模型】 ==========
class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list[dict]   # 命中的岗位及相似度，方便追溯

# ===== 工具：sqlite元组行转字典 =====
def row_to_dict(row) -> dict:
    keys = ["id", "title", "company", "city", "source", "industry",
            "experience", "education", "salary_min", "salary_max",
            "skills", "notes"]
    return dict(zip(keys, row))

# ===== 统一错误格式 =====
@app.exception_handler(HTTPException)
async def unify_http_exception(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"code": exc.status_code, "message": str(exc.detail), "data": None},
    )

# ========== 【U6新增①】检索：把U5的semantic检索抽成函数复用 ==========
# 卡片要点：多取 notes/skills/薪资字段——LLM靠JD原文回答，只给岗位名没料可答
def search_jd(question: str, top_k: int = 5, score_threshold: float = 0.0) -> list[dict]:
    try:
        vec_data = np.load(VEC_PATH)
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="向量文件jd_vectors.npz不存在，请先运行jir_site_embed.py生成向量库")

    q_vec = model.encode(question, normalize_embeddings=True)
    sim_scores = vec_data["vectors"] @ q_vec
    top_idx = np.argsort(-sim_scores)[:top_k]

    hits = []
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        for i in top_idx:
            jid = int(vec_data["ids"][i])
            score = float(sim_scores[i])
            if score < score_threshold:
                continue
            cur.execute(
                "SELECT title, company, city, skills, notes, salary_min, salary_max FROM jobs WHERE id=?",
                (jid,))
            row = cur.fetchone()
            if row:
                title, company, city, skills, notes, smin, smax = row
                hits.append({
                    "id": jid, "title": title, "company": company, "city": city,
                    "skills": skills, "notes": notes,
                    "salary_min": smin, "salary_max": smax,
                    "score": round(score, 4),
                })
    return hits

# ===== GET /jobs 列表，可选过滤 =====
@app.get("/jobs", response_model=JobListResponse)
def list_jobs(
    city: Optional[str] = Query(None, description="按城市精确过滤"),
    min_salary: Optional[float] = Query(None, ge=0, description="薪资下限≥此值"),
    keyword: Optional[str] = Query(None, description="技能关键词模糊检索"),
    limit: int = Query(30, ge=1, le=100),
):
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        sql = "SELECT * FROM jobs WHERE 1=1"
        params = []
        if city:
            sql += " AND city = ?"; params.append(city)
        if min_salary is not None:
            sql += " AND salary_min >= ?"; params.append(min_salary)
        if keyword:
            sql += " AND skills LIKE ?"; params.append(f"%{keyword}%")
        sql += " LIMIT ?"; params.append(limit)
        cur.execute(sql, params)
        rows = cur.fetchall()
    return {"count": len(rows), "jobs": [row_to_dict(r) for r in rows]}

# ========== 【U5语义检索接口 GET /jobs/semantic】【U6改为复用search_jd，不再各写一份】 ==========
@app.get("/jobs/semantic", response_model=JobSemanticResponse)
def semantic_search(
    query: str = Query(..., description="语义检索关键词"),
    top_k: int = Query(5, ge=1, le=20, description="返回结果数量"),
    score_threshold: float = Query(0.0, ge=0.0, le=1.0, description="相似度阈值（优化点）")
):
    hits = search_jd(query, top_k=top_k, score_threshold=score_threshold)
    return {
        "count": len(hits),
        "jobs": [{"id": h["id"], "title": h["title"], "company": h["company"],
                  "city": h["city"], "score": h["score"]} for h in hits]
    }

# ========== 【U6新增②】拼Prompt：三段结构（角色+资料+问题） ==========
# 卡片要点：那句"资料里没有就明确说，不要编造"是防幻觉命门
def build_prompt(question: str, hits: list[dict]) -> list[dict]:
    system = ("你是岗位情报助手。只依据下面的岗位资料回答用户问题，回答中的每个岗位都要带上[编号]以便查证。"
              "资料里没有的信息，明确回答'资料里没有'，不要编造。")
    if not hits:
        material = "（未检索到相关岗位资料）"
    else:
        lines = []
        for h in hits:
            lines.append(
                f"[{h['id']}] {h['title']} @ {h['company']}（{h['city']}）"
                f"薪资{h['salary_min']}-{h['salary_max']}k，技能：{h['skills']}，职责：{h['notes']}"
            )
        material = "\n".join(lines)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": f"【岗位资料】\n{material}\n\n【用户问题】\n{question}"},
    ]

# ========== 【U6新增③】调LLM：DeepSeek，OpenAI兼容只换base_url ==========
# 卡片要点：temperature 低=老实转述资料，不要创意发挥
def ask_llm(messages: list[dict], temperature: float = 0.3) -> str:
    client = OpenAI(api_key=DEEPSEEK_KEY, base_url="https://api.deepseek.com")
    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        temperature=temperature,
    )
    return resp.choices[0].message.content

# ===== 【U6新增】RAG问答接口：/ask 与 /jobs/ask 双路径（卡片验收是 /jobs/ask） =====
@app.post("/ask", response_model=AskResponse)
@app.post("/jobs/ask", response_model=AskResponse)
def ask(ask: AskIn):
    # 无key → 503（卡片验收点：不设key直接跑 → 503）
    if not DEEPSEEK_KEY:
        raise HTTPException(status_code=503, detail="请先设置环境变量 DEEPSEEK_API_KEY")

    hits = search_jd(ask.question, ask.top_k)          # ① 检索
    answer = ask_llm(build_prompt(ask.question, hits),  # ② 拼Prompt + ③ 调LLM
                     temperature=ask.temperature)
    return {
        "question": ask.question,
        "answer": answer,
        "sources": [{"id": h["id"], "title": h["title"],
                     "company": h["company"], "score": h["score"]} for h in hits],
    }

# ===== GET /jobs/{job_id} 单条【动态路径参数放到语义接口下方】 =====
@app.get("/jobs/{job_id}", response_model=JobOut)
def get_job(job_id: int):
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute("SELECT * FROM jobs WHERE id = ?", (job_id,))
        row = cur.fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail=f"岗位id={job_id}不存在")
    return row_to_dict(row)

# ===== POST /jobs 新增 =====
@app.post("/jobs", response_model=JobCreateResponse, status_code=201)
def add_job(job: JobIn):
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute(
            """INSERT INTO jobs
               (title, company, city, source, industry, experience,
                education, salary_min, salary_max, skills, notes)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (job.title, job.company, job.city, job.source, job.industry,
             job.experience, job.education, job.salary_min, job.salary_max,
             job.skills, job.notes),
        )
        new_id = cur.lastrowid
    return {"id": new_id, "message": "新增成功", "job": job}

# ===== DELETE /jobs/{job_id} 删除 =====
@app.delete("/jobs/{job_id}")
def del_job(job_id: int):
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
        affected = cur.rowcount
        if affected == 0:
            raise HTTPException(status_code=404, detail=f"岗位id={job_id}不存在，删除失败")
    return {"deleted": affected, "message": f"id={job_id}删除成功"}

# ========== 【U6三步仪式①】改动测试：A/B对比，结果自动写进RAG评估.md ==========
def ab_compare(question: str):
    """同一问题、不同配置各调一次LLM，对比答案质量/老实程度。

    三组设计（一次只动一个变量，控制变量）：
      A top_k=3  temp=0.3   → 基线：答案聚焦、老实转述
      B top_k=8  temp=0.3   → 只变检索量：资料变多，看答案是更全还是更杂
      C top_k=3  temp=1.0   → 只变温度：看模型是否开始自由发挥/编造
    用法：python jir_site_api.py --abtest "有哪些和机器视觉相关的岗位"
    """
    if not DEEPSEEK_KEY:
        print("[AB] 未设置DEEPSEEK_API_KEY，无法调用LLM，请先 setx DEEPSEEK_API_KEY 并重开终端")
        return
    groups = [
        ("A | top_k=3  temp=0.3", 3, 0.3),
        ("B | top_k=8  temp=0.3", 8, 0.3),
        ("C | top_k=3  temp=1.0", 3, 1.0),
    ]
    results = []
    for label, tk, temp in groups:
        print(f"\n===== {label} =====")
        try:
            hits = search_jd(question, top_k=tk)
            answer = ask_llm(build_prompt(question, hits), temperature=temp)
        except HTTPException as e:
            print(f"[AB] 出错：{e.detail}")
            continue
        print(f"[检索命中 {len(hits)} 条]")
        print(answer)
        results.append({"label": label, "top_k": tk, "temperature": temp,
                        "hit_count": len(hits), "answer": answer})

    # 结果追加到 RAG评估.md（文件不存在会自动创建，与jir_site_api.py同目录）
    md_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "RAG评估.md")
    with open(md_path, "a", encoding="utf-8") as f:
        f.write(f"\n\n---\n## 改动测试 {datetime.now():%Y-%m-%d %H:%M}\n")
        f.write(f"**问题**：{question}\n\n")
        for r in results:
            f.write(f"### {r['label']}（命中{r['hit_count']}条）\n\n{r['answer']}\n\n")
    print(f"\n[AB] 结果已追加到 {md_path}")

if __name__ == "__main__":
    import sys
    import uvicorn
    if len(sys.argv) >= 3 and sys.argv[1] == "--abtest":
        ab_compare(sys.argv[2])                         # 改动测试模式
    else:
        uvicorn.run(app, host="127.0.0.1", port=8000)   # 正常启动：对象写法，文件名随便改
