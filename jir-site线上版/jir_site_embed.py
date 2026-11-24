# jir_site_embed.py学习演示脚本，输出只打印'id+分数'，不返回完整结构化岗位数据，不能给 FastAPI 直接调用。
import os
import sqlite3
import numpy as np

# 镜像配置，必须放在导入模型之前
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from sentence_transformers import SentenceTransformer

# 路径配置
# 【U7部署改动】绝对路径→相对脚本目录（同api）
_DIR = __file__.rsplit("\\", 1)[0].rsplit("/", 1)[0]
DB_PATH = _DIR + "/情报站.db"
VEC_PATH = _DIR + "/jd_vectors.npz"

# 全局加载模型（只加载一次）
model = SentenceTransformer("model")


def build_vectors():
    """读取jobs表，拼接文本，生成向量并保存npz文件"""
    # 连接数据库读取JD
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT id, title, skills, notes FROM jobs")
    rows = cur.fetchall()
    conn.close()

    ids = []
    texts = []
    for row in rows:
        jd_id, title, skills, notes = row
        # 拼接策略：标题+技能+备注，过滤空字符串
        content_parts = [p for p in [title, skills, notes] if p]
        ids.append(jd_id)
        texts.append("。".join(content_parts))

    # 生成向量，开启归一化（余弦相似度等价于矩阵点乘）
    embeddings = model.encode(texts, normalize_embeddings=True)
    # 保存id数组+向量数组
    np.savez(VEC_PATH, ids=np.array(ids), vectors=embeddings)
    print(f"✅ 成功向量化 {len(ids)} 条JD，向量维度：{embeddings.shape[1]}")


def compare_search(query: str):
    """关键词检索 vs 语义检索对比（评估用）"""
    # 1. 关键词模糊检索
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "SELECT id, title FROM jobs WHERE skills LIKE ? OR title LIKE ?",
        (f"%{query}%", f"%{query}%")
    )
    kw_hits = cur.fetchall()
    conn.close()

    # 2. 语义向量检索
    data = np.load(VEC_PATH)
    ids_arr, vec_arr = data["ids"], data["vectors"]
    q_vec = model.encode([query], normalize_embeddings=True)[0]
    sim_scores = vec_arr @ q_vec
    top_idx = np.argsort(-sim_scores)[:5]

    sem_hits = []
    for idx in top_idx:
        jd_id = int(ids_arr[idx])
        score = round(float(sim_scores[idx]), 4)
        sem_hits.append((jd_id, score))

    # 打印对比结果
    print("-" * 60)
    print(f"查询词：{query}")
    print(f"关键词检索命中条数：{len(kw_hits)}，标题预览：{[t for _, t in kw_hits[:3]]}")
    print(f"语义检索TOP5（id,相似度）：{sem_hits}")


if __name__ == "__main__":
    build_vectors()
    # 三组测试query
    compare_search("神经网络图像识别")
    compare_search("大模型应用开发")
    compare_search("设备故障预测")
