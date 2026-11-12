import sqlite3
import os
import pandas as pd

# ===================== 配置区【本地修改这里】 =====================
DB_PATH = r"E:\Aaaaaaa.Item\Item--Job Information Report\Jir-Site\情报站.db"
XLSX_PATH = r"E:\Aaaaaaa.Item\Item--Job Information Report\AI交叉岗位情报站.xlsx"

def init_db():
    """初始化数据库：DROP旧表重建jobs表
    （U5改：原os.remove删整个db文件，Windows下文件被占用/回收站机制会偶发失败；
    DROP TABLE IF EXISTS不碰文件系统，更稳，也保留db文件本身）"""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS jobs")

    create_sql = """
    CREATE TABLE IF NOT EXISTS jobs (
        id INTEGER PRIMARY KEY,
        title TEXT NOT NULL,
        company TEXT NOT NULL,
        city TEXT NOT NULL,
        source TEXT NOT NULL,
        industry TEXT,
        experience TEXT,
        education TEXT,
        salary_min REAL,
        salary_max REAL,
        skills TEXT,
        notes TEXT
    );
    """
    cur.execute(create_sql)
    conn.commit()
    conn.close()
    print("✅ 数据表初始化完成")


def load_excel_rows(filepath):
    """pandas读取xlsx，输出可入库的元组列表"""
    df = pd.read_excel(filepath, engine="openpyxl")
    row_list = []
    for _, row in df.iterrows():
        one = (
            int(row["序号"]),
            row["岗位名称"],
            row["公司"],
            row["城市"],
            row["来源"],
            row["行业赛道"],
            row["经验要求"],
            row["学历要求"],
            float(row["薪资下限(k/月)"]),
            float(row["薪资上限(k/月)"]),
            row["技能关键词"],
            row["备注"]
        )
        row_list.append(one)
    return row_list


def insert_data(row_list):
    """参数化入库，?占位符"""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    insert_sql = "INSERT INTO jobs VALUES (?,?,?,?,?,?,?,?,?,?,?,?)"
    for r in row_list:
        cur.execute(insert_sql, r)
    conn.commit()
    conn.close()
    print(f"✅ 一共插入 {len(row_list)} 条岗位数据")


def query_demo():
    """4个基础查询 + 改动测试第5个查询"""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    print("\n===== 查询1：总条数 + 前5条数据 =====")
    cur.execute("SELECT COUNT(*) FROM jobs")
    total = cur.fetchone()[0]
    print(f"岗位总数：{total}")
    cur.execute("SELECT * FROM jobs LIMIT 5")
    for item in cur.fetchall():
        print(item)

    print("\n===== 查询2：城市=广州 的岗位数量【参数化】 =====")
    city_target = "广州"
    cur.execute("SELECT COUNT(*) FROM jobs WHERE city = ?", (city_target,))
    cnt_gz = cur.fetchone()[0]
    print(f"广州岗位数量：{cnt_gz}")

    print("\n===== 查询3：薪资下限 ≥15k，展示岗位名、公司、薪资区间 =====")
    cur.execute("SELECT title, company, salary_min, salary_max FROM jobs WHERE salary_min >= ?", (15,))
    res_sal = cur.fetchall()
    for r in res_sal:
        print(r)

    print("\n===== 查询4：技能包含Python 的岗位数量（LIKE参数化） =====")
    kw = "%Python%"
    cur.execute("SELECT COUNT(*) FROM jobs WHERE skills LIKE ?", (kw,))
    cnt_py = cur.fetchone()[0]
    print(f"技能带Python的岗位数：{cnt_py}")

    # ========== ✅改动测试查询：学历='在读研究生'，按薪资下限降序 ==========
    print("\n===== 查询5【改动测试】：学历为在读研究生的岗位，薪资下限降序 =====")
    edu_target = "在读研究生"
    cur.execute("""
        SELECT title, company, education, salary_min, salary_max
        FROM jobs
        WHERE education = ?
        ORDER BY salary_min DESC
    """, (edu_target,))
    res_edu = cur.fetchall()
    for r in res_edu:
        print(r)

    conn.close()


if __name__ == "__main__":
    init_db()
    rows = load_excel_rows(XLSX_PATH)
    insert_data(rows)
    query_demo()
