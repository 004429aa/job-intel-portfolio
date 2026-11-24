import pytest
from fastapi.testclient import TestClient
from jir_site_api import app
import jir_site_db

client = TestClient(app)   # TestClient 不起真实HTTP服务，直接调app，速度快

@pytest.fixture(autouse=True)
def reset_db():
    """每个测试前重置db为30条基准。autouse=True 自动应用到所有测试。
    没这个的话，POST测试会污染db，后面的测试数据就错了。"""
    jir_site_db.init_db()
    jir_site_db.insert_data(
        jir_site_db.load_excel_rows(
            r"E:\Aaaaaaa.Item\Item--Job Information Report\AI交叉岗位情报站.xlsx"
        )
    )
    yield   # 测试在这里跑
    # yield 后面是teardown，下个测试fixture会重新重置，无需额外清理

# ===== GET /jobs 测试 =====
def test_list_all_jobs():
    """全量查询应返回30条。"""
    r = client.get("/jobs")
    assert r.status_code == 200
    assert r.json()["count"] == 30

def test_filter_by_city():
    """city=广州 应返回12条（参数URL编码由TestClient自动处理，不用手动%）"""
    r = client.get("/jobs", params={"city": "广州"})
    assert r.status_code == 200
    assert r.json()["count"] == 12

def test_filter_by_keyword():
    """keyword=Python 模糊查skills，应返回23条"""
    r = client.get("/jobs", params={"keyword": "Python"})
    assert r.json()["count"] == 23

def test_filter_combined():
    """组合过滤：city=深圳 + min_salary=20"""
    r = client.get("/jobs", params={"city": "深圳", "min_salary": 20})
    assert r.status_code == 200
    # 组合结果数不写死，只校验返回的所有岗位都满足条件（更稳健）
    for job in r.json()["jobs"]:
        assert job["city"] == "深圳"
        assert job["salary_min"] >= 20

# ===== GET /jobs/{id} 测试 =====
def test_get_single_job():
    """id=1 应返回岗位详情"""
    r = client.get("/jobs/1")
    assert r.status_code == 200
    assert r.json()["id"] == 1

def test_get_nonexistent_job():
    """id=999 应404"""
    r = client.get("/jobs/999")
    assert r.status_code == 404

def test_get_invalid_id_type():
    """id=abc 应422（FastAPI自动按int解析，非int直接422）"""
    r = client.get("/jobs/abc")
    assert r.status_code == 422

# ===== POST /jobs 测试 =====
def test_create_job_success():
    """正常POST应返回201 + 新id"""
    payload = {
        "title": "测试岗位", "company": "测试公司", "city": "深圳",
        "salary_min": 20, "salary_max": 30, "skills": "Python"
    }
    r = client.post("/jobs", json=payload)
    assert r.status_code == 201
    assert "id" in r.json()

def test_create_job_salary_invalid():
    """薪资下限>上限。
    【U5改】跨字段校验从手动raise 400升级为@model_validator，
    pydantic在请求体校验阶段拦截→FastAPI返回422，不再是400"""
    payload = {
        "title": "错岗", "company": "X", "city": "X",
        "salary_min": 50, "salary_max": 10
    }
    r = client.post("/jobs", json=payload)
    assert r.status_code == 422

def test_create_job_missing_title():
    """缺必填字段title 应422"""
    payload = {"company": "X", "city": "X", "salary_min": 1, "salary_max": 2}
    r = client.post("/jobs", json=payload)
    assert r.status_code == 422

# ===== DELETE /jobs/{id} 测试 =====
def test_delete_job_success():
    """先POST一条再DELETE，应返回deleted=1"""
    payload = {"title": "待删", "company": "X", "city": "X",
               "salary_min": 1, "salary_max": 2}
    new_id = client.post("/jobs", json=payload).json()["id"]
    r = client.delete(f"/jobs/{new_id}")
    assert r.status_code == 200
    assert r.json()["deleted"] == 1

def test_delete_nonexistent():
    """DELETE不存在的id 应404"""
    r = client.delete("/jobs/999")
    assert r.status_code == 404

# ===== 改动测试 + semantic语义检索用例 =====

def test_semantic_basic_search():
    """语义检索：正常调用，200返回，验证返回结构、score字段存在"""
    resp = client.get("/jobs/semantic?query=大模型应用开发&top_k=3")
    assert resp.status_code == 200
    data = resp.json()
    assert "count" in data
    assert "jobs" in data
    for job in data["jobs"]:
        assert "id" in job
        assert "title" in job
        assert "score" in job
        # score 0~1之间
        assert 0.0 <= job["score"] <= 1.0


def test_semantic_score_threshold_filter():
    """【改动测试项】测试score_threshold阈值过滤，高于阈值保留，低分过滤"""
    # 设置很高阈值，大部分结果会被过滤掉
    resp = client.get("/jobs/semantic?query=大模型应用开发&top_k=5&score_threshold=0.95")
    assert resp.status_code == 200
    data = resp.json()
    for item in data["jobs"]:
        assert item["score"] >= 0.95


def test_semantic_query_required_422():
    """query是必填查询参数，不传query返回422"""
    resp = client.get("/jobs/semantic?top_k=3")
    assert resp.status_code == 422


def test_jobs_list_limit_param_check():
    """改动测试：/jobs接口 limit边界约束
    limit=0 ge=1 →422；limit=200 le=100 →422
    """
    resp1 = client.get("/jobs?limit=0")
    assert resp1.status_code == 422

    resp2 = client.get("/jobs?limit=200")
    assert resp2.status_code == 422


def test_post_title_minlength_422():
    """改动测试：POST新增岗位 title仅1个字符，min_length=2，预期422"""
    payload = {
        "title": "A",
        "company": "测试公司",
        "city": "广州",
        "salary_min": 10,
        "salary_max": 20
    }
    resp = client.post("/jobs", json=payload)
    assert resp.status_code == 422


def test_post_missing_salary_min_422():
    """改动测试：POST缺少必填字段 salary_min，预期422"""
    payload = {
        "title": "算法工程师",
        "company": "测试公司",
        "city": "广州",
        "salary_max": 30
    }
    resp = client.post("/jobs", json=payload)
    assert resp.status_code == 422
