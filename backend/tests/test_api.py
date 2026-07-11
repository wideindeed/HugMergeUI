from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app

FIXTURES = Path(__file__).parent / "fixtures" / "mergekit_examples"
client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_parse_config_slerp():
    yaml_text = (FIXTURES / "gradient-slerp.yml").read_text()
    response = client.post("/parse-config", json={"yaml_text": yaml_text})
    assert response.status_code == 200
    body = response.json()
    assert body["merge_method"] == "slerp"
    assert len(body["layers"]) == 40


def test_parse_config_invalid_yaml_returns_400():
    response = client.post("/parse-config", json={"yaml_text": "not: valid: yaml: ["})
    assert response.status_code == 400


def test_parse_config_linear_missing_num_layers_returns_400():
    yaml_text = (FIXTURES / "linear.yml").read_text()
    response = client.post("/parse-config", json={"yaml_text": yaml_text})
    assert response.status_code == 400


def test_check_architecture_route():
    yaml_text = """
merge_method: linear
models:
  - model: Qwen/Qwen2.5-0.5B
  - model: Qwen/Qwen2.5-0.5B-Instruct
"""
    response = client.post("/check-architecture", json={"yaml_text": yaml_text})
    assert response.status_code == 200
    body = response.json()
    assert body["warnings"] == []
    assert body["num_layers"] > 0


def test_hf_search_route():
    response = client.get("/hf-search", params={"q": "Qwen2.5-1.5B"})
    assert response.status_code == 200
    body = response.json()
    assert isinstance(body, list)
    assert any("Qwen2.5-1.5B" in m for m in body)


def test_hf_search_route_blank_query_returns_empty():
    response = client.get("/hf-search", params={"q": ""})
    assert response.status_code == 200
    assert response.json() == []


def test_model_check_route():
    response = client.get("/model-check", params={"model_id": "Qwen/Qwen2.5-1.5B"})
    assert response.status_code == 200
    body = response.json()
    assert body["model_type"] == "qwen2"
    assert body["validated"] is True


def test_curated_models_route():
    response = client.get("/curated-models")
    assert response.status_code == 200
    body = response.json()
    assert isinstance(body, list)
    assert len(body) > 0
    assert {"id", "model_type", "total_params", "downloads"} <= body[0].keys()


def test_conflict_score_route():
    response = client.post("/conflict-score", json={
        "base_model": "Qwen/Qwen2.5-0.5B",
        "model_a": "Qwen/Qwen2.5-0.5B-Instruct",
        "model_b": "Qwen/Qwen2.5-0.5B-Instruct",
    })
    assert response.status_code == 200
    body = response.json()
    assert len(body["layers"]) > 0
    assert all(layer["conflict"] == 0.0 for layer in body["layers"])


def test_conflict_score_route_unknown_model_returns_400():
    response = client.post("/conflict-score", json={
        "base_model": "this-org/does-not-exist-hugmergeui-test",
        "model_a": "Qwen/Qwen2.5-0.5B-Instruct",
        "model_b": "Qwen/Qwen2.5-0.5B-Instruct",
    })
    assert response.status_code == 400


def test_conflict_score_route_invalid_density_returns_400():
    response = client.post("/conflict-score", json={
        "base_model": "Qwen/Qwen2.5-0.5B",
        "model_a": "Qwen/Qwen2.5-0.5B-Instruct",
        "model_b": "Qwen/Qwen2.5-0.5B-Instruct",
        "density": 1.5,
    })
    assert response.status_code == 400
