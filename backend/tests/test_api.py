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
