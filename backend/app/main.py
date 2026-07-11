import json

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from .conflict.engine import ModelWeightsFetchError, score_model_pair, score_model_pair_progress
from .hf.client import search_model_ids
from .hf.service import browse_validated_models, check_architecture, check_model
from .parser.loader import load_config, parse_raw_config

app = FastAPI(title="HugMergeUI API")


class ParseConfigRequest(BaseModel):
    yaml_text: str
    num_layers: int | None = None


class CheckArchitectureRequest(BaseModel):
    yaml_text: str


class ConflictScoreRequest(BaseModel):
    base_model: str
    model_a: str
    model_b: str
    density: float = 0.5


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/parse-config")
def parse_config(request: ParseConfigRequest) -> dict:
    try:
        return load_config(request.yaml_text, num_layers=request.num_layers)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.post("/check-architecture")
def check_architecture_route(request: CheckArchitectureRequest) -> dict:
    try:
        raw = parse_raw_config(request.yaml_text)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    return check_architecture(raw)


@app.get("/hf-search")
def hf_search_route(q: str = "") -> list[str]:
    return search_model_ids(q)


@app.get("/model-check")
def model_check_route(model_id: str) -> dict:
    return check_model(model_id)


@app.get("/curated-models")
def curated_models_route() -> list[dict]:
    return browse_validated_models()


@app.post("/conflict-score")
def conflict_score_route(request: ConflictScoreRequest) -> dict:
    try:
        return score_model_pair(
            request.base_model, request.model_a, request.model_b, density=request.density
        )
    except (ModelWeightsFetchError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.get("/conflict-score-stream")
def conflict_score_stream(base_model: str, model_a: str, model_b: str, density: float = 0.5) -> StreamingResponse:
    def events():
        try:
            for event in score_model_pair_progress(base_model, model_a, model_b, density=density):
                yield f"data: {json.dumps(event)}\n\n"
        except (ModelWeightsFetchError, ValueError) as e:
            yield f"data: {json.dumps({'stage': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")
