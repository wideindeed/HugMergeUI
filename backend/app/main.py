from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .hf.service import check_architecture
from .parser.loader import load_config, parse_raw_config

app = FastAPI(title="HugMergeUI API")


class ParseConfigRequest(BaseModel):
    yaml_text: str
    num_layers: int | None = None


class CheckArchitectureRequest(BaseModel):
    yaml_text: str


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
