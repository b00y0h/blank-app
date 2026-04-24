"""FastAPI backend shared by the web app and the iOS client.

Endpoints:
    GET  /health    — liveness + which engines are installed.
    POST /analyze   — multipart: image OR fen, engine, orientation, side_to_move,
                       time_limit_s | depth. Returns detected FEN + best move.

Run locally:
    uvicorn api:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

from typing import Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from chess_analyzer.analyzer import AnalyzeRequest, analyze_screenshot
from chess_analyzer.engines import (
    EngineName,
    EngineUnavailable,
    available_engines,
)

app = FastAPI(title="Chess Move Analyzer", version="1.0.0")

# The iOS app and local web clients need cross-origin access during dev. Tighten
# allow_origins for production deployments.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "engines": available_engines(),
    }


@app.post("/analyze")
async def analyze(
    image: UploadFile | None = File(None),
    fen: str | None = Form(None),
    engine: Literal["stockfish", "lc0"] = Form("stockfish"),
    orientation: Literal["white", "black"] = Form("white"),
    side_to_move: Literal["w", "b"] = Form("w"),
    time_limit_s: float = Form(1.0),
    depth: int | None = Form(None),
    multipv: int = Form(1),
) -> dict[str, object]:
    if image is None and not fen:
        raise HTTPException(
            status_code=400, detail="Provide either an image file or a FEN."
        )

    image_bytes: bytes | None = None
    if image is not None:
        image_bytes = await image.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Uploaded image is empty.")

    req = AnalyzeRequest(
        image_bytes=image_bytes,
        fen=fen,
        engine=engine,  # type: ignore[arg-type]
        orientation=orientation,
        side_to_move=side_to_move,
        time_limit_s=time_limit_s,
        depth=depth,
        multipv=multipv,
    )

    try:
        resp = analyze_screenshot(req)
    except EngineUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    er = resp.engine_result
    return {
        "fen": resp.fen,
        "detected_from_image": resp.detected_from_image,
        "board_found": resp.board_found,
        "engine": er.engine,
        "best_move_uci": er.best_move_uci,
        "best_move_san": er.best_move_san,
        "evaluation": er.evaluation,
        "principal_variation": er.principal_variation,
        "depth": er.depth,
        "nodes": er.nodes,
        "time_ms": er.time_ms,
    }
