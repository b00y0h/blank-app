"""FastAPI backend shared by the web app and the iOS client.

Endpoints:
    GET  /health    — liveness + which engines are installed + which template
                      sets are available + whether the VLM classifier is
                      configured.
    POST /analyze   — multipart: image OR fen, engine, orientation,
                      side_to_move, time_limit_s | depth, classifier,
                      template_set. Returns detected FEN, best move, and the
                      resulting position's FEN for turn-by-turn play.
    POST /play_move — JSON: {fen, move_uci}. Returns the FEN after the move
                      plus game-over state. The server is stateless — clients
                      maintain the current board.

Run locally:
    uvicorn api:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import os
from typing import Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from chess_analyzer.analyzer import (
    AnalyzeRequest,
    ClassifierName,
    analyze_screenshot,
    play_move,
)
from chess_analyzer.engines import EngineUnavailable, available_engines
from chess_analyzer.vision import _png_template_sets

app = FastAPI(title="Chess Move Analyzer", version="1.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, object]:
    try:
        from chess_analyzer.vision_neural import is_available as _neural_avail

        neural_ok = _neural_avail()
    except Exception:  # noqa: BLE001
        neural_ok = False

    return {
        "status": "ok",
        "engines": available_engines(),
        "template_sets": sorted(_png_template_sets().keys()),
        "vlm_configured": bool(os.environ.get("ANTHROPIC_API_KEY")),
        "neural_available": neural_ok,
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
    classifier: Literal["auto", "templates", "vlm", "neural"] = Form("auto"),
    template_set: str | None = Form(None),
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
        classifier=classifier,  # type: ignore[arg-type]
        template_set=template_set,
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
        "next_fen": resp.next_fen,
        "detected_from_image": resp.detected_from_image,
        "board_found": resp.board_found,
        "classifier_used": resp.classifier_used,
        "engine": er.engine,
        "best_move_uci": er.best_move_uci,
        "best_move_san": er.best_move_san,
        "evaluation": er.evaluation,
        "principal_variation": er.principal_variation,
        "depth": er.depth,
        "nodes": er.nodes,
        "time_ms": er.time_ms,
    }


class PlayMoveRequest(BaseModel):
    fen: str = Field(..., description="Position before the move (FEN).")
    move_uci: str = Field(..., description="Move in UCI notation, e.g. e2e4.")


@app.post("/play_move")
def play_move_endpoint(req: PlayMoveRequest) -> dict[str, object]:
    try:
        result = play_move(req.fen, req.move_uci)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "fen": result.fen,
        "last_move_san": result.last_move_san,
        "last_move_uci": result.last_move_uci,
        "is_game_over": result.is_game_over,
        "outcome": result.outcome,
    }
