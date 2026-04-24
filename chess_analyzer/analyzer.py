"""Top-level glue: screenshot -> detected FEN -> engine -> best move."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

from .engines import EngineName, EngineResult, analyze_fen
from .vision import BoardDetectionResult, detect_board_fen


@dataclass
class AnalyzeRequest:
    image_bytes: bytes | None = None  # one of image_bytes or fen must be set
    fen: str | None = None
    engine: EngineName = "stockfish"
    orientation: str = "white"  # "white" or "black" at the bottom of the image
    side_to_move: str = "w"  # "w" or "b"
    time_limit_s: float = 1.0
    depth: int | None = None
    multipv: int = 1


@dataclass
class AnalyzeResponse:
    fen: str
    detected_from_image: bool
    engine_result: EngineResult
    board_found: bool = False

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # EngineResult is already a dataclass, asdict handles it.
        return d


def analyze_screenshot(req: AnalyzeRequest) -> AnalyzeResponse:
    if req.fen:
        fen = req.fen
        board_found = True
        detected = False
    else:
        if not req.image_bytes:
            raise ValueError("AnalyzeRequest needs either fen or image_bytes.")
        det: BoardDetectionResult = detect_board_fen(
            req.image_bytes,
            orientation=req.orientation,
            side_to_move=req.side_to_move,
        )
        fen = det.fen
        board_found = det.found_board
        detected = True

    engine_result = analyze_fen(
        fen,
        engine=req.engine,
        time_limit_s=req.time_limit_s,
        depth=req.depth,
        multipv=req.multipv,
    )
    return AnalyzeResponse(
        fen=fen,
        detected_from_image=detected,
        engine_result=engine_result,
        board_found=board_found,
    )
