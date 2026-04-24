"""Top-level glue: screenshot -> detected FEN -> engine -> best move.

Also exposes a stateless `play_move` helper that advances a position by one
move. UIs keep their own board state and use the helper to step forward,
supporting the play-both-sides loop.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import chess

from .engines import EngineName, EngineResult, analyze_fen
from .vision import BoardDetectionResult, detect_board_fen

ClassifierName = Literal["auto", "templates", "vlm"]


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
    classifier: ClassifierName = "auto"
    template_set: str | None = None  # e.g. "chesscom", "lichess"


@dataclass
class AnalyzeResponse:
    fen: str
    next_fen: str  # position after playing best_move; useful for back-and-forth
    detected_from_image: bool
    engine_result: EngineResult
    board_found: bool = False
    classifier_used: str = "templates"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PlayMoveResult:
    fen: str
    last_move_san: str
    last_move_uci: str
    is_game_over: bool
    outcome: str | None  # e.g. "1-0", "0-1", "1/2-1/2" when game_over


def analyze_screenshot(req: AnalyzeRequest) -> AnalyzeResponse:
    classifier_used = "fen"
    if req.fen:
        fen = req.fen
        board_found = True
        detected = False
    else:
        if not req.image_bytes:
            raise ValueError("AnalyzeRequest needs either fen or image_bytes.")
        fen, board_found, classifier_used = _detect_fen(req)
        detected = True

    engine_result = analyze_fen(
        fen,
        engine=req.engine,
        time_limit_s=req.time_limit_s,
        depth=req.depth,
        multipv=req.multipv,
    )

    next_fen = _compute_next_fen(fen, engine_result.best_move_uci)

    return AnalyzeResponse(
        fen=fen,
        next_fen=next_fen,
        detected_from_image=detected,
        engine_result=engine_result,
        board_found=board_found,
        classifier_used=classifier_used,
    )


def play_move(fen: str, move_uci: str) -> PlayMoveResult:
    """Advance a position by one UCI move. Raises ValueError on illegality."""
    try:
        board = chess.Board(fen)
    except ValueError as exc:
        raise ValueError(f"Invalid FEN: {exc}") from exc

    try:
        move = chess.Move.from_uci(move_uci)
    except chess.InvalidMoveError as exc:
        raise ValueError(f"Invalid UCI move: {exc}") from exc

    if move not in board.legal_moves:
        raise ValueError(f"Illegal move for this position: {move_uci}")

    san = board.san(move)
    board.push(move)
    outcome = board.outcome(claim_draw=True)
    return PlayMoveResult(
        fen=board.fen(),
        last_move_san=san,
        last_move_uci=move.uci(),
        is_game_over=outcome is not None,
        outcome=outcome.result() if outcome else None,
    )


def _compute_next_fen(fen: str, best_move_uci: str) -> str:
    """Return the FEN after playing best_move_uci, or the original on failure."""
    try:
        board = chess.Board(fen)
        move = chess.Move.from_uci(best_move_uci)
        if move not in board.legal_moves:
            return fen
        board.push(move)
        return board.fen()
    except (ValueError, chess.InvalidMoveError):
        return fen


def _detect_fen(req: AnalyzeRequest) -> tuple[str, bool, str]:
    """Run the configured classifier(s) and return (fen, board_found, label)."""
    assert req.image_bytes is not None

    if req.classifier in ("vlm", "auto"):
        try:
            from .vision_vlm import detect_fen_via_claude, VLMUnavailable
            try:
                fen = detect_fen_via_claude(
                    req.image_bytes,
                    orientation=req.orientation,
                    side_to_move=req.side_to_move,
                )
                return fen, True, "vlm"
            except VLMUnavailable:
                if req.classifier == "vlm":
                    raise
                # "auto" falls through to templates
        except ImportError:
            if req.classifier == "vlm":
                raise

    det: BoardDetectionResult = detect_board_fen(
        req.image_bytes,
        orientation=req.orientation,
        side_to_move=req.side_to_move,
        template_set=req.template_set,
    )
    return det.fen, det.found_board, f"templates:{det.template_set_used or 'unicode'}"
