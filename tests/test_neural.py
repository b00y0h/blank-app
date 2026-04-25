"""Tests for the offline neural classifier path.

When board_to_fen / TensorFlow aren't available the tests skip cleanly.
When they are, we render a realistic 2D board via python-chess's SVG
output (cburnett-style pieces) and verify the CNN reads it correctly.
"""

from __future__ import annotations

import io

import chess
import chess.svg
import pytest

from chess_analyzer.analyzer import AnalyzeRequest, _detect_fen


def _has_neural() -> bool:
    try:
        from chess_analyzer.vision_neural import is_available
    except Exception:  # noqa: BLE001
        return False
    return is_available()


def _render_realistic_board(fen: str, *, size: int = 512) -> bytes:
    """Render via python-chess + cairosvg if available; fall back otherwise."""
    cairosvg = pytest.importorskip("cairosvg")
    board = chess.Board(fen)
    svg = chess.svg.board(board, size=size, coordinates=False)
    return cairosvg.svg2png(
        bytestring=svg.encode("utf-8"), output_width=size, output_height=size
    )


@pytest.mark.skipif(not _has_neural(), reason="board_to_fen not installed")
def test_neural_reads_starting_position():
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    png = _render_realistic_board(fen)
    detected, _, label = _detect_fen(
        AnalyzeRequest(image_bytes=png, classifier="neural")
    )
    assert detected.split(" ")[0] == fen.split(" ")[0]
    assert label == "neural"


@pytest.mark.skipif(not _has_neural(), reason="board_to_fen not installed")
def test_neural_reads_midgame_position():
    fen = "r5k1/p2r2pp/2p5/3bB3/P1B1P3/1PK1b3/7P/3R3R b - - 0 27"
    png = _render_realistic_board(fen)
    detected, _, label = _detect_fen(
        AnalyzeRequest(image_bytes=png, classifier="neural")
    )
    assert detected.split(" ")[0] == fen.split(" ")[0]
    assert label == "neural"


def test_classifier_falls_back_to_templates_when_neural_missing():
    """If neural isn't installed, classifier='auto' must still produce a FEN
    via the template path rather than blowing up."""
    if _has_neural():
        pytest.skip("Neural is installed; this fallback path doesn't trigger.")

    cairosvg = pytest.importorskip("cairosvg")
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    board = chess.Board(fen)
    svg = chess.svg.board(board, size=512, coordinates=False)
    png = cairosvg.svg2png(
        bytestring=svg.encode("utf-8"), output_width=512, output_height=512
    )
    detected, _, label = _detect_fen(
        AnalyzeRequest(image_bytes=png, classifier="auto")
    )
    assert isinstance(detected, str) and len(detected) > 0
    assert label.startswith("templates")
