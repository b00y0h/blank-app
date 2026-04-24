"""Tests for the vision pipeline that don't require engine binaries.

We render a synthetic board from a known FEN using the same glyph templates
the detector matches against, then round-trip it through detect_board_fen
and assert the detector recovers the position.
"""

from __future__ import annotations

import io

import chess
import cv2
import numpy as np
import pytest
from PIL import Image, ImageDraw

from chess_analyzer.vision import (
    BOARD_SIZE_PX,
    CELL_SIZE_PX,
    GLYPHS,
    _grid_to_fen,
    _load_font,
    detect_board_fen,
)


def _render_board_png(fen: str, *, flipped: bool = False) -> bytes:
    board = chess.Board(fen)
    img = Image.new("RGB", (BOARD_SIZE_PX, BOARD_SIZE_PX), "white")
    draw = ImageDraw.Draw(img)
    light = (240, 217, 181)
    dark = (181, 136, 99)
    for rank in range(8):
        for file in range(8):
            color = light if (rank + file) % 2 == 0 else dark
            x0 = file * CELL_SIZE_PX
            y0 = rank * CELL_SIZE_PX
            draw.rectangle([x0, y0, x0 + CELL_SIZE_PX, y0 + CELL_SIZE_PX], fill=color)

    font = _load_font(CELL_SIZE_PX)
    # Render both colors with the filled Unicode glyphs (same shape across
    # colors), black pieces in black ink and white pieces in white ink — this
    # mirrors how real chess UIs draw pieces and matches our shape-only
    # templates.
    filled_glyph = {
        "K": GLYPHS["k"], "Q": GLYPHS["q"], "R": GLYPHS["r"],
        "B": GLYPHS["b"], "N": GLYPHS["n"], "P": GLYPHS["p"],
        "k": GLYPHS["k"], "q": GLYPHS["q"], "r": GLYPHS["r"],
        "b": GLYPHS["b"], "n": GLYPHS["n"], "p": GLYPHS["p"],
    }
    for square, piece in board.piece_map().items():
        if flipped:
            f = 7 - chess.square_file(square)
            r = chess.square_rank(square)
        else:
            f = chess.square_file(square)
            r = 7 - chess.square_rank(square)
        glyph = filled_glyph[piece.symbol()]
        x = f * CELL_SIZE_PX + CELL_SIZE_PX // 2
        y = r * CELL_SIZE_PX + CELL_SIZE_PX // 2
        color = (20, 20, 20) if piece.color == chess.BLACK else (250, 250, 250)
        try:
            bbox = draw.textbbox((0, 0), glyph, font=font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            draw.text((x - tw // 2 - bbox[0], y - th // 2 - bbox[1]), glyph,
                      font=font, fill=color)
        except AttributeError:
            tw, th = draw.textsize(glyph, font=font)
            draw.text((x - tw // 2, y - th // 2), glyph, font=font, fill=color)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_grid_to_fen_empty():
    grid = [["." for _ in range(8)] for _ in range(8)]
    assert _grid_to_fen(grid, side_to_move="w").startswith("8/8/8/8/8/8/8/8 w")


def test_grid_to_fen_starting():
    grid = [
        list("rnbqkbnr"),
        list("pppppppp"),
        list("........"),
        list("........"),
        list("........"),
        list("........"),
        list("PPPPPPPP"),
        list("RNBQKBNR"),
    ]
    fen = _grid_to_fen(grid, side_to_move="w")
    placement = fen.split(" ")[0]
    assert placement == "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"


def test_detect_board_starting_position():
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    png = _render_board_png(fen)
    result = detect_board_fen(png, orientation="white", side_to_move="w")
    assert result.fen.split(" ")[0] == fen.split(" ")[0]


def test_detect_handles_black_orientation():
    # A non-symmetric position so orientation handling is actually tested.
    fen = "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 1"
    png = _render_board_png(fen, flipped=True)
    result = detect_board_fen(png, orientation="black", side_to_move="w")
    assert result.fen.split(" ")[0] == fen.split(" ")[0]


def test_detect_rejects_corrupt_image():
    with pytest.raises(ValueError):
        detect_board_fen(b"not an image")
