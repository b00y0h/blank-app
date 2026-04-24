"""Tests for the VLM classifier's pure helpers (no API call)."""

from __future__ import annotations

import chess
import pytest

from chess_analyzer.vision_vlm import _normalise_fen, _guess_media_type


def test_normalise_adds_side_when_missing():
    placement = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
    fen = _normalise_fen(placement, side_to_move="b")
    assert fen.split(" ")[1] == "b"
    chess.Board(fen)  # should not raise


def test_normalise_passes_through_full_fen():
    full = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    assert _normalise_fen(full, side_to_move="w") == full


def test_normalise_strips_code_fences():
    wrapped = "`rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1`"
    out = _normalise_fen(wrapped, side_to_move="w")
    chess.Board(out)


@pytest.mark.parametrize(
    "prefix,expected",
    [
        (b"\x89PNG\r\n\x1a\n", "image/png"),
        (b"\xff\xd8\xff\xe0", "image/jpeg"),
        (b"RIFFabcdWEBP----", "image/webp"),
        (b"BM------", "image/bmp"),
    ],
)
def test_guess_media_type(prefix: bytes, expected: str):
    assert _guess_media_type(prefix + b"\x00" * 16) == expected
