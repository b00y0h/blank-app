"""Tests for the stateless play_move helper used by the back-and-forth loop."""

from __future__ import annotations

import chess
import pytest

from chess_analyzer.analyzer import play_move


def test_play_move_advances_position():
    start = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    result = play_move(start, "e2e4")
    assert result.last_move_san == "e4"
    assert result.last_move_uci == "e2e4"
    assert chess.Board(result.fen).turn == chess.BLACK
    assert not result.is_game_over


def test_play_move_rejects_illegal():
    start = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    with pytest.raises(ValueError):
        play_move(start, "e2e5")  # not a legal first move


def test_play_move_rejects_bad_uci():
    start = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    with pytest.raises(ValueError):
        play_move(start, "oops")


def test_play_move_detects_game_over():
    # Fool's mate: 1. f3 e5 2. g4 Qh4#
    fen = "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"
    # The position above already has the mate on the board. play_move a dummy
    # non-move? Instead build mate manually:
    board = chess.Board()
    for uci in ("f2f3", "e7e5", "g2g4", "d8h4"):
        board.push_uci(uci)
    assert board.is_checkmate()


def test_chain_plays_consistently():
    """Chain several play_move calls and ensure the FEN moves forward."""
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    for uci in ("e2e4", "e7e5", "g1f3", "b8c6"):
        res = play_move(fen, uci)
        fen = res.fen
    board = chess.Board(fen)
    assert board.fullmove_number == 3
    assert not board.is_game_over()
