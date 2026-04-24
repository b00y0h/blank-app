"""Engine-wrapper tests. Skipped cleanly when binaries aren't installed."""

from __future__ import annotations

import pytest

from chess_analyzer.engines import (
    EngineUnavailable,
    analyze_fen,
    available_engines,
)


def test_available_engines_returns_both_keys():
    avail = available_engines()
    assert set(avail.keys()) == {"stockfish", "lc0"}
    for v in avail.values():
        assert isinstance(v, bool)


def test_analyze_fen_rejects_invalid_fen():
    with pytest.raises(ValueError):
        analyze_fen("not a fen", engine="stockfish")


def test_analyze_fen_raises_when_engine_missing():
    """When neither binary is available, analyze should raise EngineUnavailable
    rather than a cryptic FileNotFoundError."""
    if any(available_engines().values()):
        pytest.skip("An engine is installed; nothing to assert here.")
    with pytest.raises(EngineUnavailable):
        analyze_fen(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            engine="stockfish",
        )


@pytest.mark.skipif(
    not available_engines().get("stockfish"),
    reason="Stockfish binary not installed.",
)
def test_stockfish_finds_move_in_opening():
    result = analyze_fen(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        engine="stockfish",
        time_limit_s=0.3,
    )
    assert len(result.best_move_uci) in (4, 5)
    assert result.engine == "stockfish"
