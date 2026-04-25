"""UCI engine wrappers for Stockfish and Leela Chess Zero.

We shell out to the engine binaries via python-chess's UCI bridge. The host
must have the binaries installed (see README for install instructions); at
call time we check availability and raise EngineUnavailable with a friendly
message so the UI can surface it instead of crashing.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from typing import Literal

import chess
import chess.engine

EngineName = Literal["stockfish", "lc0"]


class EngineUnavailable(RuntimeError):
    """Raised when the chosen engine binary can't be found or launched."""


@dataclass
class EngineResult:
    engine: EngineName
    best_move_uci: str
    best_move_san: str
    evaluation: str  # e.g. "+0.45", "-#3"
    principal_variation: list[str]  # SAN
    depth: int | None
    nodes: int | None
    time_ms: int | None


DEFAULT_BINARIES: dict[EngineName, str] = {
    "stockfish": os.environ.get("STOCKFISH_PATH", "stockfish"),
    "lc0": os.environ.get("LC0_PATH", "lc0"),
}

# Fallback locations checked when `shutil.which` doesn't find the binary on
# PATH. /usr/games/* is where Debian's apt packages land for stockfish/lc0,
# and that directory isn't always on PATH for non-login subprocesses (e.g.
# Streamlit Community Cloud). The other paths cover Homebrew and common
# Linux install prefixes.
_FALLBACK_PATHS: dict[EngineName, tuple[str, ...]] = {
    "stockfish": (
        "/usr/games/stockfish",
        "/usr/bin/stockfish",
        "/usr/local/bin/stockfish",
        "/opt/homebrew/bin/stockfish",
        "/snap/bin/stockfish",
    ),
    "lc0": (
        "/usr/games/lc0",
        "/usr/bin/lc0",
        "/usr/local/bin/lc0",
        "/opt/homebrew/bin/lc0",
    ),
}


def _resolve_binary(engine: EngineName) -> str:
    path = DEFAULT_BINARIES[engine]
    if os.path.isabs(path):
        if os.path.exists(path):
            return path
    else:
        which = shutil.which(path)
        if which:
            return which
        for candidate in _FALLBACK_PATHS[engine]:
            if os.path.exists(candidate) and os.access(candidate, os.X_OK):
                return candidate
    env_var = "STOCKFISH_PATH" if engine == "stockfish" else "LC0_PATH"
    raise EngineUnavailable(
        f"{engine} binary not found on PATH or in common install locations. "
        f"Install it (apt-get install {engine}) or set {env_var} to its "
        "absolute path."
    )


def analyze_fen(
    fen: str,
    *,
    engine: EngineName = "stockfish",
    time_limit_s: float = 1.0,
    depth: int | None = None,
    multipv: int = 1,
    threads: int | None = None,
    hash_mb: int | None = None,
) -> EngineResult:
    """Analyze a position and return the engine's best move + evaluation.

    Exactly one of time_limit_s or depth governs the search: depth wins if set.
    """
    try:
        board = chess.Board(fen)
    except ValueError as exc:
        raise ValueError(f"Invalid FEN: {exc}") from exc

    if board.is_game_over(claim_draw=True):
        outcome = board.outcome(claim_draw=True)
        raise ValueError(
            f"Position is terminal ({outcome.result() if outcome else 'unknown'}); "
            "no move to analyze."
        )

    binary = _resolve_binary(engine)

    try:
        transport = chess.engine.SimpleEngine.popen_uci(binary)
    except (FileNotFoundError, OSError) as exc:
        raise EngineUnavailable(f"Failed to launch {engine}: {exc}") from exc

    try:
        options: dict[str, object] = {}
        if threads is not None:
            options["Threads"] = threads
        if hash_mb is not None:
            options["Hash"] = hash_mb
        if options:
            try:
                transport.configure(options)
            except chess.engine.EngineError:
                # Some engines/binaries reject unknown options; ignore.
                pass

        limit = (
            chess.engine.Limit(depth=depth)
            if depth is not None
            else chess.engine.Limit(time=time_limit_s)
        )
        info = transport.analyse(board, limit, multipv=multipv)
        top = info[0] if isinstance(info, list) else info
        pv_moves: list[chess.Move] = list(top.get("pv") or [])
        if not pv_moves:
            # Fallback to a play() call for a guaranteed best move.
            play_result = transport.play(board, limit)
            if play_result.move is None:
                raise RuntimeError("Engine returned no move.")
            pv_moves = [play_result.move]

        best_move = pv_moves[0]
        score = top.get("score")
        eval_str = _format_score(score, board.turn)
        pv_san: list[str] = []
        tmp = board.copy(stack=False)
        for mv in pv_moves:
            try:
                pv_san.append(tmp.san(mv))
                tmp.push(mv)
            except (chess.IllegalMoveError, AssertionError):
                break

        return EngineResult(
            engine=engine,
            best_move_uci=best_move.uci(),
            best_move_san=board.san(best_move),
            evaluation=eval_str,
            principal_variation=pv_san,
            depth=top.get("depth"),
            nodes=top.get("nodes"),
            time_ms=int(top.get("time", 0) * 1000) if top.get("time") else None,
        )
    finally:
        try:
            transport.quit()
        except chess.engine.EngineError:
            transport.close()


def _format_score(
    score: chess.engine.PovScore | None, side_to_move: chess.Color
) -> str:
    if score is None:
        return "0.00"
    pov = score.white() if side_to_move == chess.WHITE else score.black()
    # We always report from the perspective of the side to move, positive good.
    pov_side = score.pov(side_to_move)
    if pov_side.is_mate():
        mate_in = pov_side.mate()
        sign = "+" if (mate_in or 0) > 0 else "-"
        return f"{sign}M{abs(mate_in) if mate_in else 0}"
    cp = pov_side.score(mate_score=100000)
    return f"{cp / 100:+.2f}"


def available_engines() -> dict[EngineName, bool]:
    """Return which engines are currently available on the host."""
    result: dict[EngineName, bool] = {}
    for name in ("stockfish", "lc0"):
        try:
            _resolve_binary(name)  # type: ignore[arg-type]
            result[name] = True  # type: ignore[index]
        except EngineUnavailable:
            result[name] = False  # type: ignore[index]
    return result
