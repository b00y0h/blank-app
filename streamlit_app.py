"""Streamlit web UI for the chess move analyzer.

Flow:
    1. Upload a screenshot of a chessboard, or paste a FEN.
    2. We detect the board (via the chosen classifier) and show an editable
       FEN for user review.
    3. Pick engine (Stockfish / Lc0), search limit, side to move, orientation,
       and classifier (auto / VLM / templates).
    4. "Find best move" runs analysis. From that point the board is tracked
       in session state — you can keep hitting "Play best move" to alternate
       sides, or "Undo" / "Reset" the game.
"""

from __future__ import annotations

import os

import streamlit as st

# On Streamlit Community Cloud users add ANTHROPIC_API_KEY via the dashboard's
# Secrets UI. We hoist it into the environment before the analyzer imports run
# so the VLM module — which reads from os.environ — picks it up too. Wrapped
# defensively because st.secrets raises if no secrets.toml is configured at
# all (e.g. during local development).
if not os.environ.get("ANTHROPIC_API_KEY"):
    try:
        secret_value = st.secrets.get("ANTHROPIC_API_KEY")  # type: ignore[attr-defined]
    except Exception:
        secret_value = None
    if secret_value:
        os.environ["ANTHROPIC_API_KEY"] = secret_value

import chess  # noqa: E402
import chess.svg  # noqa: E402

from chess_analyzer.analyzer import AnalyzeRequest, analyze_screenshot, play_move  # noqa: E402
from chess_analyzer.engines import EngineUnavailable, available_engines  # noqa: E402
from chess_analyzer.vision import (  # noqa: E402
    _png_template_sets,
    board_image_to_pil,
    detect_board_fen,
)

st.set_page_config(page_title="Chess Move Analyzer", page_icon="♟️", layout="wide")

st.title("♟️ Chess Move Analyzer")
st.caption(
    "Upload a screenshot or paste a FEN. Stockfish or Leela Chess Zero gives "
    "the best move, and you can keep hitting **Play best move** to have the "
    "engine play both sides turn by turn."
)

engines_available = available_engines()
template_sets = sorted(_png_template_sets().keys())
vlm_configured = bool(os.environ.get("ANTHROPIC_API_KEY"))


def _neural_available() -> bool:
    try:
        from chess_analyzer.vision_neural import is_available

        return is_available()
    except Exception:  # noqa: BLE001
        return False


neural_available = _neural_available()


# Status banner: shows which classifiers are actually loaded so it's
# obvious at a glance whether the cloud build picked up the neural extras.
def _status_chip(label: str, ok: bool) -> str:
    icon = "✅" if ok else "❌"
    return f"{icon} **{label}**"


st.markdown(
    " · ".join(
        [
            _status_chip("Neural CNN", neural_available),
            _status_chip("Claude VLM", vlm_configured),
            _status_chip("Stockfish", engines_available.get("stockfish", False)),
            _status_chip("Lc0", engines_available.get("lc0", False)),
        ]
    )
)
if not neural_available and not vlm_configured:
    st.warning(
        "Neither the offline neural classifier nor the Claude VLM is loaded — "
        "detection will fall back to template matching, which is unreliable on "
        "real chess UIs. **Manage app → Reboot app** to pull the latest "
        "requirements (or Settings → Secrets to add `ANTHROPIC_API_KEY`)."
    )


# ---- Session state ---------------------------------------------------------
def _init_state() -> None:
    st.session_state.setdefault("current_fen", None)
    st.session_state.setdefault("history", [])  # list of {san, uci, fen_before, fen_after}
    st.session_state.setdefault("last_engine_result", None)
    st.session_state.setdefault("pending_detected_fen", None)


_init_state()


def _reset_game() -> None:
    st.session_state.current_fen = None
    st.session_state.history = []
    st.session_state.last_engine_result = None
    st.session_state.pending_detected_fen = None


def _undo_last() -> None:
    if st.session_state.history:
        last = st.session_state.history.pop()
        st.session_state.current_fen = last["fen_before"]
        st.session_state.last_engine_result = None


# ---- Sidebar ---------------------------------------------------------------
with st.sidebar:
    st.header("Settings")

    engine_choice = st.radio(
        "Engine",
        ["stockfish", "lc0"],
        format_func=lambda e: {
            "stockfish": "Stockfish" + ("" if engines_available["stockfish"] else " (not installed)"),
            "lc0": "Leela Chess Zero" + ("" if engines_available["lc0"] else " (not installed)"),
        }[e],
        index=0,
    )

    classifier_opts = ["auto", "neural", "vlm", "templates"]
    classifier_labels = {
        "auto": "Auto (neural → VLM → templates)",
        "neural": "Offline CNN (board_to_fen)"
        + ("" if neural_available else " (not installed)"),
        "vlm": "Claude VLM"
        + ("" if vlm_configured else " (ANTHROPIC_API_KEY missing)"),
        "templates": "Templates only",
    }
    classifier_choice = st.radio(
        "Vision classifier",
        classifier_opts,
        format_func=lambda c: classifier_labels[c],
        index=0,
        help=(
            "Auto picks the offline CNN first (highest accuracy on screenshots), "
            "falls back to Claude VLM if an API key is configured, then templates."
        ),
    )

    template_set_choice: str | None = None
    if classifier_choice in ("templates", "auto") and template_sets:
        template_set_choice = st.selectbox(
            "Template set",
            options=["(auto)"] + template_sets,
            index=0,
        )
        template_set_choice = None if template_set_choice == "(auto)" else template_set_choice

    orientation = st.selectbox(
        "Board orientation",
        options=["white", "black"],
        index=0,
        help="Which color is at the *bottom* of the screenshot.",
    )

    side_to_move = st.selectbox(
        "Side to move (initial)",
        options=["w", "b"],
        format_func=lambda s: "White" if s == "w" else "Black",
        index=0,
    )

    search_mode = st.radio("Search limit", ["Time (s)", "Depth"], index=0)
    if search_mode == "Time (s)":
        time_limit = st.slider("Think time (seconds)", 0.1, 10.0, 1.0, 0.1)
        depth = None
    else:
        depth = st.slider("Depth", 1, 30, 15)
        time_limit = 1.0

    multipv = st.slider("Lines (MultiPV)", 1, 5, 1)


# ---- Setup phase (no live game yet) ----------------------------------------
if st.session_state.current_fen is None:
    setup_col, preview_col = st.columns([1, 1])

    with setup_col:
        st.subheader("1. Upload screenshot")
        uploaded = st.file_uploader(
            "PNG, JPG, or WEBP screenshot",
            type=["png", "jpg", "jpeg", "webp", "bmp"],
        )
        manual_fen = st.text_input(
            "…or paste a FEN directly",
            placeholder="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        )

    # Seed the FEN field once. Subsequent updates (new upload / pasted FEN)
    # write directly into st.session_state["fen_input_setup"] so the widget
    # picks them up on the next rerun. Streamlit ignores `value=` for keyed
    # widgets after the first render, which was silently pinning a stale
    # default in place.
    DEFAULT_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    st.session_state.setdefault("fen_input_setup", DEFAULT_FEN)
    st.session_state.setdefault("last_upload_id", None)

    detection_warning: str | None = None

    if uploaded is not None and not manual_fen.strip():
        image_bytes = uploaded.getvalue()
        try:
            # Route through the classifier picker (neural / vlm / templates)
            # rather than templates-only.
            from chess_analyzer.analyzer import _detect_fen as _route_detect

            detected_fen, board_found, classifier_used = _route_detect(
                AnalyzeRequest(
                    image_bytes=image_bytes,
                    classifier=classifier_choice,  # type: ignore[arg-type]
                    orientation=orientation,
                    side_to_move=side_to_move,
                    template_set=template_set_choice,
                )
            )
            # Always also produce a warped preview from the OpenCV pipeline
            # so the user can see what the classifier looked at, regardless
            # of which classifier ran.
            from chess_analyzer.vision import _decode_image, _find_and_warp_board

            warped, _ = _find_and_warp_board(_decode_image(image_bytes))
        except Exception as exc:  # noqa: BLE001 - surfaced to user
            st.error(f"Could not process image: {exc}")
        else:
            new_upload = st.session_state.last_upload_id != uploaded.file_id
            if new_upload:
                st.session_state.fen_input_setup = detected_fen
                st.session_state.last_upload_id = uploaded.file_id
            st.session_state.pending_detected_fen = detected_fen

            piece_count = sum(1 for c in detected_fen.split(" ")[0] if c.isalpha())
            if piece_count < 8:
                fix_hint = ""
                if not neural_available and not vlm_configured:
                    fix_hint = (
                        " Reboot the app from **Manage app → Reboot** so the "
                        "build picks up the neural extras, or set "
                        "`ANTHROPIC_API_KEY` for the VLM path."
                    )
                elif not neural_available:
                    fix_hint = (
                        " The offline neural classifier isn't loaded — "
                        "reboot the app to pick up the latest requirements."
                    )
                detection_warning = (
                    f"Only {piece_count} pieces recognised via **{classifier_used}**. "
                    "Edit the FEN below by hand to correct it." + fix_hint
                )

            with preview_col:
                st.subheader("2. Detected board")
                st.image(board_image_to_pil(warped), caption="Warped board")
                st.caption(f"Classifier used: **{classifier_used}**")
                if not board_found:
                    st.warning("No board rectangle found; used a centered crop.")

    elif manual_fen.strip():
        # Pasted FEN replaces whatever's in the box.
        if st.session_state.fen_input_setup != manual_fen.strip():
            st.session_state.fen_input_setup = manual_fen.strip()

    st.subheader("3. Review FEN")
    if detection_warning:
        st.warning(detection_warning)
    fen_value = st.text_input("FEN", key="fen_input_setup")

    fen_valid = True
    try:
        preview_board = chess.Board(fen_value)
    except ValueError as exc:
        fen_valid = False
        st.error(f"Invalid FEN: {exc}")
        preview_board = None

    if preview_board is not None:
        svg = chess.svg.board(preview_board, size=360, flipped=(orientation == "black"))
        st.markdown(f"<div style='display:flex;justify-content:center'>{svg}</div>",
                    unsafe_allow_html=True)

    if st.button("Start game from this FEN", type="primary", disabled=not fen_valid):
        st.session_state.current_fen = fen_value
        st.session_state.history = []
        st.session_state.last_engine_result = None
        st.rerun()

# ---- Game phase ------------------------------------------------------------
else:
    current_fen = st.session_state.current_fen
    board = chess.Board(current_fen)
    stm_now = "White" if board.turn == chess.WHITE else "Black"

    col_board, col_panel = st.columns([1.2, 1])

    with col_board:
        arrows = []
        er = st.session_state.last_engine_result
        if er and er.get("fen") == current_fen:
            try:
                mv = chess.Move.from_uci(er["best_move_uci"])
                arrows = [chess.svg.Arrow(mv.from_square, mv.to_square, color="#2a7")]
            except (chess.InvalidMoveError, ValueError):
                pass
        svg = chess.svg.board(
            board,
            size=480,
            flipped=(orientation == "black"),
            arrows=arrows,
            lastmove=(
                chess.Move.from_uci(st.session_state.history[-1]["uci"])
                if st.session_state.history else None
            ),
        )
        st.markdown(f"<div style='display:flex;justify-content:center'>{svg}</div>",
                    unsafe_allow_html=True)

    with col_panel:
        st.subheader(f"{stm_now} to move")

        if board.is_game_over(claim_draw=True):
            outcome = board.outcome(claim_draw=True)
            st.success(f"Game over — result: **{outcome.result() if outcome else 'unknown'}**")
            analyze_clicked = False
        else:
            analyze_clicked = st.button(
                "Find best move", type="primary", use_container_width=True
            )

        if analyze_clicked:
            if not engines_available.get(engine_choice, False):
                st.error(f"{engine_choice} not installed on this host.")
            else:
                with st.spinner(f"{engine_choice} is thinking…"):
                    try:
                        resp = analyze_screenshot(
                            AnalyzeRequest(
                                fen=current_fen,
                                engine=engine_choice,  # type: ignore[arg-type]
                                orientation=orientation,
                                side_to_move=(
                                    "w" if board.turn == chess.WHITE else "b"
                                ),
                                time_limit_s=time_limit,
                                depth=depth,
                                multipv=multipv,
                                classifier=classifier_choice,  # type: ignore[arg-type]
                                template_set=template_set_choice,
                            )
                        )
                    except (EngineUnavailable, ValueError) as exc:
                        st.error(str(exc))
                    else:
                        st.session_state.last_engine_result = {
                            "fen": resp.fen,
                            "best_move_uci": resp.engine_result.best_move_uci,
                            "best_move_san": resp.engine_result.best_move_san,
                            "evaluation": resp.engine_result.evaluation,
                            "principal_variation": resp.engine_result.principal_variation,
                            "depth": resp.engine_result.depth,
                            "next_fen": resp.next_fen,
                            "engine": resp.engine_result.engine,
                        }
                        st.rerun()

        er = st.session_state.last_engine_result
        if er and er["fen"] == current_fen:
            st.metric("Best move", er["best_move_san"])
            st.caption(f"{er['engine']} · eval {er['evaluation']}"
                       + (f" · depth {er['depth']}" if er["depth"] else ""))
            if er["principal_variation"]:
                st.caption("PV: " + " ".join(er["principal_variation"]))

            play_col, undo_col, reset_col = st.columns(3)
            if play_col.button("Play best move", type="primary", use_container_width=True):
                try:
                    result = play_move(current_fen, er["best_move_uci"])
                except ValueError as exc:
                    st.error(str(exc))
                else:
                    st.session_state.history.append({
                        "san": result.last_move_san,
                        "uci": result.last_move_uci,
                        "fen_before": current_fen,
                        "fen_after": result.fen,
                    })
                    st.session_state.current_fen = result.fen
                    st.session_state.last_engine_result = None
                    st.rerun()
            if undo_col.button("Undo", use_container_width=True,
                               disabled=not st.session_state.history):
                _undo_last()
                st.rerun()
            if reset_col.button("Reset", use_container_width=True):
                _reset_game()
                st.rerun()
        else:
            undo_col, reset_col = st.columns(2)
            if undo_col.button("Undo", use_container_width=True,
                               disabled=not st.session_state.history):
                _undo_last()
                st.rerun()
            if reset_col.button("Reset", use_container_width=True):
                _reset_game()
                st.rerun()

        st.caption(f"FEN: `{current_fen}`")

    st.subheader("Move history")
    if not st.session_state.history:
        st.write("_No moves yet._")
    else:
        # Render as 1. e4 e5  2. Nf3 Nc6 …
        pgn_parts: list[str] = []
        for i, mv in enumerate(st.session_state.history):
            if i % 2 == 0:
                pgn_parts.append(f"{i // 2 + 1}. {mv['san']}")
            else:
                pgn_parts.append(mv["san"])
        st.code(" ".join(pgn_parts), language="text")

st.divider()
with st.expander("About detection accuracy"):
    st.markdown(
        """
Three classifier modes are available:

- **VLM (Claude)** — a vision-language model reads the board directly.
  Generalizes across any site / theme / camera angle. Needs
  `ANTHROPIC_API_KEY` and network access.
- **Templates** — PNG piece sets under `chess_analyzer/templates/<set>/`.
  Fastest and offline; add sets for each site you care about
  (see that folder's README). Falls back to Unicode-glyph shape templates
  when no PNG sets are present.
- **Auto** — tries VLM first, falls back to templates on failure.

Whichever classifier is used, the detected FEN is editable before you start
the game, and the engine is just a UCI call either way.
        """
    )
