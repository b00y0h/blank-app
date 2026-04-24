"""Streamlit web UI for the chess move analyzer.

Flow:
    1. Upload a screenshot of a chessboard.
    2. We detect the board, classify pieces, and show an editable FEN.
    3. Pick engine (Stockfish / Lc0), time/depth, side to move, orientation.
    4. Engine returns best move + evaluation + PV.
"""

from __future__ import annotations

import io

import chess
import chess.svg
import streamlit as st

from chess_analyzer.analyzer import AnalyzeRequest, analyze_screenshot
from chess_analyzer.engines import EngineUnavailable, available_engines
from chess_analyzer.vision import board_image_to_pil, detect_board_fen

st.set_page_config(page_title="Chess Move Analyzer", page_icon="♟️", layout="wide")

st.title("♟️ Chess Move Analyzer")
st.caption(
    "Upload a screenshot of a chess position. We'll detect the board, let you "
    "confirm the FEN, and ask Stockfish or Leela Chess Zero for the best move."
)

engines_available = available_engines()


with st.sidebar:
    st.header("Settings")

    engine_options = ["stockfish", "lc0"]
    engine_labels = {
        "stockfish": "Stockfish" + ("" if engines_available["stockfish"] else " (not installed)"),
        "lc0": "Leela Chess Zero" + ("" if engines_available["lc0"] else " (not installed)"),
    }
    engine_choice = st.radio(
        "Engine",
        engine_options,
        format_func=lambda e: engine_labels[e],
        index=0,
    )

    orientation = st.selectbox(
        "Board orientation in screenshot",
        options=["white", "black"],
        index=0,
        help="Which color's pieces appear at the *bottom* of the screenshot?",
    )

    side_to_move = st.selectbox(
        "Side to move",
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


upload_col, preview_col = st.columns([1, 1])

with upload_col:
    st.subheader("1. Upload screenshot")
    uploaded = st.file_uploader(
        "PNG, JPG, or WEBP screenshot of a board",
        type=["png", "jpg", "jpeg", "webp", "bmp"],
        accept_multiple_files=False,
    )
    manual_fen = st.text_input(
        "…or paste a FEN directly",
        value="",
        placeholder="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    )

detected_fen: str | None = None
board_found = False

if uploaded is not None and not manual_fen.strip():
    image_bytes = uploaded.getvalue()
    try:
        det = detect_board_fen(
            image_bytes, orientation=orientation, side_to_move=side_to_move
        )
    except Exception as exc:  # noqa: BLE001 - surfaced to user
        st.error(f"Could not process image: {exc}")
    else:
        detected_fen = det.fen
        board_found = det.found_board
        with preview_col:
            st.subheader("2. Detected board")
            st.image(board_image_to_pil(det.board_image), caption="Warped board")
            if not det.found_board:
                st.warning(
                    "Couldn't find a clear board rectangle, so I cropped a "
                    "centered square. Double-check the detected FEN below."
                )

st.subheader("3. Review FEN")
initial_fen = manual_fen.strip() or detected_fen or "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
fen_value = st.text_input("FEN", value=initial_fen, key="fen_input")

fen_valid = True
try:
    preview_board = chess.Board(fen_value)
except ValueError as exc:
    fen_valid = False
    st.error(f"Invalid FEN: {exc}")
    preview_board = None

if preview_board is not None:
    svg = chess.svg.board(preview_board, size=360, flipped=(orientation == "black"))
    st.markdown(
        f"<div style='display:flex;justify-content:center'>{svg}</div>",
        unsafe_allow_html=True,
    )

st.subheader("4. Analyze")
analyze_clicked = st.button(
    "Find best move",
    type="primary",
    disabled=not fen_valid,
    use_container_width=True,
)

if analyze_clicked and fen_valid:
    if not engines_available.get(engine_choice, False):
        st.error(
            f"{engine_choice} binary isn't available on this host. "
            "See README for install instructions."
        )
    else:
        with st.spinner(f"{engine_choice} is thinking…"):
            try:
                resp = analyze_screenshot(
                    AnalyzeRequest(
                        fen=fen_value,
                        engine=engine_choice,  # type: ignore[arg-type]
                        orientation=orientation,
                        side_to_move=side_to_move,
                        time_limit_s=time_limit,
                        depth=depth,
                        multipv=multipv,
                    )
                )
            except EngineUnavailable as exc:
                st.error(str(exc))
            except ValueError as exc:
                st.error(str(exc))
            else:
                er = resp.engine_result
                c1, c2, c3 = st.columns(3)
                c1.metric("Best move", er.best_move_san)
                c2.metric("Evaluation", er.evaluation)
                c3.metric("Depth", str(er.depth) if er.depth else "—")

                st.caption(f"UCI: `{er.best_move_uci}` • Engine: `{er.engine}`")
                if er.principal_variation:
                    st.write("**Principal variation:** " + " ".join(er.principal_variation))

                # Draw board with the best move arrow
                try:
                    bb = chess.Board(resp.fen)
                    mv = chess.Move.from_uci(er.best_move_uci)
                    arrow_svg = chess.svg.board(
                        bb,
                        size=400,
                        arrows=[chess.svg.Arrow(mv.from_square, mv.to_square, color="#2a7")],
                        flipped=(orientation == "black"),
                    )
                    st.markdown(
                        f"<div style='display:flex;justify-content:center'>{arrow_svg}</div>",
                        unsafe_allow_html=True,
                    )
                except (ValueError, chess.InvalidMoveError):
                    pass

st.divider()
with st.expander("About detection accuracy"):
    st.markdown(
        """
The board detector locates the largest square-ish contour and warps it to an
8×8 grid. Each cell is classified against Unicode chess-glyph templates. That
works well on clean 2D boards (Lichess, chess.com 2D) but can misclassify
fancy 3D sets or low-contrast themes. Always review the detected FEN before
hitting **Find best move** — you can edit it directly.

To analyze without an image at all, paste a FEN in the upload column.
        """
    )
