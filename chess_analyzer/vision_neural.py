"""Offline CNN-based piece classifier built on `board_to_fen`.

`board_to_fen` (MIT) bundles a small Keras CNN trained on Lichess-style 2D
boards, classifying each of the 64 cells of a tightly-cropped board image
into one of 13 classes (12 pieces + empty). It runs fully offline once
TensorFlow is installed and far outperforms template matching on real
screenshots from chess.com, Lichess, and similar 2D UIs.

Hard install dependency on TensorFlow — heavy, ~500 MB on disk — so the
package is treated as an optional extra: import is lazy and the rest of
the app keeps working without it.

Pipeline:
    1. Decode the user's screenshot.
    2. Use the existing OpenCV pipeline to find and warp the board to a
       fixed-size square (board_to_fen requires a tightly cropped board).
    3. Feed the warped board to board_to_fen's CNN.
    4. Append side-to-move / castling defaults so the result is a valid
       FEN that python-chess can load.
"""

from __future__ import annotations

import os
import sys

# board_to_fen ships a Keras-2 SavedModel. With TensorFlow >= 2.16 the
# default Keras package is v3, which can't load that legacy artifact. We
# monkey-patch `sys.modules["keras"]` to the Keras-2 backport (`tf_keras`)
# before board_to_fen does its `from keras import models, layers`, so the
# legacy model loads regardless of which TF is installed. Also flip the
# documented env switch for code paths that go through `tf.keras` directly.
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
try:
    import tf_keras as _tf_keras  # type: ignore[import-not-found]

    sys.modules.setdefault("keras", _tf_keras)
except ImportError:
    pass

import chess
import cv2
from PIL import Image

from .vision import _decode_image, _find_and_warp_board


class NeuralUnavailable(RuntimeError):
    """Raised when board_to_fen / TensorFlow can't be loaded."""


def detect_fen_via_cnn(
    image_bytes: bytes,
    *,
    orientation: str = "white",
    side_to_move: str = "w",
) -> tuple[str, bool]:
    """Run the offline CNN and return (fen, board_found_flag).

    `orientation` is forwarded to board_to_fen as its `black_view` argument.
    """
    try:
        from board_to_fen.predict import get_fen_from_image
    except ImportError as exc:
        raise NeuralUnavailable(
            "board_to_fen / TensorFlow not installed. Install the neural "
            "extras: `pip install -r requirements-neural.txt` (or "
            "`pip install board_to_fen tensorflow==2.15.* keras==2.15.*`)."
        ) from exc
    except Exception as exc:  # noqa: BLE001 - import-time failures are noisy
        raise NeuralUnavailable(
            f"Could not load board_to_fen ({exc}). See README for the "
            "matched TensorFlow / Keras versions."
        ) from exc

    img = _decode_image(image_bytes)
    warped, found = _find_and_warp_board(img)
    pil = Image.fromarray(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))

    try:
        placement = get_fen_from_image(pil, black_view=(orientation == "black"))
    except Exception as exc:  # noqa: BLE001 - any failure -> caller falls back
        raise NeuralUnavailable(f"board_to_fen inference failed: {exc}") from exc

    # board_to_fen returns just the placement field. Build a full FEN, then
    # validate; an invalid result raises so the analyzer can fall back.
    fen = f"{placement} {side_to_move} - - 0 1"
    try:
        chess.Board(fen)
    except ValueError as exc:
        raise NeuralUnavailable(
            f"board_to_fen returned an invalid placement: {placement!r} ({exc})"
        ) from exc
    return fen, found


def is_available() -> bool:
    """Lightweight probe — returns True only if board_to_fen imports cleanly."""
    try:
        import board_to_fen.predict  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return True
