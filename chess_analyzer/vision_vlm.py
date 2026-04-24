"""Claude-based FEN extraction from an arbitrary board screenshot.

This is the bulletproof-across-sites path: a vision-language model reads the
board directly, so chess.com, Lichess, 3D boards, camera photos, and custom
themes all route through the same extractor without bespoke templates.

Requires:
    - `ANTHROPIC_API_KEY` in the environment.
    - `anthropic` Python package (see requirements.txt).

This module is imported lazily by analyzer.py so the rest of the app still
works when the API key / SDK isn't available.
"""

from __future__ import annotations

import base64
import os
import re

import chess


class VLMUnavailable(RuntimeError):
    """Raised when the Claude VLM path can't be used (no key / SDK / network)."""


# Claude 4.x vision is excellent for this task. Sonnet 4.6 is a good cost/quality
# default; Opus 4.7 is available for tougher positions.
_DEFAULT_MODEL = os.environ.get("CHESS_VLM_MODEL", "claude-sonnet-4-6")


_SYSTEM_PROMPT = (
    "You are a chess position reader. You will be given a single image of a "
    "chessboard (from any site, theme, or angle). Identify every piece on "
    "every square and return ONLY a valid FEN placement string — the piece-"
    "placement field followed by side-to-move, castling, en-passant, halfmove "
    "clock, and fullmove number. Do not include any other text, commentary, "
    "or markdown.\n\n"
    "Rules:\n"
    "- Output exactly one FEN and nothing else.\n"
    "- Use '-' for fields you cannot infer (castling, en-passant).\n"
    "- If the image does not contain a clearly identifiable chessboard, "
    "respond with the literal string UNKNOWN."
)


def detect_fen_via_claude(
    image_bytes: bytes,
    *,
    orientation: str = "white",
    side_to_move: str = "w",
    model: str | None = None,
) -> str:
    """Ask Claude to read the board image and return a FEN.

    The orientation and side_to_move hints are passed in so Claude doesn't
    have to guess from the image alone. The returned FEN is validated with
    python-chess before being returned — invalid output raises VLMUnavailable.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise VLMUnavailable(
            "ANTHROPIC_API_KEY is not set; VLM classifier unavailable."
        )

    try:
        from anthropic import Anthropic, APIError  # lazy import
    except ImportError as exc:
        raise VLMUnavailable(
            "anthropic SDK not installed; run `pip install anthropic`."
        ) from exc

    client = Anthropic(api_key=api_key)
    b64 = base64.standard_b64encode(image_bytes).decode("ascii")
    media_type = _guess_media_type(image_bytes)

    user_text = (
        f"The board is shown with {orientation} at the bottom. "
        f"Side to move is {'White' if side_to_move == 'w' else 'Black'}."
    )

    try:
        response = client.messages.create(
            model=model or _DEFAULT_MODEL,
            max_tokens=200,
            system=_SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": b64,
                            },
                        },
                        {"type": "text", "text": user_text},
                    ],
                }
            ],
        )
    except APIError as exc:
        raise VLMUnavailable(f"Claude API error: {exc}") from exc

    text = _first_text_block(response).strip()
    if text.upper() == "UNKNOWN":
        raise VLMUnavailable("Model did not recognise a board in the image.")

    fen = _normalise_fen(text, side_to_move=side_to_move)
    try:
        chess.Board(fen)
    except ValueError as exc:
        raise VLMUnavailable(f"Model returned an invalid FEN: {text!r} ({exc})") from exc
    return fen


def _first_text_block(response) -> str:
    for block in response.content:
        if getattr(block, "type", None) == "text":
            return block.text
    raise VLMUnavailable("Model response had no text block.")


def _normalise_fen(text: str, *, side_to_move: str) -> str:
    """Pull out a FEN-shaped token from the model's response and fill in the
    side-to-move if the model omitted it."""
    # Strip code fences / backticks if the model insisted on them.
    text = text.strip().strip("`").strip()
    # If the model returned only the placement field, append the rest.
    if re.fullmatch(r"[rnbqkpRNBQKP1-8/]+", text):
        return f"{text} {side_to_move} - - 0 1"
    parts = text.split()
    if len(parts) >= 1:
        placement = parts[0]
        stm = parts[1] if len(parts) >= 2 else side_to_move
        castling = parts[2] if len(parts) >= 3 else "-"
        ep = parts[3] if len(parts) >= 4 else "-"
        half = parts[4] if len(parts) >= 5 else "0"
        full = parts[5] if len(parts) >= 6 else "1"
        return f"{placement} {stm} {castling} {ep} {half} {full}"
    return text


def _guess_media_type(image_bytes: bytes) -> str:
    if image_bytes[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if image_bytes[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if image_bytes[:4] == b"RIFF" and image_bytes[8:12] == b"WEBP":
        return "image/webp"
    if image_bytes[:2] == b"BM":
        return "image/bmp"
    return "image/png"  # reasonable default
