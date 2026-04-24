"""Chess board detection and piece recognition from a screenshot.

The pipeline:
    1. Find the board quadrilateral in the image (largest near-square contour,
       with a full-image fallback).
    2. Warp it to a fixed 8x8 grid.
    3. For each of the 64 cells: decide empty vs occupied via local contrast,
       then pick the best-matching template (12 piece classes) by normalized
       cross-correlation on a silhouette of the piece.
    4. Assemble a FEN from the grid.

Template matching against Unicode chess glyphs is a reasonable default that
works on screenshots where pieces are drawn as solid, recognisable silhouettes
(Lichess, chess.com 2D, most 2D boards). Complex 3D sets may fool it, so the
UI always exposes the detected FEN for manual correction before engine
analysis.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Iterable

import chess
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

BOARD_SIZE_PX = 512
CELL_SIZE_PX = BOARD_SIZE_PX // 8

# Unicode chess glyphs. Uppercase = white, lowercase = black in FEN.
GLYPHS: dict[str, str] = {
    "K": "♔", "Q": "♕", "R": "♖",
    "B": "♗", "N": "♘", "P": "♙",
    "k": "♚", "q": "♛", "r": "♜",
    "b": "♝", "n": "♞", "p": "♟",
}

# Templates used for shape-only matching. Unicode "white" glyphs are outlined
# while "black" glyphs are filled, so their silhouettes don't share a shape
# budget. We anchor shape matching on the filled glyphs only, then separately
# decide piece color by comparing the piece's mean luminance to its square's
# background luminance.
SHAPE_TEMPLATE_PIECES: tuple[str, ...] = ("k", "q", "r", "b", "n", "p")


@dataclass
class BoardDetectionResult:
    fen: str
    board_image: np.ndarray  # warped BGR board (BOARD_SIZE_PX square)
    cell_confidences: list[float] = field(default_factory=list)
    found_board: bool = False


def detect_board_fen(
    image_bytes: bytes,
    *,
    orientation: str = "white",  # "white" = white pieces at bottom of image
    side_to_move: str = "w",
) -> BoardDetectionResult:
    """Detect chessboard in image_bytes and return an inferred FEN."""
    if orientation not in {"white", "black"}:
        raise ValueError("orientation must be 'white' or 'black'")
    if side_to_move not in {"w", "b"}:
        raise ValueError("side_to_move must be 'w' or 'b'")

    img = _decode_image(image_bytes)
    warped, found = _find_and_warp_board(img)

    grid, confidences = _classify_cells(warped)
    if orientation == "black":
        grid = [row[::-1] for row in grid[::-1]]

    fen = _grid_to_fen(grid, side_to_move=side_to_move)
    return BoardDetectionResult(
        fen=fen,
        board_image=warped,
        cell_confidences=confidences,
        found_board=found,
    )


def _decode_image(image_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image; unsupported format or corrupt data.")
    return img


def _find_and_warp_board(img: np.ndarray) -> tuple[np.ndarray, bool]:
    """Return (warped_board, found_bool). Falls back to cropping a centered
    square from the image when no quadrilateral is found."""
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_quad: np.ndarray | None = None
    best_area = 0.0
    image_area = float(h * w)

    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) != 4:
            continue
        area = cv2.contourArea(approx)
        if area < 0.15 * image_area:
            continue
        x, y, bw, bh = cv2.boundingRect(approx)
        if bh == 0:
            continue
        aspect = bw / bh
        if aspect < 0.85 or aspect > 1.15:
            continue
        # Reject rotated quads (diamonds formed by the checkerboard's diagonal
        # edges fool us otherwise). A true board quad fills most of its
        # axis-aligned bounding rect.
        if area < 0.80 * (bw * bh):
            continue
        if area > best_area:
            best_area = area
            best_quad = approx.reshape(4, 2).astype(np.float32)

    if best_quad is None:
        # Fallback: centered square crop covering the smaller dimension
        side = min(h, w)
        y0 = (h - side) // 2
        x0 = (w - side) // 2
        crop = img[y0 : y0 + side, x0 : x0 + side]
        warped = cv2.resize(crop, (BOARD_SIZE_PX, BOARD_SIZE_PX))
        return warped, False

    ordered = _order_quad(best_quad)
    dst = np.array(
        [
            [0, 0],
            [BOARD_SIZE_PX - 1, 0],
            [BOARD_SIZE_PX - 1, BOARD_SIZE_PX - 1],
            [0, BOARD_SIZE_PX - 1],
        ],
        dtype=np.float32,
    )
    M = cv2.getPerspectiveTransform(ordered, dst)
    warped = cv2.warpPerspective(img, M, (BOARD_SIZE_PX, BOARD_SIZE_PX))
    return warped, True


def _order_quad(pts: np.ndarray) -> np.ndarray:
    """Return quad points ordered TL, TR, BR, BL."""
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    return np.stack([tl, tr, br, bl]).astype(np.float32)


def _classify_cells(board_img: np.ndarray) -> tuple[list[list[str]], list[float]]:
    """Return an 8x8 grid of piece codes ('.' for empty) + per-cell confidence.

    Strategy: for each cell we build a silhouette (Otsu threshold, minority =
    piece), use its fill ratio for occupancy, match shape against six filled
    glyph templates to name the piece type, then determine color by comparing
    the piece mean to the square background mean.
    """
    gray = cv2.cvtColor(board_img, cv2.COLOR_BGR2GRAY)
    shape_templates = _shape_templates()
    grid: list[list[str]] = []
    confs: list[float] = []

    # Pieces cover an intentionally wide band because rendered glyphs vary.
    # The lower bound excludes cells where Otsu fires on rendering noise.
    min_fill = 0.05
    max_fill = 0.80

    for r in range(8):
        row: list[str] = []
        for f in range(8):
            cell = _extract_cell(gray, r, f)
            silhouette = _cell_to_silhouette(cell)
            fill_ratio = float(cv2.countNonZero(silhouette)) / float(silhouette.size)
            if not (min_fill <= fill_ratio <= max_fill):
                row.append(".")
                confs.append(0.0)
                continue

            piece_type, score = _best_shape_match(silhouette, shape_templates)
            if piece_type == ".":
                row.append(".")
                confs.append(score)
                continue

            piece_mean = float(cell[silhouette > 0].mean())
            bg_mean = float(cell[silhouette == 0].mean()) if (silhouette == 0).any() else 128.0
            is_white_piece = piece_mean > bg_mean

            row.append(piece_type.upper() if is_white_piece else piece_type.lower())
            confs.append(score)
        grid.append(row)
    return grid, confs


def _extract_cell(gray: np.ndarray, row: int, file: int) -> np.ndarray:
    y0 = row * CELL_SIZE_PX
    x0 = file * CELL_SIZE_PX
    return gray[y0 : y0 + CELL_SIZE_PX, x0 : x0 + CELL_SIZE_PX].copy()


def _cell_to_silhouette(cell: np.ndarray) -> np.ndarray:
    """Produce a 0/255 mask of the piece in a cell, agnostic to square color."""
    # Otsu splits bg/fg; we don't know which side is the piece, so we use the
    # minority-pixel class as the piece.
    _, th = cv2.threshold(cell, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if cv2.countNonZero(th) > th.size // 2:
        th = cv2.bitwise_not(th)
    # Clean small noise
    kernel = np.ones((3, 3), np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)
    return th


def _best_shape_match(
    silhouette: np.ndarray, templates: dict[str, np.ndarray]
) -> tuple[str, float]:
    best_piece = "."
    best_score = -1.0
    for piece, tmpl in templates.items():
        res = cv2.matchTemplate(silhouette, tmpl, cv2.TM_CCOEFF_NORMED)
        score = float(res.max())
        if score > best_score:
            best_score = score
            best_piece = piece
    if best_score < 0.15:
        return ".", best_score
    return best_piece, best_score


def _grid_to_fen(grid: list[list[str]], *, side_to_move: str) -> str:
    rows: list[str] = []
    for r in range(8):
        run = 0
        out: list[str] = []
        for f in range(8):
            cell = grid[r][f]
            if cell == ".":
                run += 1
                continue
            if run:
                out.append(str(run))
                run = 0
            out.append(cell)
        if run:
            out.append(str(run))
        rows.append("".join(out))

    placement = "/".join(rows)
    fen = f"{placement} {side_to_move} - - 0 1"

    # Validate: if python-chess rejects the FEN, return an empty board so the
    # caller / UI can prompt the user to correct it rather than crashing.
    try:
        chess.Board(fen)
    except ValueError:
        fen = f"8/8/8/8/8/8/8/8 {side_to_move} - - 0 1"
    return fen


@lru_cache(maxsize=1)
def _shape_templates() -> dict[str, np.ndarray]:
    """Render the six filled Unicode glyphs as CELL_SIZE_PX silhouettes.

    Keys are lowercase piece letters (k, q, r, b, n, p) indicating piece type
    only — color is resolved separately via luminance comparison.
    """
    font = _load_font(CELL_SIZE_PX)
    templates: dict[str, np.ndarray] = {}
    for piece in SHAPE_TEMPLATE_PIECES:
        glyph = GLYPHS[piece]
        img = Image.new("L", (CELL_SIZE_PX, CELL_SIZE_PX), 0)
        draw = ImageDraw.Draw(img)
        # Both white and black glyphs in Unicode have thick strokes; we fill
        # either way and threshold to a silhouette.
        try:
            bbox = draw.textbbox((0, 0), glyph, font=font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            x = (CELL_SIZE_PX - tw) // 2 - bbox[0]
            y = (CELL_SIZE_PX - th) // 2 - bbox[1]
        except AttributeError:
            tw, th = draw.textsize(glyph, font=font)
            x = (CELL_SIZE_PX - tw) // 2
            y = (CELL_SIZE_PX - th) // 2
        draw.text((x, y), glyph, fill=255, font=font)
        arr = np.array(img, dtype=np.uint8)
        _, arr = cv2.threshold(arr, 64, 255, cv2.THRESH_BINARY)
        templates[piece] = arr
    return templates


def _load_font(size: int) -> ImageFont.ImageFont:
    candidates: Iterable[str] = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSerif.ttf",
        "/Library/Fonts/Arial Unicode.ttf",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/System/Library/Fonts/Apple Symbols.ttf",
        "C:/Windows/Fonts/seguisym.ttf",
        "C:/Windows/Fonts/arial.ttf",
    )
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def encode_board_image(board_img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", board_img)
    if not ok:
        raise RuntimeError("Failed to encode warped board image.")
    return bytes(buf)


def board_image_to_pil(board_img: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(board_img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)
