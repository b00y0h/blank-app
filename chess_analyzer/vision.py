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
    template_set_used: str | None = None


def detect_board_fen(
    image_bytes: bytes,
    *,
    orientation: str = "white",  # "white" = white pieces at bottom of image
    side_to_move: str = "w",
    template_set: str | None = None,
) -> BoardDetectionResult:
    """Detect chessboard in image_bytes and return an inferred FEN.

    `template_set` selects a bundled PNG template set under
    `chess_analyzer/templates/<name>/`. When None, every available set is
    tried and the one with the highest mean cell confidence wins. When no
    PNG sets are present we fall back to Unicode-glyph shape templates.
    """
    if orientation not in {"white", "black"}:
        raise ValueError("orientation must be 'white' or 'black'")
    if side_to_move not in {"w", "b"}:
        raise ValueError("side_to_move must be 'w' or 'b'")

    img = _decode_image(image_bytes)
    warped, found = _find_and_warp_board(img)

    grid, confidences, set_used = _classify_cells(warped, template_set=template_set)
    if orientation == "black":
        grid = [row[::-1] for row in grid[::-1]]

    fen = _grid_to_fen(grid, side_to_move=side_to_move)
    return BoardDetectionResult(
        fen=fen,
        board_image=warped,
        cell_confidences=confidences,
        found_board=found,
        template_set_used=set_used,
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


def _classify_cells(
    board_img: np.ndarray, *, template_set: str | None = None
) -> tuple[list[list[str]], list[float], str | None]:
    """Return an 8x8 grid of piece codes ('.' for empty) + per-cell confidence
    + the name of the template set used.

    Strategy:
      1. If PNG template sets are available under `chess_analyzer/templates/`,
         try the named set (or every set when `template_set` is None) and pick
         the one with the highest mean cell confidence. PNG sets resolve both
         piece TYPE and COLOR directly.
      2. Otherwise fall back to Unicode-glyph shape templates (type only),
         with color decided by comparing piece luminance to square background.
    """
    gray = cv2.cvtColor(board_img, cv2.COLOR_BGR2GRAY)

    png_sets = _png_template_sets()
    if template_set is not None:
        png_sets = {template_set: png_sets[template_set]} if template_set in png_sets else {}

    if png_sets:
        best: tuple[list[list[str]], list[float], str] | None = None
        for name, templates in png_sets.items():
            grid, confs = _classify_with_color_templates(gray, templates)
            mean_conf = sum(confs) / max(len(confs), 1)
            if best is None or mean_conf > sum(best[1]) / max(len(best[1]), 1):
                best = (grid, confs, name)
        assert best is not None
        return best

    shape_templates = _shape_templates()
    grid, confs = _classify_with_shape_templates(gray, shape_templates)
    return grid, confs, None


def _classify_with_shape_templates(
    gray: np.ndarray, templates: dict[str, np.ndarray]
) -> tuple[list[list[str]], list[float]]:
    grid: list[list[str]] = []
    confs: list[float] = []
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

            piece_type, score = _best_shape_match(silhouette, templates)
            if piece_type == ".":
                row.append(".")
                confs.append(score)
                continue

            piece_mean = float(cell[silhouette > 0].mean())
            bg_mean = (
                float(cell[silhouette == 0].mean())
                if (silhouette == 0).any()
                else 128.0
            )
            is_white_piece = piece_mean > bg_mean
            row.append(piece_type.upper() if is_white_piece else piece_type.lower())
            confs.append(score)
        grid.append(row)
    return grid, confs


def _classify_with_color_templates(
    gray: np.ndarray, templates: dict[str, np.ndarray]
) -> tuple[list[list[str]], list[float]]:
    """Templates are keyed by full piece code ('K','k',...) — one per color.

    We correlate every cell against every template and use the normalized
    cross-correlation peak to decide piece type and color together. An
    additional "empty" threshold prunes clearly-empty squares first.
    """
    grid: list[list[str]] = []
    confs: list[float] = []
    min_fill = 0.03

    for r in range(8):
        row: list[str] = []
        for f in range(8):
            cell = _extract_cell(gray, r, f)
            silhouette = _cell_to_silhouette(cell)
            fill_ratio = float(cv2.countNonZero(silhouette)) / float(silhouette.size)
            if fill_ratio < min_fill:
                row.append(".")
                confs.append(0.0)
                continue

            best_piece = "."
            best_score = -1.0
            for piece, tmpl in templates.items():
                res = cv2.matchTemplate(cell, tmpl, cv2.TM_CCOEFF_NORMED)
                score = float(res.max())
                if score > best_score:
                    best_score = score
                    best_piece = piece
            if best_score < 0.30:
                row.append(".")
                confs.append(best_score)
            else:
                row.append(best_piece)
                confs.append(best_score)
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


_TEMPLATE_DIR_NAME = "templates"
_PIECE_FILENAMES: dict[str, tuple[str, ...]] = {
    # FEN code -> filenames we'll accept (case-insensitive, without .png).
    "K": ("wk", "whiteking", "king_w"),
    "Q": ("wq", "whitequeen", "queen_w"),
    "R": ("wr", "whiterook", "rook_w"),
    "B": ("wb", "whitebishop", "bishop_w"),
    "N": ("wn", "whiteknight", "knight_w"),
    "P": ("wp", "whitepawn", "pawn_w"),
    "k": ("bk", "blackking", "king_b"),
    "q": ("bq", "blackqueen", "queen_b"),
    "r": ("br", "blackrook", "rook_b"),
    "b": ("bb", "blackbishop", "bishop_b"),
    "n": ("bn", "blackknight", "knight_b"),
    "p": ("bp", "blackpawn", "pawn_b"),
}


@lru_cache(maxsize=1)
def _png_template_sets() -> dict[str, dict[str, np.ndarray]]:
    """Scan templates/<set>/*.png and return loaded, resized, grayscale sets.

    Each set must contain all 12 piece files (any accepted filename per piece,
    any reasonable image size — we resize to CELL_SIZE_PX). Sets missing pieces
    are skipped with a note; we never partially load a set.
    """
    import os

    base = os.path.join(os.path.dirname(__file__), _TEMPLATE_DIR_NAME)
    if not os.path.isdir(base):
        return {}

    sets: dict[str, dict[str, np.ndarray]] = {}
    for set_name in sorted(os.listdir(base)):
        set_dir = os.path.join(base, set_name)
        if not os.path.isdir(set_dir):
            continue
        entries = {f.lower(): f for f in os.listdir(set_dir) if f.lower().endswith(".png")}
        loaded: dict[str, np.ndarray] = {}
        ok = True
        for piece, candidates in _PIECE_FILENAMES.items():
            match = None
            for cand in candidates:
                key = f"{cand}.png"
                if key in entries:
                    match = entries[key]
                    break
            if match is None:
                ok = False
                break
            img = cv2.imread(os.path.join(set_dir, match), cv2.IMREAD_UNCHANGED)
            if img is None:
                ok = False
                break
            # Composite any alpha channel onto a mid-gray backdrop so matching
            # doesn't lock onto transparent borders.
            if img.ndim == 3 and img.shape[2] == 4:
                alpha = img[:, :, 3:4].astype(np.float32) / 255.0
                bgr = img[:, :, :3].astype(np.float32)
                backdrop = np.full_like(bgr, 128.0)
                img = (alpha * bgr + (1 - alpha) * backdrop).astype(np.uint8)
            if img.ndim == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (CELL_SIZE_PX, CELL_SIZE_PX), interpolation=cv2.INTER_AREA)
            loaded[piece] = img
        if ok:
            sets[set_name] = loaded
    return sets


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
