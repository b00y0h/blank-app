# Chess Move Analyzer

Point your camera (or screenshot tool) at a chessboard, pick an engine, and
get the best move. Stockfish and Leela Chess Zero (Lc0) are both supported —
the user picks which one runs.

Three surfaces share one Python backend:

- **Web app** — `streamlit_app.py`, a Streamlit UI for upload → review FEN → analyze.
- **HTTP API** — `api.py`, a FastAPI endpoint used by the iOS app (and any other client).
- **iOS app** — SwiftUI sources under `ios/ChessMoveAnalyzer/`.

## How it works

1. **Board detection.** OpenCV finds the largest axis-aligned square in the
   image and warps it to a 512×512 canvas.
2. **Piece classification.** Each of the 64 cells is converted to a silhouette
   via Otsu thresholding. Fill-ratio decides occupancy; template matching
   against six filled Unicode chess glyphs picks the piece type; the piece's
   mean luminance vs. its square's background decides color.
3. **Review FEN.** Both UIs expose the detected FEN as an editable text field
   so the user can correct misclassifications before running the engine. You
   can also paste a FEN directly and skip vision entirely.
4. **Engine analysis.** The FEN is handed to Stockfish or Lc0 over UCI (via
   `python-chess`), with a configurable time or depth limit. The response
   includes the best move (SAN + UCI), evaluation, and principal variation.

The Unicode-glyph templates are a reasonable default for clean 2D boards but
are not tuned for any specific site's piece set. For better accuracy on e.g.
chess.com or Lichess, drop real piece PNGs into `_shape_templates()` in
`chess_analyzer/vision.py`.

## Setup

```bash
pip install -r requirements.txt
```

You also need the engine binaries on PATH (or pointed to by env vars):

- **Stockfish**
  - macOS: `brew install stockfish`
  - Debian/Ubuntu: `apt-get install stockfish`
  - Windows: download from <https://stockfishchess.org/download/>
  - Or set `STOCKFISH_PATH=/absolute/path/to/stockfish`.
- **Leela Chess Zero**
  - See <https://lczero.org/play/download/>. Lc0 also needs a network weights
    file (`.pb.gz`) — point to it with Lc0's `WeightsFile` option (you can
    configure this via the `LC0_PATH` binary wrapper, or run `lc0` directly
    to set its default weights).
  - Set `LC0_PATH=/absolute/path/to/lc0` if it's not on PATH.

The web UI and API both show which engines are detected at startup.

## Run the web app

```bash
streamlit run streamlit_app.py
```

Upload a screenshot, review the detected FEN, pick Stockfish or Lc0, adjust
the think-time / depth, and hit **Find best move**.

## Run the API

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

Endpoints:

- `GET /health` — liveness + which engines are available.
- `POST /analyze` — multipart form:
  - `image` *(file, optional)* — screenshot. Omit if sending `fen`.
  - `fen` *(string, optional)* — skip vision entirely.
  - `engine` — `stockfish` or `lc0` (default `stockfish`).
  - `orientation` — `white` or `black` (which color is at the image bottom).
  - `side_to_move` — `w` or `b`.
  - `time_limit_s` — float seconds (default `1.0`).
  - `depth` — optional int; overrides time limit.
  - `multipv` — lines of analysis (default 1).

  Returns JSON with `fen`, `best_move_san`, `best_move_uci`, `evaluation`,
  `principal_variation`, `depth`, `nodes`, `time_ms`, `engine`.

## Run the iOS app

See [`ios/README.md`](ios/README.md). Short version: create an Xcode SwiftUI
app project, drop in the three `.swift` files and the `Info.plist` entries,
point it at your backend URL, and build.

## Tests

```bash
pytest -q
```

The test suite exercises the vision pipeline end-to-end on synthetic boards;
engine tests auto-skip when the binaries aren't installed.

## Repo layout

```
chess_analyzer/      Python package: vision, engines, analyzer glue
streamlit_app.py     Web UI
api.py               FastAPI backend
ios/                 SwiftUI sources + setup guide
tests/               Vision and engine tests
requirements.txt     Python deps
```
