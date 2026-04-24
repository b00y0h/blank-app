# Chess Move Analyzer

Upload or photograph a chessboard, pick an engine (Stockfish or Leela Chess
Zero), and get the best move. Then keep hitting **Play best move** — the
engine plays both sides and you can step through the game.

Three surfaces share one Python backend:

- **Web app** — `streamlit_app.py` (upload → review FEN → analyze → play).
- **HTTP API** — `api.py` (FastAPI, used by the iOS app and any client).
- **iOS app** — SwiftUI sources under `ios/ChessMoveAnalyzer/`.

## Vision: three classifier modes

Board recognition across every site/theme/camera is an open problem. This
project picks its battles:

| Mode          | How it works                                                   | When to use |
| ------------- | -------------------------------------------------------------- | ----------- |
| **`vlm`**     | Claude reads the board image and returns a FEN directly.        | Maximum robustness; any site, any theme, phone photos. Needs `ANTHROPIC_API_KEY` + network. |
| **`templates`** | PNG piece sets under `chess_analyzer/templates/<name>/` are matched per-cell. Falls back to Unicode-glyph shape templates if no PNGs are present. | Offline, fast, cheap. Most reliable when you add a template set per site you use. |
| **`auto`**    | Try VLM first, fall back to templates on failure.               | Default. |

Bundling a site's piece set (e.g. chess.com) is 12 PNGs — see
[`chess_analyzer/templates/README.md`](chess_analyzer/templates/README.md).
The loader auto-picks whichever set best matches the screenshot, so you can
keep multiple sets around (chess.com, lichess, your own 3D set) without
configuration.

Regardless of mode, both UIs surface the detected FEN as an editable field
before the game starts.

## Setup

```bash
pip install -r requirements.txt
```

Engine binaries (install whichever you'll use):

- **Stockfish** — `brew install stockfish` (macOS) · `apt-get install stockfish` (Debian/Ubuntu) · or set `STOCKFISH_PATH=/abs/path/to/stockfish`.
- **Leela Chess Zero** — See <https://lczero.org/play/download/>. Lc0 also needs a weights file; point to it via Lc0's own config or `LC0_PATH`.

Optional for VLM classifier:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

## Run the web app

```bash
streamlit run streamlit_app.py
```

1. Upload a screenshot (or paste a FEN).
2. Confirm the detected FEN and hit **Start game from this FEN**.
3. **Find best move** → **Play best move** alternates sides turn by turn.
4. **Undo** / **Reset** to step back or start a new game.

## Run the API

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

Endpoints:

- `GET /health` — liveness, installed engines, available template sets, VLM-configured flag.
- `POST /analyze` — multipart form:
  - `image` *(file, optional)* or `fen` *(string, optional)* — one is required.
  - `engine` — `stockfish` | `lc0`.
  - `classifier` — `auto` | `vlm` | `templates`.
  - `template_set` — bundled set name (optional; auto-selected if omitted).
  - `orientation` — `white` | `black`.
  - `side_to_move` — `w` | `b`.
  - `time_limit_s` *(float)* — or `depth` *(int)* to override.
  - `multipv` *(int)*.

  Returns `fen`, `next_fen` (position after best move), `best_move_san`,
  `best_move_uci`, `evaluation`, `principal_variation`, `classifier_used`,
  plus engine metadata.

- `POST /play_move` — JSON `{ "fen": "...", "move_uci": "e2e4" }` →
  `{ fen, last_move_san, last_move_uci, is_game_over, outcome }`. The server
  is stateless; clients maintain the board.

## Run the iOS app

See [`ios/README.md`](ios/README.md). Short version: create an Xcode SwiftUI
project, drop in the three `.swift` files and the `Info.plist` entries, point
it at your backend URL, and build.

Flow in the app:

1. Pick a screenshot (or paste FEN) → **Start game**.
2. **Find best move** → **Play \<move\>** to push it.
3. The FEN advances; repeat. **Undo** / **Reset** are always available.

## Tests

```bash
pytest -q
```

The vision pipeline is exercised on synthetic boards; play-move tests cover
chaining, legality, and game-over detection; the VLM module's helpers test
without hitting the API. Engine tests skip cleanly when binaries aren't
installed.

## Repo layout

```
chess_analyzer/        Python package
├── analyzer.py        Glue + play-move helper
├── engines.py         Stockfish + Lc0 UCI wrappers
├── vision.py          OpenCV board detection + template classifier
├── vision_vlm.py      Claude VLM classifier
└── templates/         Drop bundled PNG piece sets here
streamlit_app.py       Web UI
api.py                 FastAPI backend
ios/                   SwiftUI sources + setup guide
tests/                 Vision, engines, play-move, VLM tests
```
