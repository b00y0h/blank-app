# Chess Move Analyzer

Upload or photograph a chessboard, pick an engine (Stockfish or Leela Chess
Zero), and get the best move. Then keep hitting **Play best move** — the
engine plays both sides and you can step through the game.

Three surfaces share one Python backend:

- **Web app** — `streamlit_app.py` (upload → review FEN → analyze → play).
- **HTTP API** — `api.py` (FastAPI, used by the iOS app and any client).
- **iOS app** — SwiftUI sources under `ios/ChessMoveAnalyzer/`.

## Vision: four classifier modes

Board recognition across every site/theme/camera is an open problem. This
project picks its battles:

| Mode          | How it works                                                   | When to use |
| ------------- | -------------------------------------------------------------- | ----------- |
| **`neural`**  | [`board_to_fen`](https://github.com/mcdominik/board_to_fen) — a small Keras CNN trained on Lichess-style 2D boards classifies each square. Bundled weights, runs **fully offline**. | Default whenever you want offline, multi-site detection. Requires the optional install (see below). |
| **`vlm`**     | Claude reads the board image and returns a FEN directly.        | Photos of physical boards, exotic 3D themes, anywhere the CNN struggles. Needs `ANTHROPIC_API_KEY` + network. |
| **`templates`** | PNG piece sets under `chess_analyzer/templates/<name>/` are matched per-cell. Falls back to Unicode-glyph shape templates if no PNGs are present. | Lightweight fallback when neither neural nor VLM is configured. |
| **`auto`**    | Try neural → VLM → templates, falling through on failure.       | Default. |

Whichever classifier wins, both UIs surface the detected FEN as an editable
field before the game starts.

### Install the offline neural classifier

The neural path is an optional extra because it pulls in TensorFlow
(~500 MB on disk):

```bash
pip install -r requirements-neural.txt
```

That installs `board_to_fen` plus the matching `tensorflow==2.15.*` and
`keras==2.15.*` versions (board_to_fen's bundled SavedModel needs Keras 2;
later versions can't load it). After install, the **neural** classifier
becomes available and is auto-selected first; nothing else changes.

For Streamlit Community Cloud the neural extras can exceed the free-tier
limits, so the cloud deploy stays on the smaller default dependency set
(`requirements.txt`). For a personal device or your own VM, the neural
extras give you offline, no-network, CNN-grade accuracy on chess.com,
Lichess, and most 2D boards.

## Setup

```bash
pip install -r requirements.txt
```

Engine binaries (install whichever you'll use):

- **Stockfish** — `brew install stockfish` (macOS) · `apt-get install stockfish` (Debian/Ubuntu) · or set `STOCKFISH_PATH=/abs/path/to/stockfish`.
- **Leela Chess Zero** — See <https://lczero.org/play/download/>. Lc0 also needs a weights file; point to it via Lc0's own config or `LC0_PATH`.

Optional add-ons:

- **Offline neural classifier** (recommended for accuracy on real screenshots):
  ```bash
  pip install -r requirements-neural.txt
  ```
  Adds `board_to_fen` + matched TensorFlow / Keras (~500 MB).
- **Claude VLM classifier** (handles photos / exotic boards over the network):
  ```bash
  export ANTHROPIC_API_KEY=sk-ant-...
  ```

## Run the web app

### Locally

```bash
streamlit run streamlit_app.py
```

### Streamlit Community Cloud (deploy from your phone)

The repo ships ready to deploy: `streamlit_app.py` at the root,
`requirements.txt` for Python deps, and `packages.txt` so Stockfish gets
`apt-get`-installed automatically.

1. Open <https://share.streamlit.io> in your phone browser and sign in with
   GitHub.
2. Tap **Create app** → **Deploy a public app from GitHub**.
3. Repository: `b00y0h/blank-app`. Branch:
   `claude/chess-move-analyzer-UgpGU` (or `main` once you merge). Main file
   path: `streamlit_app.py`.
4. (Optional, for the VLM classifier) Tap **Advanced settings** →
   **Secrets** and paste:
   ```toml
   ANTHROPIC_API_KEY = "sk-ant-..."
   ```
5. **Deploy**. First boot installs Stockfish + Python deps (~1–2 minutes).

When the app comes up:

1. Upload a screenshot (or paste a FEN).
2. Confirm the detected FEN and hit **Start game from this FEN**.
3. **Find best move** → **Play best move** alternates sides turn by turn.
4. **Undo** / **Reset** to step back or start a new game.

Notes for the cloud deploy:

- Stockfish runs out of the box. Lc0 isn't in `apt`, so the Lc0 option will
  show **(not installed)**; pick Stockfish.
- Without `ANTHROPIC_API_KEY`, the **Auto** classifier silently falls back
  to the bundled-template / Unicode-glyph path. The FEN field is editable
  either way, so you can correct any misreads before starting the game.

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
