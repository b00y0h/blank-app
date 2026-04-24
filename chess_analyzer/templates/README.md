# Piece template sets

Drop each piece set in its own subdirectory. The loader scans every
subdirectory at import time and adds it as a named set.

## Directory layout

```
chess_analyzer/templates/
├── chesscom/
│   ├── wK.png   wQ.png   wR.png   wB.png   wN.png   wP.png
│   └── bK.png   bQ.png   bR.png   bB.png   bN.png   bP.png
├── lichess/
│   └── ...same 12 files...
└── my-custom-set/
    └── ...
```

The set name is the folder name. The detector will auto-pick whichever set
best matches the screenshot (highest mean cell confidence); you can also
force a specific set with `template_set="chesscom"` in code or via the API.

## File naming

Each piece accepts a short or long filename (case-insensitive):

| FEN | Short | Long          |
| --- | ----- | ------------- |
| K   | wK    | whiteKing     |
| Q   | wQ    | whiteQueen    |
| R   | wR    | whiteRook     |
| B   | wB    | whiteBishop   |
| N   | wN    | whiteKnight   |
| P   | wP    | whitePawn     |
| k   | bK    | blackKing     |
| q   | bQ    | blackQueen    |
| r   | bR    | blackRook     |
| b   | bB    | blackBishop   |
| n   | bN    | blackKnight   |
| p   | bP    | blackPawn     |

A set is loaded only if all 12 files are present (case-insensitive). Sets
missing any piece are skipped.

## Image format

- PNG, any size (auto-resized to the 64×64 cell size used by the detector).
- RGBA is supported — transparency is composited onto a neutral gray so the
  matcher doesn't lock onto the transparent border.
- Use a clean piece rendering on a neutral/no background. Including the
  board square colour in your templates will bias matching against that
  specific theme.

## Getting sets

- **Chess.com pieces** are not openly redistributable. The intended workflow
  is to screenshot one of each piece from your own chess.com board (zoom
  in, crop a single square per piece, remove the square background in any
  image editor), save as `wK.png`, `wQ.png`, … `bP.png`.
- **Lichess pieces** are available under a free licence from the
  [lila repository](https://github.com/lichess-org/lila/tree/master/public/piece)
  — pick any set (cburnett, merida, alpha, etc.). Convert the SVGs to PNG
  (e.g. `rsvg-convert -h 128 wK.svg > wK.png`).

Once added, restart the app; the loader runs at import time and caches
results.
