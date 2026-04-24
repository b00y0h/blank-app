# Chess Move Analyzer — iOS app

SwiftUI client that sends a chessboard screenshot to the FastAPI backend and
displays the engine's best move.

## Create the Xcode project

Swift source files are committed here, but the Xcode project file isn't
(Xcode-generated `.pbxproj` files are noisy and diverge per machine). Create
one in ~1 minute:

1. Open Xcode → **File → New → Project…**
2. iOS → **App** → Next.
3. Product Name: `ChessMoveAnalyzer` · Interface: **SwiftUI** · Language: **Swift**.
4. Save it somewhere convenient (not inside this repo).
5. In Finder, delete the `ContentView.swift` / `ChessMoveAnalyzerApp.swift`
   that Xcode generated.
6. Drag the three `.swift` files from `ChessMoveAnalyzer/` in this repo into
   the new Xcode project (check **Copy items if needed**).
7. Open the generated `Info.plist` and add, or replace with the entries in
   `ChessMoveAnalyzer/Info.plist` here. The important keys are:
   - `NSPhotoLibraryUsageDescription` — required for `PhotosPicker`.
   - `NSAppTransportSecurity → NSAllowsLocalNetworking = YES` — lets the
     simulator/device hit your laptop's `http://` dev backend.
8. Build & run on the iOS 17+ simulator or a device.

## Point the app at your backend

Start the backend from the repo root:

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

In the app, paste the backend URL in the **Backend** field:

- Simulator: `http://localhost:8000`
- Physical device on the same Wi-Fi: `http://<your-laptop-lan-ip>:8000`

Pick a screenshot, choose Stockfish or Leela Chess Zero, tap **Find best move**.

## Deploying publicly

`NSAllowsLocalNetworking` is only for LAN dev. For a production deployment,
host the backend behind HTTPS (e.g. Fly.io, Render, Cloud Run) and swap the
URL — iOS will accept HTTPS out of the box without any ATS exceptions.
