import PhotosUI
import SwiftUI

struct PlayedMove: Identifiable, Hashable {
    let id = UUID()
    let san: String
    let uci: String
    let fenBefore: String
    let fenAfter: String
}

struct ContentView: View {
    @AppStorage("backendURL") private var backendURL: String = "http://localhost:8000"

    // Setup state
    @State private var photoItem: PhotosPickerItem?
    @State private var boardImage: UIImage?
    @State private var fenDraft: String = ""

    // Settings
    @State private var engine: EngineChoice = .stockfish
    @State private var classifier: ClassifierChoice = .auto
    @State private var orientation: Orientation = .white
    @State private var sideToMove: SideToMove = .white
    @State private var thinkSeconds: Double = 1.0

    // Game state
    @State private var currentFen: String?
    @State private var history: [PlayedMove] = []
    @State private var lastResult: AnalyzeResponse?

    // UX state
    @State private var isWorking = false
    @State private var errorMessage: String?

    var body: some View {
        NavigationStack {
            Form {
                backendSection
                settingsSection
                if currentFen == nil {
                    setupSection
                } else {
                    gameSection
                }
                if let msg = errorMessage {
                    Section { Text(msg).foregroundStyle(.red) }
                }
            }
            .navigationTitle("Chess Analyzer")
            .onChange(of: photoItem) { _, new in
                Task { await loadImage(from: new) }
            }
        }
    }

    // MARK: - Sections

    private var backendSection: some View {
        Section("Backend") {
            TextField("URL (e.g. http://192.168.1.5:8000)", text: $backendURL)
                .textInputAutocapitalization(.never)
                .keyboardType(.URL)
                .autocorrectionDisabled()
        }
    }

    private var settingsSection: some View {
        Section("Engine & vision") {
            Picker("Engine", selection: $engine) {
                ForEach(EngineChoice.allCases) { Text($0.displayName).tag($0) }
            }
            .pickerStyle(.segmented)

            Picker("Classifier", selection: $classifier) {
                ForEach(ClassifierChoice.allCases) { Text($0.displayName).tag($0) }
            }
            .pickerStyle(.segmented)

            Picker("Orientation", selection: $orientation) {
                ForEach(Orientation.allCases) { Text($0.displayName).tag($0) }
            }

            Picker("Side to move", selection: $sideToMove) {
                ForEach(SideToMove.allCases) { Text($0.displayName).tag($0) }
            }

            VStack(alignment: .leading) {
                Text("Think time: \(thinkSeconds, specifier: "%.1f")s")
                Slider(value: $thinkSeconds, in: 0.1...10.0, step: 0.1)
            }
        }
    }

    @ViewBuilder
    private var setupSection: some View {
        Section("Screenshot") {
            PhotosPicker(selection: $photoItem, matching: .images) {
                Label(boardImage == nil ? "Pick a board screenshot" : "Change screenshot",
                      systemImage: "photo.on.rectangle")
            }
            if let img = boardImage {
                Image(uiImage: img)
                    .resizable()
                    .scaledToFit()
                    .frame(maxHeight: 240)
                    .cornerRadius(8)
            }
            TextField("…or paste a FEN", text: $fenDraft, axis: .vertical)
                .font(.system(.caption, design: .monospaced))
                .textInputAutocapitalization(.never)
                .autocorrectionDisabled()
        }
        Section {
            Button {
                Task { await startGame() }
            } label: {
                if isWorking {
                    ProgressView().frame(maxWidth: .infinity)
                } else {
                    Text("Start game").bold().frame(maxWidth: .infinity)
                }
            }
            .disabled(isWorking || (boardImage == nil && fenDraft.trimmingCharacters(in: .whitespaces).isEmpty))
        }
    }

    @ViewBuilder
    private var gameSection: some View {
        Section("Current position") {
            if let fen = currentFen {
                Text(fen).font(.system(.caption, design: .monospaced))
            }
            HStack {
                Button {
                    Task { await findBestMove() }
                } label: {
                    if isWorking {
                        ProgressView().frame(maxWidth: .infinity)
                    } else {
                        Text("Find best move").bold().frame(maxWidth: .infinity)
                    }
                }
                .buttonStyle(.borderedProminent)
                .disabled(isWorking)
            }
            if let r = lastResult, r.fen == currentFen {
                LabeledContent("Best move", value: r.bestMoveSan)
                LabeledContent("UCI", value: r.bestMoveUci)
                LabeledContent("Evaluation", value: r.evaluation)
                if let d = r.depth { LabeledContent("Depth", value: String(d)) }
                LabeledContent("Engine", value: r.engine)
                if let cls = r.classifierUsed { LabeledContent("Classifier", value: cls) }
                if !r.principalVariation.isEmpty {
                    VStack(alignment: .leading, spacing: 4) {
                        Text("PV").font(.subheadline).foregroundStyle(.secondary)
                        Text(r.principalVariation.joined(separator: " "))
                            .font(.system(.body, design: .monospaced))
                    }
                }
                Button {
                    Task { await playBestMove() }
                } label: {
                    Text("Play \(r.bestMoveSan)").bold().frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .disabled(isWorking)
            }
        }
        Section("Move history") {
            if history.isEmpty {
                Text("_No moves yet._").italic().foregroundStyle(.secondary)
            } else {
                Text(pgnString).font(.system(.body, design: .monospaced))
            }
            HStack {
                Button(role: .destructive) {
                    undoLast()
                } label: {
                    Text("Undo").frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered)
                .disabled(history.isEmpty || isWorking)

                Button(role: .destructive) {
                    resetGame()
                } label: {
                    Text("Reset").frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered)
                .disabled(isWorking)
            }
        }
    }

    private var pgnString: String {
        history.enumerated().map { i, mv in
            i % 2 == 0 ? "\(i / 2 + 1). \(mv.san)" : mv.san
        }.joined(separator: " ")
    }

    // MARK: - Actions

    private func loadImage(from item: PhotosPickerItem?) async {
        guard let item else { return }
        if let data = try? await item.loadTransferable(type: Data.self),
           let img = UIImage(data: data) {
            boardImage = img
            errorMessage = nil
        }
    }

    private func service() -> AnalyzerService? {
        guard let url = URL(string: backendURL.trimmingCharacters(in: .whitespaces)) else {
            errorMessage = "Backend URL is invalid."
            return nil
        }
        return AnalyzerService(baseURL: url)
    }

    private func startGame() async {
        guard let svc = service() else { return }
        errorMessage = nil
        isWorking = true
        defer { isWorking = false }
        do {
            let response = try await svc.analyze(
                image: fenDraft.trimmingCharacters(in: .whitespaces).isEmpty ? boardImage : nil,
                fen: fenDraft.trimmingCharacters(in: .whitespaces).isEmpty ? nil : fenDraft,
                engine: engine,
                classifier: classifier,
                orientation: orientation,
                sideToMove: sideToMove,
                timeLimitSeconds: thinkSeconds,
                templateSet: nil
            )
            currentFen = response.fen
            lastResult = response
            history = []
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    private func findBestMove() async {
        guard let svc = service(), let fen = currentFen else { return }
        errorMessage = nil
        isWorking = true
        defer { isWorking = false }
        do {
            let response = try await svc.analyze(
                image: nil,
                fen: fen,
                engine: engine,
                classifier: classifier,
                orientation: orientation,
                sideToMove: currentSide(from: fen),
                timeLimitSeconds: thinkSeconds,
                templateSet: nil
            )
            lastResult = response
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    private func playBestMove() async {
        guard let svc = service(), let fen = currentFen, let r = lastResult else { return }
        errorMessage = nil
        isWorking = true
        defer { isWorking = false }
        do {
            let response = try await svc.playMove(fen: fen, moveUci: r.bestMoveUci)
            history.append(PlayedMove(
                san: response.lastMoveSan,
                uci: response.lastMoveUci,
                fenBefore: fen,
                fenAfter: response.fen
            ))
            currentFen = response.fen
            lastResult = nil
            if response.isGameOver, let outcome = response.outcome {
                errorMessage = "Game over — \(outcome)"
            }
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    private func undoLast() {
        guard let last = history.popLast() else { return }
        currentFen = last.fenBefore
        lastResult = nil
        errorMessage = nil
    }

    private func resetGame() {
        currentFen = nil
        history = []
        lastResult = nil
        errorMessage = nil
    }

    private func currentSide(from fen: String) -> SideToMove {
        // FEN field 2 is side-to-move.
        let parts = fen.split(separator: " ")
        if parts.count >= 2, parts[1] == "b" { return .black }
        return .white
    }
}

#Preview {
    ContentView()
}
