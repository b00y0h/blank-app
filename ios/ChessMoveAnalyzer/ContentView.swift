import PhotosUI
import SwiftUI

struct ContentView: View {
    @AppStorage("backendURL") private var backendURL: String = "http://localhost:8000"

    @State private var photoItem: PhotosPickerItem?
    @State private var boardImage: UIImage?

    @State private var engine: EngineChoice = .stockfish
    @State private var orientation: Orientation = .white
    @State private var sideToMove: SideToMove = .white
    @State private var thinkSeconds: Double = 1.0

    @State private var isAnalyzing = false
    @State private var result: AnalyzeResponse?
    @State private var errorMessage: String?

    var body: some View {
        NavigationStack {
            Form {
                Section("Backend") {
                    TextField("URL (e.g. http://192.168.1.5:8000)", text: $backendURL)
                        .textInputAutocapitalization(.never)
                        .keyboardType(.URL)
                        .autocorrectionDisabled()
                }

                Section("Screenshot") {
                    PhotosPicker(selection: $photoItem, matching: .images) {
                        Label(boardImage == nil ? "Pick a board screenshot" : "Change screenshot",
                              systemImage: "photo.on.rectangle")
                    }
                    if let img = boardImage {
                        Image(uiImage: img)
                            .resizable()
                            .scaledToFit()
                            .frame(maxHeight: 260)
                            .cornerRadius(8)
                    }
                }

                Section("Engine") {
                    Picker("Engine", selection: $engine) {
                        ForEach(EngineChoice.allCases) { e in
                            Text(e.displayName).tag(e)
                        }
                    }
                    .pickerStyle(.segmented)

                    Picker("Orientation", selection: $orientation) {
                        ForEach(Orientation.allCases) { o in
                            Text(o.displayName).tag(o)
                        }
                    }

                    Picker("Side to move", selection: $sideToMove) {
                        ForEach(SideToMove.allCases) { s in
                            Text(s.displayName).tag(s)
                        }
                    }

                    VStack(alignment: .leading) {
                        Text("Think time: \(thinkSeconds, specifier: "%.1f")s")
                        Slider(value: $thinkSeconds, in: 0.1...10.0, step: 0.1)
                    }
                }

                Section {
                    Button {
                        Task { await analyze() }
                    } label: {
                        if isAnalyzing {
                            ProgressView().frame(maxWidth: .infinity)
                        } else {
                            Text("Find best move")
                                .bold()
                                .frame(maxWidth: .infinity)
                        }
                    }
                    .disabled(boardImage == nil || isAnalyzing)
                }

                if let r = result {
                    Section("Result") {
                        LabeledContent("Best move", value: r.bestMoveSan)
                        LabeledContent("UCI", value: r.bestMoveUci)
                        LabeledContent("Evaluation", value: r.evaluation)
                        if let d = r.depth { LabeledContent("Depth", value: String(d)) }
                        LabeledContent("Engine", value: r.engine)
                        if !r.principalVariation.isEmpty {
                            VStack(alignment: .leading, spacing: 4) {
                                Text("Principal variation").font(.subheadline).foregroundStyle(.secondary)
                                Text(r.principalVariation.joined(separator: " "))
                                    .font(.system(.body, design: .monospaced))
                            }
                        }
                        VStack(alignment: .leading, spacing: 4) {
                            Text("Detected FEN").font(.subheadline).foregroundStyle(.secondary)
                            Text(r.fen).font(.system(.caption, design: .monospaced))
                        }
                    }
                }

                if let msg = errorMessage {
                    Section {
                        Text(msg).foregroundStyle(.red)
                    }
                }
            }
            .navigationTitle("Chess Analyzer")
            .onChange(of: photoItem) { _, new in
                Task { await loadImage(from: new) }
            }
        }
    }

    private func loadImage(from item: PhotosPickerItem?) async {
        guard let item else { return }
        if let data = try? await item.loadTransferable(type: Data.self),
           let img = UIImage(data: data) {
            boardImage = img
            result = nil
            errorMessage = nil
        }
    }

    private func analyze() async {
        guard let img = boardImage else { return }
        guard let url = URL(string: backendURL.trimmingCharacters(in: .whitespaces)) else {
            errorMessage = "Backend URL is invalid."
            return
        }
        isAnalyzing = true
        defer { isAnalyzing = false }
        errorMessage = nil
        result = nil
        do {
            let service = AnalyzerService(baseURL: url)
            result = try await service.analyze(
                image: img,
                engine: engine,
                orientation: orientation,
                sideToMove: sideToMove,
                timeLimitSeconds: thinkSeconds
            )
        } catch {
            errorMessage = error.localizedDescription
        }
    }
}

#Preview {
    ContentView()
}
