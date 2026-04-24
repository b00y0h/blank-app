import Foundation
import UIKit

struct AnalyzeResponse: Decodable {
    let fen: String
    let nextFen: String
    let detectedFromImage: Bool
    let boardFound: Bool
    let classifierUsed: String?
    let engine: String
    let bestMoveUci: String
    let bestMoveSan: String
    let evaluation: String
    let principalVariation: [String]
    let depth: Int?
    let nodes: Int?
    let timeMs: Int?

    enum CodingKeys: String, CodingKey {
        case fen
        case nextFen = "next_fen"
        case detectedFromImage = "detected_from_image"
        case boardFound = "board_found"
        case classifierUsed = "classifier_used"
        case engine
        case bestMoveUci = "best_move_uci"
        case bestMoveSan = "best_move_san"
        case evaluation
        case principalVariation = "principal_variation"
        case depth
        case nodes
        case timeMs = "time_ms"
    }
}

struct PlayMoveResponse: Decodable {
    let fen: String
    let lastMoveSan: String
    let lastMoveUci: String
    let isGameOver: Bool
    let outcome: String?

    enum CodingKeys: String, CodingKey {
        case fen
        case lastMoveSan = "last_move_san"
        case lastMoveUci = "last_move_uci"
        case isGameOver = "is_game_over"
        case outcome
    }
}

enum EngineChoice: String, CaseIterable, Identifiable {
    case stockfish
    case lc0

    var id: String { rawValue }
    var displayName: String {
        switch self {
        case .stockfish: return "Stockfish"
        case .lc0:       return "Leela Chess Zero"
        }
    }
}

enum ClassifierChoice: String, CaseIterable, Identifiable {
    case auto
    case vlm
    case templates

    var id: String { rawValue }
    var displayName: String {
        switch self {
        case .auto:      return "Auto"
        case .vlm:       return "Claude VLM"
        case .templates: return "Templates"
        }
    }
}

enum Orientation: String, CaseIterable, Identifiable {
    case white
    case black
    var id: String { rawValue }
    var displayName: String { rawValue.capitalized + " at bottom" }
}

enum SideToMove: String, CaseIterable, Identifiable {
    case white = "w"
    case black = "b"
    var id: String { rawValue }
    var displayName: String { self == .white ? "White to move" : "Black to move" }
}

enum AnalyzerError: LocalizedError {
    case badImage
    case server(String)
    case transport(String)

    var errorDescription: String? {
        switch self {
        case .badImage:             return "Could not encode the selected image."
        case .server(let msg):      return msg
        case .transport(let msg):   return "Network error: \(msg)"
        }
    }
}

struct AnalyzerService {
    var baseURL: URL

    func analyze(
        image: UIImage?,
        fen: String?,
        engine: EngineChoice,
        classifier: ClassifierChoice,
        orientation: Orientation,
        sideToMove: SideToMove,
        timeLimitSeconds: Double,
        templateSet: String?
    ) async throws -> AnalyzeResponse {
        var request = URLRequest(url: baseURL.appendingPathComponent("analyze"))
        request.httpMethod = "POST"

        let boundary = "Boundary-\(UUID().uuidString)"
        request.setValue("multipart/form-data; boundary=\(boundary)",
                         forHTTPHeaderField: "Content-Type")

        var body = Data()
        func appendField(_ name: String, _ value: String) {
            body.append("--\(boundary)\r\n")
            body.append("Content-Disposition: form-data; name=\"\(name)\"\r\n\r\n")
            body.append("\(value)\r\n")
        }

        appendField("engine", engine.rawValue)
        appendField("classifier", classifier.rawValue)
        appendField("orientation", orientation.rawValue)
        appendField("side_to_move", sideToMove.rawValue)
        appendField("time_limit_s", String(timeLimitSeconds))
        if let fen, !fen.isEmpty { appendField("fen", fen) }
        if let templateSet, !templateSet.isEmpty { appendField("template_set", templateSet) }

        if let image {
            guard let jpeg = image.jpegData(compressionQuality: 0.9) else {
                throw AnalyzerError.badImage
            }
            body.append("--\(boundary)\r\n")
            body.append("Content-Disposition: form-data; name=\"image\"; filename=\"board.jpg\"\r\n")
            body.append("Content-Type: image/jpeg\r\n\r\n")
            body.append(jpeg)
            body.append("\r\n")
        }
        body.append("--\(boundary)--\r\n")
        request.httpBody = body

        return try await send(request)
    }

    func playMove(fen: String, moveUci: String) async throws -> PlayMoveResponse {
        var request = URLRequest(url: baseURL.appendingPathComponent("play_move"))
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONEncoder().encode([
            "fen": fen,
            "move_uci": moveUci,
        ])
        return try await send(request)
    }

    private func send<T: Decodable>(_ request: URLRequest) async throws -> T {
        do {
            let (data, response) = try await URLSession.shared.data(for: request)
            guard let http = response as? HTTPURLResponse else {
                throw AnalyzerError.transport("No HTTP response.")
            }
            if !(200..<300).contains(http.statusCode) {
                let detail = (try? JSONDecoder().decode([String: String].self, from: data))?["detail"]
                    ?? String(data: data, encoding: .utf8)
                    ?? "HTTP \(http.statusCode)"
                throw AnalyzerError.server(detail)
            }
            return try JSONDecoder().decode(T.self, from: data)
        } catch let err as AnalyzerError {
            throw err
        } catch {
            throw AnalyzerError.transport(error.localizedDescription)
        }
    }
}

private extension Data {
    mutating func append(_ string: String) {
        if let d = string.data(using: .utf8) { append(d) }
    }
}
