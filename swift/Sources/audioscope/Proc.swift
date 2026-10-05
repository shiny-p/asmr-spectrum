import Foundation

/// Errors that map to a non-zero exit with a readable message.
struct ToolError: Error, CustomStringConvertible {
    let message: String
    var description: String { message }
}

enum Proc {
    /// Run a process, returning captured stdout. stderr is drained concurrently so a
    /// chatty child cannot deadlock on a full pipe.
    static func run(_ launchPath: String, _ args: [String],
                    stdin: Data? = nil) throws -> Data {
        let p = Process()
        p.executableURL = URL(fileURLWithPath: launchPath)
        p.arguments = args
        let out = Pipe()
        let err = Pipe()
        p.standardOutput = out
        p.standardError = err

        var errData = Data()
        let errLock = NSLock()
        err.fileHandleForReading.readabilityHandler = { h in
            let d = h.availableData
            if !d.isEmpty { errLock.lock(); errData.append(d); errLock.unlock() }
        }

        var inPipe: Pipe?
        if stdin != nil {
            let ip = Pipe()
            p.standardInput = ip
            inPipe = ip
        }

        do {
            try p.run()
        } catch {
            throw ToolError(message: "cannot run \(launchPath): \(error.localizedDescription)")
        }

        if let data = stdin, let ip = inPipe {
            ip.fileHandleForWriting.write(data)
            try? ip.fileHandleForWriting.close()
        }

        let data = out.fileHandleForReading.readDataToEndOfFile()
        p.waitUntilExit()
        err.fileHandleForReading.readabilityHandler = nil
        errLock.lock(); let e = errData; errLock.unlock()

        guard p.terminationStatus == 0 else {
            let msg = String(data: e, encoding: .utf8) ?? ""
            throw ToolError(message: "\(launchPath) exited \(p.terminationStatus): "
                            + msg.split(separator: "\n").prefix(3).joined(separator: " | "))
        }
        return data
    }

    static func which(_ name: String) -> String? {
        for dir in ["/opt/homebrew/bin", "/usr/local/bin", "/usr/bin", "/bin"] {
            let p = "\(dir)/\(name)"
            if FileManager.default.isExecutableFile(atPath: p) { return p }
        }
        return nil
    }
}

enum AudioIO {
    static func requireFFmpeg() throws {
        guard Proc.which("ffmpeg") != nil, Proc.which("ffprobe") != nil else {
            throw ToolError(message: "ffmpeg and ffprobe must be installed (brew install ffmpeg)")
        }
    }

    /// Probe the first audio stream plus container facts.
    static func probe(_ path: String) throws -> (stream: [String: Any], format: [String: Any]) {
        try requireFFmpeg()
        let ffprobe = Proc.which("ffprobe")!
        let data = try Proc.run(ffprobe, [
            "-v", "error", "-show_streams", "-show_format", "-of", "json", path
        ])
        guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw ToolError(message: "ffprobe returned unreadable JSON")
        }
        let streams = (root["streams"] as? [[String: Any]]) ?? []
        guard let audio = streams.first(where: { ($0["codec_type"] as? String) == "audio" }) else {
            throw ToolError(message: "no audio stream in \(path)")
        }
        return (audio, (root["format"] as? [String: Any]) ?? [:])
    }

    /// Decode to raw PCM at a chosen format. `fmt` is an ffmpeg muxer name
    /// (f32le / s32le) and `codec` the matching PCM codec.
    static func decode(_ path: String, seconds: Double, rate: Int, channels: Int = 1,
                       fmt: String, codec: String) throws -> Data {
        let ffmpeg = Proc.which("ffmpeg")!
        return try Proc.run(ffmpeg, [
            "-v", "error", "-t", String(seconds), "-i", path, "-map", "0:a:0", "-vn",
            "-f", fmt, "-acodec", codec, "-ac", String(channels), "-ar", String(rate), "-"
        ])
    }

    static func decodeFloat(_ path: String, seconds: Double, rate: Int,
                            channels: Int = 1) throws -> [Float] {
        let d = try decode(path, seconds: seconds, rate: rate, channels: channels,
                           fmt: "f32le", codec: "pcm_f32le")
        return d.withUnsafeBytes { raw in
            Array(raw.bindMemory(to: Float.self))
        }
    }

    /// Decode to Float64. ffmpeg can emit f64le directly, avoiding a lossy widening
    /// of its f32 output (whose scale factor is 2^30.5 rather than 2^31).
    static func decodeDouble(_ path: String, seconds: Double, rate: Int,
                             channels: Int = 1) throws -> [Double] {
        let d = try decode(path, seconds: seconds, rate: rate, channels: channels,
                           fmt: "f64le", codec: "pcm_f64le")
        return d.withUnsafeBytes { raw in
            Array(raw.bindMemory(to: Double.self))
        }
    }

    static func decodeInt32(_ path: String, seconds: Double, rate: Int,
                            channels: Int = 1) throws -> [Int32] {
        let d = try decode(path, seconds: seconds, rate: rate, channels: channels,
                           fmt: "s32le", codec: "pcm_s32le")
        return d.withUnsafeBytes { raw in
            Array(raw.bindMemory(to: Int32.self))
        }
    }

    static func duration(_ path: String) throws -> Double {
        let (_, fmt) = try probe(path)
        return Double(fmt["duration"] as? String ?? "0") ?? 0
    }
}
