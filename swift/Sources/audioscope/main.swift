import Foundation

let usage = """
audioscope — audio parameter audit for macOS

USAGE
  audioscope [options] <file> [<file> ...]

Reports sample rate, effective bit depth, codec, bitrate, bandwidth and
high-frequency content, then judges whether a Hi-Res label is backed by real
content. Decoding uses ffmpeg/ffprobe; the analysis is native (Accelerate).

OPTIONS
  --json <path>     write results as JSON (one object, or an array for many files)
  --seconds <n>     audio analysed for bit depth / bandwidth (default 90)
  --shallow         container metadata only, skip signal analysis
  --quiet           suppress per-file progress notes
  -h, --help        show this help
  --version         show the version

EXIT STATUS
  0 all files audited, 1 at least one failed
"""

struct Options {
    var files: [String] = []
    var jsonPath: String?
    var seconds = Tuning.probeSeconds
    var shallow = false
    var quiet = false
}

func parseArgs(_ argv: [String]) throws -> Options {
    var o = Options()
    var i = 0
    while i < argv.count {
        let a = argv[i]
        switch a {
        case "-h", "--help":
            print(usage); exit(0)
        case "--version":
            print("audioscope 1.0.0"); exit(0)
        case "--json":
            i += 1
            guard i < argv.count else { throw ToolError(message: "--json needs a path") }
            o.jsonPath = argv[i]
        case "--seconds":
            i += 1
            guard i < argv.count, let v = Double(argv[i]) else {
                throw ToolError(message: "--seconds needs a number")
            }
            o.seconds = v
        case "--shallow":
            o.shallow = true
        case "--quiet":
            o.quiet = true
        default:
            if a.hasPrefix("--") { throw ToolError(message: "unknown option \(a)") }
            o.files.append(a)
        }
        i += 1
    }
    guard !o.files.isEmpty else { throw ToolError(message: "no input files\n\n" + usage) }
    return o
}

func fmt(_ v: Double?, _ decimals: Int = 1, suffix: String = "") -> String {
    guard let v else { return "n/a" }
    return String(format: "%.\(decimals)f", v) + suffix
}

func printReport(_ r: AuditReport) {
    let d = r.declared
    let rate = d["sample_rate"] as? Int ?? 0
    let codec = d["codec"] as? String ?? "?"
    let container = r.container["format_name"] as? String ?? "?"
    let duration = r.container["duration_s"] as? Double ?? 0

    print("file            : \(r.file)")
    print("container       : \(container)")
    print("codec           : \(codec)")
    print("sample rate     : \(rate) Hz" + (rate > 0 ? "  (Nyquist \(rate / 2) Hz)" : ""))

    if let b = r.bitDepth {
        let eff = b.gridBits.map { "\($0) bit" } ?? "none (lossy/float)"
        print("bit depth       : effective \(eff)  [\(b.method)]")
        if b.gridBits == nil {
            print("                  \(b.note)")
        }
    }
    print("channels        : \(d["channels"] as? Int ?? 0)")
    if let br = d["stream_bitrate"] as? Int {
        var line = "bitrate         : stream \(br)"
        if let m = r.container["measured_bitrate"] as? Int {
            line += " | measured \(m / 1000) kbps"
        }
        print(line)
    }
    print("duration        : \(fmt(duration / 60, 2)) min")
    print("peak            : \(fmt(r.peakDBFS, 2, suffix: " dBFS"))")
    print("noise floor     : \(fmt(r.noiseFloorDB, 1, suffix: " dBFS")) (quietest 5% of frames)")

    if let bw = r.bandwidth {
        let e45 = bw.edge45.map { String(format: "%.0f Hz", $0) } ?? "no edge found"
        var line = "bandwidth       : \(e45) (-45 dB)"
        if let e60 = bw.edge60 { line += String(format: " | %.0f Hz (-60 dB)", e60) }
        print(line)
        if let ratio = bw.ratioToNyquist {
            let kind = r.stability?.fixedFilter == true
                ? "FIXED FILTER (lossy/lowpassed)"
                : "varies with content"
            print(String(format: "bandwidth       : %.3f x Nyquist   %@", ratio, kind))
        } else {
            print("bandwidth       : no spectral edge in band")
        }
    }

    if let t = r.topOctave {
        if t.applicable, let lo = t.bandLow, let hi = t.bandHigh {
            let state = (t.present == true) ? "PRESENT"
                : (rate >= Tuning.hiresMinRate
                    ? "ABSENT — upsampled from a lower rate"
                    : "ABSENT — band-limited (lowpass/lossy codec)")
            print(String(format: "HF content      : top half of passband [%.0f, %.0f] Hz -> %@",
                         lo, hi, state))
            print(String(format: "                  peak %@ dB rel. ref, %@ dB over local floor",
                         fmt(t.peakRel), fmt(t.contrast)))
        } else {
            print("HF content      : not applicable — \(t.note ?? "n/a")")
        }
    }

    if let v = r.verdict {
        print("verdict         : \(v.label)")
        for reason in v.reasons { print("                  - \(reason)") }
    }
}

do {
    let opts = try parseArgs(Array(CommandLine.arguments.dropFirst()))
    var objects: [[String: Any]] = []
    var failed = 0
    for path in opts.files {
        do {
            let report = try Audit.run(path: path, seconds: opts.seconds)
            objects.append(report.jsonObject())
            if opts.jsonPath == nil {
                if opts.files.count > 1 { print(String(repeating: "=", count: 72)) }
                printReport(report)
            }
        } catch {
            FileHandle.standardError.write(Data("error: \(path): \(error)\n".utf8))
            failed += 1
        }
    }
    if let jp = opts.jsonPath {
        let payload: Any = objects.count == 1 ? objects[0] : objects
        let data = try JSONSerialization.data(withJSONObject: payload,
                                             options: [.prettyPrinted, .sortedKeys])
        try data.write(to: URL(fileURLWithPath: jp))
        print("wrote \(jp)")
    }
    exit(failed == 0 ? 0 : 1)
} catch {
    FileHandle.standardError.write(Data("error: \(error)\n".utf8))
    exit(2)
}
