import Foundation

/// Thresholds carried over from the validated Python implementation.
enum Tuning {
    static let probeSeconds = 90.0
    static let bandwidthNFFT = 16384
    static let bandwidthRefDB = -45.0
    static let gridBits = [8, 12, 16, 20, 24]
    static let gridTolerance = 1e-6
    /// A genuine HF component must clear this level relative to the 2-10 kHz band...
    static let topOctaveMinRelDB = -40.0
    /// ...and tower over that band's own median by this much.
    static let topOctaveContrastDB = 15.0
    static let hiresMinRate = 88200
    static let hiresMinBits = 24
    static let losslessCodecs: Set<String> = [
        "flac", "alac", "wavpack", "ape", "tak", "truehd", "mlp"
    ]
}

struct BitDepth {
    var gridBits: Int?
    var maxDeviation: Double?
    var note: String
    var peakInt32: Int64?
    var method = "quantisation grid (int32 decode)"
}

struct Bandwidth {
    var referenceDB: Double
    var edge45: Double?
    var edge60: Double?
    var ratioToNyquist: Double?
    var bandRel: [String: Double?]
}

struct Stability {
    var edges: [Double?]
    var mean: Double?
    var std: Double?
    var spreadPct: Double?
    var fixedFilter: Bool?
}

struct TopOctave {
    var applicable: Bool
    var note: String?
    var bandLow: Double?
    var bandHigh: Double?
    var peakRel: Double?
    var medianRel: Double?
    var contrast: Double?
    var present: Bool?
}

struct Verdict {
    var isHires: Bool
    var lossless: Bool
    /// nil means "container says hi-res but content could not be verified".
    var genuine: Bool?
    var label: String
    var reasons: [String]
}

struct AuditReport {
    var file: String
    var container: [String: Any]
    var declared: [String: Any]
    var bitDepth: BitDepth?
    var noiseFloorDB: Double?
    var peakDBFS: Double?
    var bandwidth: Bandwidth?
    var stability: Stability?
    var topOctave: TopOctave?
    var verdict: Verdict?

    func jsonObject() -> [String: Any] {
        var o: [String: Any] = ["file": file, "container": container, "declared": declared]
        if let b = bitDepth {
            o["bit_depth"] = [
                "grid_bits": b.gridBits as Any,
                "max_deviation": b.maxDeviation as Any,
                "note": b.note,
                "peak_int32": b.peakInt32 as Any,
                "method": b.method,
            ]
        }
        if let n = noiseFloorDB { o["noise_floor_db"] = n }
        if let p = peakDBFS { o["peak_dbfs"] = p }
        if let b = bandwidth {
            o["bandwidth"] = [
                "reference_level_db": b.referenceDB,
                "edge_-45db_hz": b.edge45 as Any,
                "edge_-60db_hz": b.edge60 as Any,
                "bandwidth_to_nyquist": b.ratioToNyquist as Any,
                "band_rel_db": b.bandRel.mapValues { $0 as Any },
            ]
        }
        if let s = stability {
            o["bandwidth_stability"] = [
                "edges_hz": s.edges.map { $0 as Any },
                "edge_mean_hz": s.mean as Any,
                "edge_std_hz": s.std as Any,
                "edge_relative_spread_pct": s.spreadPct as Any,
                "looks_like_fixed_filter": s.fixedFilter as Any,
            ]
        }
        if let t = topOctave {
            let band: Any = (t.bandLow != nil && t.bandHigh != nil)
                ? [t.bandLow!, t.bandHigh!] as [Double]
                : NSNull()
            o["top_octave"] = [
                "applicable": t.applicable,
                "note": t.note as Any,
                "band_hz": band,
                "band_peak_rel_db": t.peakRel as Any,
                "band_median_rel_db": t.medianRel as Any,
                "peak_above_local_floor_db": t.contrast as Any,
                "content_present": t.present as Any,
            ]
        }
        if let v = verdict {
            o["verdict"] = [
                "is_hires": v.isHires,
                "lossless": v.lossless,
                "genuine": v.genuine as Any,
                "label": v.label,
                "reasons": v.reasons,
            ]
        }
        return o
    }
}

enum Audit {

    // MARK: bit depth

    /// Coarsest quantisation grid the integer samples land on.
    ///
    /// Integer PCM from an N-bit master consists of exact multiples of
    /// 2^(32-N) counts, so a 16-bit master reads 16 even when wrapped in a 24-bit
    /// container. Float samples must never be passed here: ffmpeg's float output is
    /// scaled by 2^30.5 rather than 2^31 (measured off by sqrt(2)), which breaks a
    /// fixed-factor grid test.
    static func detectBitDepth(_ x: [Int32]) -> BitDepth {
        var out = BitDepth(gridBits: nil, maxDeviation: nil,
                           note: "no samples", peakInt32: nil)
        guard !x.isEmpty else { return out }
        var peak: Int64 = 0
        for v in x { peak = max(peak, Int64(abs(Int64(v)))) }
        out.peakInt32 = peak
        guard peak > 0 else { return out }
        let threshold = Double(peak) * 1e-4

        var best: Double?
        for bits in Tuning.gridBits {
            let step = Double(1 << (32 - bits))
            var maxDev = 0.0
            for v in x {
                let av = abs(Double(v))
                if av <= threshold { continue }        // ignore near-silence
                let q = Double(v) / step
                let dev = abs(q - q.rounded()) * step
                if dev > maxDev { maxDev = dev }
                if maxDev >= 1e-6 { break }            // already too coarse
            }
            if best == nil || maxDev < best! { best = maxDev }
            if maxDev < Tuning.gridTolerance {
                out.gridBits = bits
                out.maxDeviation = maxDev
                out.note = "coarsest matching grid"
                return out
            }
        }
        out.maxDeviation = best
        out.note = "no integer grid found — consistent with a lossy codec, a float "
                 + "master, or a resampled/dithered signal"
        return out
    }

    // MARK: bandwidth

    static func bandwidthProfile(_ x: [Double], rate: Int) -> Bandwidth? {
        guard var nFFT = Optional(Tuning.bandwidthNFFT), x.count >= nFFT * 2 else {
            // short input: shrink the transform so an edge can still be found
            var n = 4096
            while n * 4 < x.count && n < Tuning.bandwidthNFFT { n <<= 1 }
            guard x.count >= n * 2, let psd = DSP.averagePSD(x, nFFT: n, hop: n / 2),
                  psd.frames > 0 else { return nil }
            return profile(from: psd.psd, nFFT: n, rate: rate)
        }
        nFFT = min(nFFT, Tuning.bandwidthNFFT)
        guard let psd = DSP.averagePSD(x, nFFT: nFFT, hop: nFFT / 2), psd.frames > 0 else {
            return nil
        }
        return profile(from: psd.psd, nFFT: nFFT, rate: rate)
    }

    private static func profile(from psd: [Double], nFFT: Int, rate: Int) -> Bandwidth {
        let db = DSP.toDB(psd)
        let sm = DSP.movingAverage(db, 5)
        let binHz = Double(rate) / Double(nFFT)
        let nyquist = Double(rate) / 2
        func bin(_ hz: Double) -> Int { min(sm.count - 1, max(0, Int(hz / binHz))) }

        let refLo = bin(2000), refHi = bin(10000)
        var refSum = 0.0
        for k in refLo...max(refLo, refHi) { refSum += sm[k] }
        let ref = refSum / Double(max(1, refHi - refLo + 1))

        func edgeDB(_ thr: Double) -> Double? {
            let start = bin(13000), end = min(sm.count - 1, bin(nyquist * 0.999))
            guard start < end else { return nil }
            for k in start...end where sm[k] < ref + thr {
                return Double(k) * binHz
            }
            return nil
        }
        let e45 = edgeDB(Tuning.bandwidthRefDB)
        let e60 = edgeDB(-60)

        func rel(_ lo: Double, _ hi: Double) -> Double? {
            let a = bin(lo), b = bin(min(hi, nyquist))
            guard a <= b, b < sm.count else { return nil }
            var s = 0.0
            for k in a...b { s += sm[k] }
            return s / Double(b - a + 1) - ref
        }
        let bands: [String: Double?] = [
            "2-10khz": 0.0,
            "10-16khz": rel(10000, 16000),
            "16-19khz": rel(16000, 19000),
            "19-21khz": rel(19000, 21000),
            "21-23khz": rel(21000, 23000),
            "23-24khz": rel(23000, 24000),
            "24-30khz": rel(24000, 30000),
            "30-40khz": rel(30000, 40000),
        ]
        return Bandwidth(referenceDB: ref, edge45: e45, edge60: e60,
                         ratioToNyquist: e45.map { $0 / nyquist },
                         bandRel: bands)
    }

    static func bandwidthStability(_ x: [Double], rate: Int, windows: Int = 8) -> Stability? {
        let nFFT = 16384
        guard x.count >= nFFT * 4 else { return nil }
        let w = x.count / windows
        var edges: [Double?] = []
        var found: [Double] = []
        for k in 0..<windows {
            let seg = Array(x[(k * w)..<min((k + 1) * w, x.count)])
            guard seg.count >= nFFT * 2,
                  let psd = DSP.averagePSD(seg, nFFT: nFFT, hop: nFFT / 2),
                  psd.frames > 0 else { edges.append(nil); continue }
            let db = DSP.toDB(psd.psd)
            let sm = DSP.movingAverage(db, 5)
            let binHz = Double(rate) / Double(nFFT)
            func bin(_ hz: Double) -> Int { min(sm.count - 1, max(0, Int(hz / binHz))) }
            let refLo = bin(8000), refHi = bin(12000)
            var s = 0.0
            for j in refLo...max(refLo, refHi) { s += sm[j] }
            let ref = s / Double(max(1, refHi - refLo + 1))
            var e: Double?
            let start = bin(13000), end = min(sm.count - 1, bin(Double(rate) / 2 * 0.999))
            if start < end {
                for j in start...end where sm[j] < ref + Tuning.bandwidthRefDB {
                    e = Double(j) * binHz; break
                }
            }
            edges.append(e)
            if let e { found.append(e) }
        }
        guard found.count >= 2 else {
            return Stability(edges: edges, mean: nil, std: nil, spreadPct: nil,
                             fixedFilter: nil)
        }
        let mean = found.reduce(0, +) / Double(found.count)
        let variance = found.reduce(0) { $0 + ($1 - mean) * ($1 - mean) } / Double(found.count)
        let std = variance.squareRoot()
        return Stability(edges: edges, mean: mean, std: std,
                         spreadPct: std / mean * 100, fixedFilter: std / mean < 0.02)
    }

    // MARK: top octave (the decisive Hi-Res test)

    /// Energy in the top half of the passband (Nyquist/2 .. Nyquist), measured at the
    /// file's OWN rate. Probing above the file's rate would inject the resampler's own
    /// images above its Nyquist, which look exactly like content.
    ///
    /// Measured on fixtures: a genuine 96 kHz master with a 30 kHz tone scored a peak
    /// +68 dB above the 2-10 kHz reference; the same content lowpassed at 20 kHz and
    /// upsampled scored -91 dB.
    static func topOctave(_ x: [Double], rate: Int) -> TopOctave {
        let nyq = Double(rate) / 2
        guard nyq > 20000 else {
            return TopOctave(applicable: false,
                             note: String(format: "Nyquist %.0f Hz leaves no room above "
                                          + "the audible band", nyq))
        }
        let nFFT = 1 << 15
        guard x.count >= nFFT * 2,
              let psd = DSP.averagePSD(x, nFFT: nFFT, hop: nFFT / 2), psd.frames > 0 else {
            return TopOctave(applicable: false, note: "too little audio")
        }
        let db = DSP.toDB(psd.psd)
        let binHz = Double(rate) / Double(nFFT)
        func bin(_ hz: Double) -> Int { min(db.count - 1, max(0, Int(hz / binHz))) }
        let refLo = bin(2000), refHi = bin(10000)
        var s = 0.0
        for k in refLo...max(refLo, refHi) { s += db[k] }
        let ref = s / Double(max(1, refHi - refLo + 1))

        let lo = nyq * 0.5, hi = nyq * 0.99
        let a = bin(lo), b = bin(hi)
        guard a <= b, b < db.count else { return TopOctave(applicable: false) }
        var peak = -Double.greatestFiniteMagnitude
        var vals: [Double] = []
        for k in a...b { peak = max(peak, db[k]); vals.append(db[k]) }
        vals.sort()
        let median = vals[vals.count / 2]
        let peakRel = peak - ref
        let contrast = peak - median
        let present = peakRel > Tuning.topOctaveMinRelDB
                   && contrast > Tuning.topOctaveContrastDB
        return TopOctave(applicable: true, note: nil,
                         bandLow: lo, bandHigh: hi,
                         peakRel: peakRel, medianRel: median - ref,
                         contrast: contrast, present: present)
    }

    // MARK: verdict

    static func makeVerdict(codec: String?, rate: Int?, bits: Int?,
                            topOctavePresent: Bool?,
                            bandwidthRatio: Double?) -> Verdict {
        let c = (codec ?? "").lowercased()
        let lossless = Tuning.losslessCodecs.contains(c) || c.hasPrefix("pcm_")
        let isHires = (rate ?? 0) >= Tuning.hiresMinRate && lossless
        var reasons: [String] = []
        if let r = rate {
            if r < 44100 { reasons.append("sample rate \(r) Hz is below CD") }
            else if r == 44100 { reasons.append("44.1 kHz = CD quality, not Hi-Res") }
            else if r <= 48000 {
                reasons.append("\(r) Hz is at or below 48 kHz; Hi-Res requires >48 kHz")
            }
        } else { reasons.append("sample rate unknown") }
        if !lossless { reasons.append("codec \(codec ?? "?") is lossy; Hi-Res requires lossless") }
        if let b = bits, b < Tuning.hiresMinBits {
            reasons.append("effective bit depth \(b) < 24")
        }
        var genuine: Bool? = nil
        if isHires {
            genuine = true
            if let r = rate, r > 48000, topOctavePresent == false {
                genuine = false
                reasons.append("no content above the source's own Nyquist: upsampled "
                             + "from a lower rate")
            }
            if let br = bandwidthRatio, let r = rate, r >= Tuning.hiresMinRate {
                let expected = 20000.0 / (Double(r) / 2)
                if br < max(0.45, expected * 0.9) {
                    genuine = false
                    reasons.append(String(format: "bandwidth only %.2fx Nyquist: the top "
                                         + "octaves are empty", br))
                }
            }
            if topOctavePresent == nil { genuine = nil }
        }
        let label: String
        if !isHires { label = "not hi-res" }
        else if genuine == false { label = "hi-res container, questionable content" }
        else if genuine == nil { label = "hi-res by container (content unverified)" }
        else { label = "hi-res (verified: content in the top octave)" }
        return Verdict(isHires: isHires, lossless: lossless, genuine: genuine,
                       label: label, reasons: reasons)
    }

    // MARK: top level

    static func run(path: String, seconds: Double) throws -> AuditReport {
        let (stream, format) = try AudioIO.probe(path)
        let rate = Int(stream["sample_rate"] as? String ?? "") ?? 0
        let codec = stream["codec_name"] as? String
        var container: [String: Any] = [
            "format_name": format["format_name"] as? String ?? "?",
            "duration_s": Double(format["duration"] as? String ?? "0") ?? 0,
        ]
        if let sz = format["size"] as? String, let d = container["duration_s"] as? Double, d > 0 {
            container["size_bytes"] = Int(sz) ?? 0
            container["measured_bitrate"] = Int((Double(sz) ?? 0) * 8 / d)
        }
        var declared: [String: Any] = [
            "codec": codec as Any,
            "sample_rate": rate,
            "channels": Int(stream["channels"] as? Int ?? 0),
        ]
        if let br = stream["bit_rate"] as? String, let v = Int(br) { declared["stream_bitrate"] = v }

        var report = AuditReport(file: (path as NSString).lastPathComponent,
                                 container: container, declared: declared,
                                 bitDepth: nil, noiseFloorDB: nil, peakDBFS: nil,
                                 bandwidth: nil, stability: nil, topOctave: nil,
                                 verdict: nil)
        guard rate > 0 else { return report }

        let x = try AudioIO.decodeDouble(path, seconds: seconds, rate: rate)
        guard !x.isEmpty else { return report }
        report.peakDBFS = DSP.peakDBFS(x)
        report.noiseFloorDB = DSP.noiseFloorDB(x)

        let xi = try AudioIO.decodeInt32(path, seconds: seconds, rate: rate)
        report.bitDepth = detectBitDepth(xi)
        report.bandwidth = bandwidthProfile(x, rate: rate)
        report.stability = bandwidthStability(x, rate: rate)
        let top = topOctave(x, rate: rate)
        report.topOctave = top
        report.verdict = makeVerdict(codec: codec, rate: rate,
                                     bits: report.bitDepth?.gridBits,
                                     topOctavePresent: top.applicable ? top.present : nil,
                                     bandwidthRatio: report.bandwidth?.ratioToNyquist)
        return report
    }
}
