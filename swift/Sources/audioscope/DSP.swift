import Accelerate
import Foundation

/// Real-input FFT built on Accelerate's vDSP, in **double** precision.
///
/// Precision is not a detail here. A genuine 96 kHz master whose content ends at
/// 20 kHz has essentially nothing above 24 kHz, and the peak must be measured
/// against that near-zero floor. A float32 FFT has a numerical floor around
/// -140 dB per bin, which swamps the real floor in exactly the band this tool uses
/// to tell a genuine Hi-Res master from an upsampled one: measured on one fixture,
/// float32 reported the band peak 24 dB too high and flipped the verdict from
/// "questionable" to "genuine". float64 puts the numerical floor far below the signal.
///
/// vDSP packs a real transform as split-complex pairs with DC and Nyquist sharing
/// element 0 (DC in the real part, Nyquist in the imaginary part). Everything below
/// unpacks that into a plain magnitude-squared array of n/2+1 bins.
final class RealFFT {
    let n: Int
    let log2n: vDSP_Length
    private let setup: FFTSetupD

    init?(_ n: Int) {
        guard n >= 4, n & (n - 1) == 0 else { return nil }
        self.n = n
        self.log2n = vDSP_Length(log2(Double(n)).rounded())
        guard let s = vDSP_create_fftsetupD(log2n, FFTRadix(kFFTRadix2)) else { return nil }
        self.setup = s
    }

    deinit { vDSP_destroy_fftsetupD(setup) }

    /// Magnitude squared per bin for a real signal of length `n`; n/2 + 1 bins.
    func magnitudesSquared(_ input: [Double]) -> [Double] {
        precondition(input.count == n, "input must be exactly n samples")
        var out = [Double](repeating: 0, count: n / 2 + 1)
        var re = [Double](repeating: 0, count: n / 2)
        var im = [Double](repeating: 0, count: n / 2)

        input.withUnsafeBufferPointer { inPtr in
            re.withUnsafeMutableBufferPointer { reB in
                im.withUnsafeMutableBufferPointer { imB in
                    var split = DSPDoubleSplitComplex(realp: reB.baseAddress!,
                                                      imagp: imB.baseAddress!)
                    inPtr.baseAddress!.withMemoryRebound(to: DSPDoubleComplex.self,
                                                         capacity: n / 2) { cp in
                        vDSP_ctozD(cp, 2, &split, 1, vDSP_Length(n / 2))
                    }
                    vDSP_fft_zripD(setup, &split, 1, log2n, FFTDirection(FFT_FORWARD))
                    out[0] = reB[0] * reB[0]
                    out[n / 2] = imB[0] * imB[0]
                    for k in 1..<(n / 2) {
                        let r = reB[k], i = imB[k]
                        out[k] = r * r + i * i
                    }
                }
            }
        }
        return out
    }
}

enum DSP {
    /// Periodic Hann window, matching numpy.hanning(n+1)[:n].
    static func hann(_ n: Int) -> [Double] {
        var w = [Double](repeating: 0, count: n)
        let denom = Double(n)          // numpy.hanning(n+1) divides by n
        for i in 0..<n {
            w[i] = 0.5 - 0.5 * cos(2 * Double.pi * Double(i) / denom)
        }
        return w
    }

    /// Welch-style average power spectral density, matching the Python reference:
    /// per frame, |X|^2 / sum(w^2), averaged over frames.
    ///
    /// Scaling note: `vDSP_fft_zripD` returns twice the true DFT, so magnitudes are
    /// halved before squaring. A constant factor cancels in every dB ratio this tool
    /// reports, but dropping it makes the absolute PSD wrong and would silently break
    /// any future absolute-level check.
    static func averagePSD(_ x: [Double], nFFT: Int, hop: Int) -> (psd: [Double], frames: Int)? {
        guard let fft = RealFFT(nFFT), x.count >= nFFT else { return nil }
        let w = hann(nFFT)
        let wss = w.reduce(0) { $0 + $1 * $1 }
        var acc = [Double](repeating: 0, count: nFFT / 2 + 1)
        var frame = [Double](repeating: 0, count: nFFT)
        var frames = 0
        var i = 0
        while i + nFFT <= x.count {
            for k in 0..<nFFT { frame[k] = x[i + k] * w[k] }
            let m = fft.magnitudesSquared(frame)
            for k in 0..<m.count { acc[k] += m[k] * 0.25 }   // (1/2)^2 for vDSP's 2x
            frames += 1
            i += hop
        }
        guard frames > 0 else { return nil }
        let scale = 1.0 / (Double(frames) * wss)
        for k in 0..<acc.count { acc[k] *= scale }
        return (acc, frames)
    }

    static func toDB(_ psd: [Double]) -> [Double] {
        psd.map { 10 * log10(max($0, 1e-300)) }
    }

    /// Centred moving average. Uses an exact window rather than a running sum so the
    /// result does not drift from numpy's `convolve(..., mode="same")`.
    static func movingAverage(_ x: [Double], _ width: Int) -> [Double] {
        guard width > 1, x.count > width else { return x }
        let half = width / 2
        var out = [Double](repeating: 0, count: x.count)
        for i in 0..<x.count {
            let lo = max(0, i - half), hi = min(x.count - 1, i + half)
            var s = 0.0
            for k in lo...hi { s += x[k] }
            out[i] = s / Double(hi - lo + 1)
        }
        return out
    }

    /// Mean power of the quietest `percentile`% of short frames, in dBFS.
    static func noiseFloorDB(_ x: [Double], percentile: Double = 5.0) -> Double? {
        let n = 1 << 14
        guard x.count >= n else { return nil }
        var powers: [Double] = []
        var i = 0
        while i + n <= x.count && powers.count < 4000 {
            var s = 0.0
            for k in 0..<n { let v = x[i + k]; s += v * v }
            powers.append(s / Double(n))
            i += n / 2
        }
        let pos = powers.filter { $0 > 0 }.sorted()
        guard !pos.isEmpty else { return nil }
        let idx = min(pos.count - 1, max(0, Int(Double(pos.count) * percentile / 100.0)))
        return 10 * log10(max(pos[idx], 1e-300))
    }

    /// dBFS peak of the signal.
    static func peakDBFS(_ x: [Double]) -> Double {
        var peak = 0.0
        for v in x { peak = max(peak, abs(v)) }
        return 20 * log10(max(peak, 1e-12))
    }
}
