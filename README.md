# asmr-spectrum

Lossless audio concatenation plus spectral, low-frequency, loudness and
energy-distribution analysis — pure **ffmpeg + NumPy**, no Python audio library.

It was built to answer a narrow question about long ASMR recordings: *where is
the energy, and how much of it is infrasonic?* Getting that number right is
surprisingly easy to get wrong, so the project ships with a self-test that
generates a signal with known content and checks the answers against closed-form
values.

## Why the low-frequency numbers need care

Three estimators are computed for the same file, and the disagreement between
them is a feature, not a bug:

| estimator | method | what it is good for |
|---|---|---|
| `exact` | non-overlapping 1 s Hann frames, `|X|²/Σw²`, summed | the authoritative band share |
| `psd_welch` | bins of a 16 384-point (0.34 s) Welch PSD | the analytic spectrum, cheap per-band detail |
| `time_resolved` | 1 s aligned block sums | per-second trends and source attribution |

On a measured 20 Hz share of 11.5 % these gave **11.47 / 14.67 / 10.72 %** — the
Welch band sum reads ~28 % high in relative terms. Two pitfalls cause that:

1. **Window length vs. the cutoff.** At 20 Hz a 0.34 s window holds ~7 cycles and
   its 2.9 Hz main lobe smears energy across the cutoff. A 1 s window (~20
   cycles) measures a 30 Hz tone as 0.0 % below 20 Hz.
2. **Phase cancellation.** Analysing a whole file as a *single* window lets two
   distant copies of one tone interfere. On the test fixture, one 30 Hz tone in
   two separate segments read **0.67 %** where the truth is **28.57 %**;
   averaging frames recovers the exact value.

`analyze` always reports all three under `estimator_spread_pct`, so you can see
the disagreement instead of trusting one number.

## Install

```bash
git clone https://github.com/shiny-p/asmr-spectrum.git
cd asmr-spectrum
python3 -m venv .venv
.venv/bin/pip install -e .
```

`ffmpeg` and `ffprobe` must be on `PATH` (`brew install ffmpeg`). Python ≥ 3.9.

## Usage

```bash
# 1. concatenate: decode to a pipe, encode exactly once. No trimming, no
#    crossfades, no padding, no gain changes. Mixed input rates are unified.
asmr-spectrum concat work.m4a part1.aac part2.mp3 part3.aac --write-manifest order.txt

# 2. confirm the output length equals the sum of its parts
asmr-spectrum verify work.m4a --manifest order.txt

# 3. full analysis: JSON + CSV + a six-panel figure
asmr-spectrum analyze work.m4a -o results --label work --from-files part1.aac part2.mp3

# quick level + low-frequency table for many files, no plots
asmr-spectrum scan *.aac --json scan.json

# parameter audit: sample rate, effective bit depth, codec, bitrate, HF content
asmr-spectrum audit track.aac
asmr-spectrum audit ~/Music/Music/Media.localized/"Apple Music"/Artist/Album/*.movpkg
asmr-spectrum audit *.aac --json audit.json
```

`--from-files` derives per-source time boundaries from the inputs' durations and
reports each one's own band shares, level and peak. `--source t0,t1,name` does
the same with explicit ranges.

## Auditing a file (`audit`)

Answers "what is this file, really" — including whether a Hi-Res label is backed
by actual content:

| reported | how it is established |
|---|---|
sample rate | container metadata, cross-checked against the codec configuration box |
**effective bit depth** | the quantisation grid the samples land on, from an **int32** decode |
codec / losslessness | container metadata against a known lossless list |
bitrate | stream metadata plus measured `size × 8 / duration` |
**bandwidth** | spectral edge relative to the 2–10 kHz band, and whether it is constant over time |
**HF content** | energy in the top half of the passband (Nyquist/2 … Nyquist), measured **at the file's own rate** |
verdict | `not hi-res` / `hi-res container, questionable content` / `hi-res by container (content unverified)` / `hi-res (verified)` |

Three measurement traps this tool is built around, each found by testing:

1. **Never grid-test float samples.** ffmpeg's float output is not scaled by 2³¹ —
   it measured 2³⁰·⁵, off by √2 — so a fixed scale factor misreads the grid. The
   audit decodes **int32** instead, where the grid is exact.
2. **A genuine ultrasonic component is a narrowband peak, not a band average.**
   On a real 96 kHz master with a 30 kHz tone the 24–48 kHz band *mean* sat 0.2 dB
   above the mid-band reference while the band *peak* sat 58 dB above it. The test
   therefore requires a peak that both clears an absolute level and towers over
   that band's own median.
3. **Do not probe above the file's rate to look for HF content.** Upsampling
   injects the resampler's own images above the source Nyquist, which look exactly
   like content. Measuring the top half of the passband at the native rate avoids
   this entirely.

Apple Music `*.movpkg` downloads are handled too: their audio is FairPlay
encrypted, so they are described from `boot.xml` / `m3u8` / init-fragment
manifests (codec, rate, depth, channels, bitrate, and which variant was
downloaded) and the audit states plainly that bandwidth and HF content could not
be measured instead of guessing.

Worked examples from real files: a "Hi-Res"-labelled stream turned out to be
48 kHz AAC at 19.3 kHz bandwidth; a "192 kHz / 24 bit / 9216 kbps" label was
refuted by a 48 kHz / 166 kbps AAC payload 45× smaller than the claim implies;
a genuine 96 kHz/24-bit ALAC stream was confirmed, then marked content-unverified
because DRM blocked the HF test.

## Outputs

For `<prefix>`:

| file | contents |
|---|---|
| `<prefix>.json` | every summary number, all three estimator variants, per-source attribution |
| `<prefix>_psd.csv` | 2.93 Hz-resolution spectrum with the cumulative percentage |
| `<prefix>_bands.csv` | band table: share, mean level, peak frequency and its note name |
| `<prefix>_loudness.csv` | per-second L/R RMS and peak |
| `<prefix>_trend.csv` | per-second energy in all 16 bands, plus the <20 Hz and <200 Hz shares |
| `<prefix>.png` | average spectrum, cumulative energy, band shares, 0–250 Hz detail, 1–300 Hz detail, loudness over time, loudness histogram, low-frequency share over time |

## Measurement notes

* Everything is decoded to 48 kHz stereo float so mixed input rates are comparable.
* Loudness uses **energy-weighted** averages (`10·log10(mean power)`), never a mean
  of dB values, which would read ~3 dB high on non-stationary material.
* Spectrum flatness is reported per band: ≈1 means noise-like, ≪1 would reveal a
  resonant tone. Typical close-mic'd material sits near 1 everywhere below 1 kHz.
* A concatenation re-encodes to AAC-LC 192k. That step was measured to be
  transparent below 20 Hz (a 60 s excerpt moved 11.70 % → 11.74 %); lossless
  intermediates are unnecessary but `--codec flac` is available.

## Test

```bash
.venv/bin/python tests/validate.py
```

Builds a 12 s fixture (30 Hz, 100 Hz, 1 kHz, 6 kHz, 15 kHz, silence, 300 Hz,
30 Hz, then silence — one event per second, equal energy each) and asserts:

* the dominant band of every second matches the tone that is playing
* per-second RMS is exactly −9.03 dBFS, silence gates out, crest factor is 5.37 dB
* band shares below each cutoff equal the closed-form **n/7** exactly (28.5714 %,
  42.8571 %, 71.4286 %, 85.7143 %, 100 %)
* a 30 Hz tone does not leak below 20 Hz
* the cumulative curve is monotonic and ends at 100 %

Twenty checks, no audio is committed to the repository.

## Layout

```
src/asmr_spectrum/
  config.py     sample rate, band definitions, small helpers
  audit.py      parameter audit: bit depth grid, bandwidth, HF content, verdict
  movpkg.py     Apple Music .movpkg (encrypted HLS) description from manifests
  audio.py      ffprobe/ffmpeg probing and streamed decoding
  core.py       Welch spectrum, per-block bands, exact band-limited energy, loudness
  report.py     derived tables, JSON/CSV writers, source attribution
  plotting.py   the six-panel figure
  spectrum.py   per-file pipeline + one-glance scan
  concat.py     pipe-decode / single-encode concatenation
  sources.py    source boundaries and manifests
  cli.py        argparse entry points
tests/validate.py         self-test for the spectral analysis
tests/validate_audit.py   self-test for the parameter audit
scripts/run_tests.sh
```

## License

MIT — see [LICENSE](LICENSE).
