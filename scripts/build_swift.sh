#!/bin/bash
# Build the native Swift `audioscope` binary and install it into ~/bin.
#
# Usage:  scripts/build_swift.sh [install-dir]      (default: ~/bin)
#
# The binary is self-contained apart from ffmpeg/ffprobe, which it shells out to for
# decoding (matching the Python implementation's approach).
set -euo pipefail
cd "$(dirname "$0")/../swift"

DEST="${1:-$HOME/bin}"

echo "== building (release)"
swift build -c release

BIN=".build/release/audioscope"
[ -x "$BIN" ] || { echo "build produced no binary at $BIN" >&2; exit 1; }

echo "== verifying"
"$BIN" --version

mkdir -p "$DEST"
install -m 755 "$BIN" "$DEST/audioscope"
echo "== installed $DEST/audioscope"
case ":$PATH:" in
  *":$DEST:"*) echo "   $DEST is on PATH — run: audioscope <file>" ;;
  *)           echo "   NOTE: $DEST is not on PATH" ;;
esac

# Cross-check against the validated Python reference, if a venv is present.
ROOT="$(cd .. && pwd)"
TEST="$ROOT/tests/crosscheck_swift.py"
for cand in "$ROOT/.venv/bin/python" "$ROOT/swift/.venv/bin/python"; do
  if [ -x "$cand" ] && [ -f "$TEST" ]; then
    echo "== cross-checking against the Python reference"
    (cd "$ROOT" && "$cand" "$TEST" "$ROOT/swift/$BIN") || true
    break
  fi
done
