#!/usr/bin/env bash
set -euo pipefail

export PATH="$HOME/.bun/bin:$PATH"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
DIST_DIR="$ROOT_DIR/dist"

VERSION=$(node -p "require('$ROOT_DIR/package.json').version")

echo "Building extty v$VERSION"

cd "$ROOT_DIR"

rm -rf "$DIST_DIR"/*.gz "$DIST_DIR"/extty-*

TARGETS=(
    "bun-darwin-arm64"
    "bun-linux-x64"
)

for target in "${TARGETS[@]}"; do
    platform="${target#bun-}"
    output_name="extty-$platform"

    echo "Building for $platform..."
    bun build src/index.tsx --compile --target="$target" --outfile="$DIST_DIR/$output_name"

    echo "Compressing $output_name..."
    gzip -9 -k "$DIST_DIR/$output_name"

    mv "$DIST_DIR/$output_name.gz" "$DIST_DIR/$output_name-v$VERSION.gz"
    rm "$DIST_DIR/$output_name"

    echo "  → $DIST_DIR/$output_name-v$VERSION.gz"
done

echo ""
echo "Build complete:"
ls -lh "$DIST_DIR"/*.gz

echo ""
echo "Checksums:"
cd "$DIST_DIR"
shasum -a 256 *.gz > checksums.txt
cat checksums.txt
