#!/usr/bin/env bash
set -euo pipefail

INSTALL_DIR="${EXTTY_INSTALL_DIR:-$HOME/.local/bin}"
BASE_URL="${EXTTY_BASE_URL:-https://github.com/YOUR_ORG/extty/releases/latest/download}"

detect_platform() {
    local os arch

    case "$(uname -s)" in
        Darwin) os="darwin" ;;
        Linux)  os="linux" ;;
        *)
            echo "Error: Unsupported operating system: $(uname -s)" >&2
            exit 1
            ;;
    esac

    case "$(uname -m)" in
        arm64|aarch64) arch="arm64" ;;
        x86_64|amd64)  arch="x64" ;;
        *)
            echo "Error: Unsupported architecture: $(uname -m)" >&2
            exit 1
            ;;
    esac

    if [[ "$os" == "darwin" && "$arch" == "x64" ]]; then
        echo "Warning: macOS x64 not available, attempting arm64 (Rosetta 2 required)" >&2
        arch="arm64"
    fi

    if [[ "$os" == "linux" && "$arch" == "arm64" ]]; then
        echo "Error: Linux arm64 not yet available" >&2
        exit 1
    fi

    echo "$os-$arch"
}

main() {
    local platform binary_name download_url temp_dir

    platform=$(detect_platform)
    binary_name="extty-$platform"

    echo "Detected platform: $platform"

    download_url="$BASE_URL/$binary_name.gz"

    temp_dir=$(mktemp -d)
    trap "rm -rf $temp_dir" EXIT

    echo "Downloading extty..."
    if command -v curl &> /dev/null; then
        curl -fsSL "$download_url" -o "$temp_dir/$binary_name.gz"
    elif command -v wget &> /dev/null; then
        wget -q "$download_url" -O "$temp_dir/$binary_name.gz"
    else
        echo "Error: curl or wget required" >&2
        exit 1
    fi

    echo "Extracting..."
    gunzip "$temp_dir/$binary_name.gz"
    chmod +x "$temp_dir/$binary_name"

    mkdir -p "$INSTALL_DIR"
    mv "$temp_dir/$binary_name" "$INSTALL_DIR/extty"

    echo ""
    echo "extty installed to $INSTALL_DIR/extty"

    if [[ ":$PATH:" != *":$INSTALL_DIR:"* ]]; then
        echo ""
        echo "Add $INSTALL_DIR to your PATH:"
        echo "  export PATH=\"\$PATH:$INSTALL_DIR\""
    fi

    echo ""
    echo "Run 'extty' to start"
}

main "$@"
