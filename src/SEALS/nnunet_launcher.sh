#!/bin/bash
# Wrapper script for SEALS nnUNet launcher
# Defaults to v1 (Docker-compatible), can be switched to v2 with --version v2

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERSION="v1"  # Default to v1

# Parse arguments - extract --version if present
ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --version)
            VERSION="$2"
            shift 2
            ;;
        *)
            # Collect remaining arguments
            ARGS+=("$1")
            shift
            ;;
    esac
done

# Call the appropriate versioned launcher
if [ "$VERSION" = "v1" ]; then
    exec "$SCRIPT_DIR/nnunet_launcher_v1.sh" "${ARGS[@]}"
elif [ "$VERSION" = "v2" ]; then
    exec "$SCRIPT_DIR/nnunet_launcher_v2.sh" "${ARGS[@]}"
else
    echo "Error: Invalid version '$VERSION'. Must be 'v1' or 'v2'." >&2
    exit 1
fi

