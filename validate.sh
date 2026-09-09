#!/usr/bin/env bash
# Thin wrapper around the real validator, which is Python and cross-platform.
#
# On Windows run the Python entrypoint directly:
#     python tools\validate.py --fresh
#
# Usage:  ./validate.sh [--fresh] [--quick] [--slice 1/2]
set -euo pipefail
cd "$(dirname "$0")"
exec "${PYTHON:-python3}" tools/validate.py "$@"
