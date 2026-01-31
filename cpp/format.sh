#!/bin/bash

# Format all C++ source files in the cpp directory using clang-format

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Find all C++ source files
find . -type f \( -name "*.cpp" -o -name "*.hpp" -o -name "*.h" -o -name "*.cc" -o -name "*.cxx" \) ! -path "./build/*" ! -path "./.*" -exec clang-format -i {} +
