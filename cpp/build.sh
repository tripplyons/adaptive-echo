#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"

# Detect number of CPU cores
if command -v nproc &> /dev/null; then
    NUM_JOBS=$(nproc)
elif command -v sysctl &> /dev/null; then
    NUM_JOBS=$(sysctl -n hw.ncpu)
else
    NUM_JOBS=4
fi

# Clean previous build
rm -rf "${BUILD_DIR}"
mkdir -p "${BUILD_DIR}"

# Configure and build
cd "${BUILD_DIR}"
CMAKE_ARGS=(
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_CXX_FLAGS_RELEASE="-O3 -DNDEBUG -ffast-math -funroll-loops -fomit-frame-pointer -ftree-vectorize"
    -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON
)
if [ -n "$VST3_SDK" ]; then
    CMAKE_ARGS+=(-DVST3_SDK="$VST3_SDK")
fi
cmake .. "${CMAKE_ARGS[@]}" > /dev/null

make -j${NUM_JOBS} > /dev/null

echo "Build successful"
