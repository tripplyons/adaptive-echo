#!/bin/bash
# Setup script for worktree environment
# Sets up environment variables needed for building/running
# Copies build caches from root worktree to speed up builds
# Uses $ROOT_WORKTREE_PATH provided by Cursor

set -e

# Get the worktree root directory (where this script is located)
SETUP_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKTREE_ROOT="$(cd "${SETUP_SCRIPT_DIR}/.." && pwd)"

# Export worktree root for use by other scripts
export WORKTREE_ROOT

# ROOT_WORKTREE_PATH is provided by Cursor and points to the main worktree
# Use it if available, otherwise fall back to current directory
ROOT_WORKTREE="${ROOT_WORKTREE_PATH:-${WORKTREE_ROOT}}"

echo "Setting up worktree..."
echo "  Worktree root: ${WORKTREE_ROOT}"
echo "  Root worktree: ${ROOT_WORKTREE}"

# Copy build caches from root worktree to speed up builds
if [ "${ROOT_WORKTREE}" != "${WORKTREE_ROOT}" ] && [ -d "${ROOT_WORKTREE}" ]; then
    echo "Copying build caches from root worktree..."
    
    # Copy C++ CMake build cache (if it exists and is not too large)
    ROOT_CPP_BUILD="${ROOT_WORKTREE}/cpp/build"
    WORKTREE_CPP_BUILD="${WORKTREE_ROOT}/cpp/build"
    if [ -d "${ROOT_CPP_BUILD}" ]; then
        # Check size before copying (limit to 500MB to avoid copying huge caches)
        BUILD_SIZE=$(du -sm "${ROOT_CPP_BUILD}" 2>/dev/null | cut -f1 || echo "0")
        if [ "${BUILD_SIZE}" -lt 500 ]; then
            echo "  Copying C++ build cache (${BUILD_SIZE}MB)..."
            mkdir -p "${WORKTREE_ROOT}/cpp"
            # Use rsync if available for efficient copying, otherwise cp
            if command -v rsync >/dev/null 2>&1; then
                rsync -a --delete "${ROOT_CPP_BUILD}/" "${WORKTREE_CPP_BUILD}/" 2>/dev/null || true
            else
                cp -r "${ROOT_CPP_BUILD}" "${WORKTREE_CPP_BUILD}" 2>/dev/null || true
            fi
        else
            echo "  Skipping C++ build cache (too large: ${BUILD_SIZE}MB)"
        fi
    fi
    
    # Copy Python virtual environment (if it exists)
    ROOT_VENV="${ROOT_WORKTREE}/adaptive_echo_jax/.venv"
    WORKTREE_VENV="${WORKTREE_ROOT}/adaptive_echo_jax/.venv"
    if [ -d "${ROOT_VENV}" ]; then
        VENV_SIZE=$(du -sm "${ROOT_VENV}" 2>/dev/null | cut -f1 || echo "0")
        if [ "${VENV_SIZE}" -lt 1000 ]; then
            echo "  Copying Python virtual environment (${VENV_SIZE}MB)..."
            mkdir -p "${WORKTREE_ROOT}/adaptive_echo_jax"
            if command -v rsync >/dev/null 2>&1; then
                rsync -a --delete "${ROOT_VENV}/" "${WORKTREE_VENV}/" 2>/dev/null || true
            else
                cp -r "${ROOT_VENV}" "${WORKTREE_VENV}" 2>/dev/null || true
            fi
        else
            echo "  Skipping Python venv (too large: ${VENV_SIZE}MB)"
        fi
    fi
fi

# Set up Python dependencies with uv if .venv doesn't exist
if [ ! -d "${WORKTREE_ROOT}/adaptive_echo_jax/.venv" ]; then
    echo "Setting up Python dependencies..."
    cd "${WORKTREE_ROOT}/adaptive_echo_jax"
    if command -v uv >/dev/null 2>&1; then
        uv sync --no-dev || uv pip install -e . || echo "Warning: Failed to install Python dependencies"
    else
        echo "Warning: uv not found, skipping Python dependency installation"
    fi
    cd "${WORKTREE_ROOT}"
fi

# VST3_SDK path - only set if not already set and if it exists in a relative location
# Don't use global paths - user should set this themselves if needed
if [ -z "$VST3_SDK" ]; then
    # Check for VST3 SDK in common relative locations within worktree
    # If not found, leave unset (will be handled by build scripts)
    if [ -d "${WORKTREE_ROOT}/external/vst3sdk" ]; then
        export VST3_SDK="${WORKTREE_ROOT}/external/vst3sdk"
    elif [ -d "${WORKTREE_ROOT}/plugin/external/vst3sdk" ]; then
        export VST3_SDK="${WORKTREE_ROOT}/plugin/external/vst3sdk"
    fi
fi

echo "Worktree setup complete!"
