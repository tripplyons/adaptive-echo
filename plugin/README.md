# Adaptive Echo Plugin

## Note

Any paths referenced in this document are relative to the folder containing this README (`adaptive-echo/plugin/`).

## Requirements

- Install CMake from [here](https://cmake.org/download/) or your preferred package manager.
- Install Ninja from [here](https://ninja-build.org/) or your preferred package manager.
- Install Clang from [here](https://clang.llvm.org/) or your preferred package manager.

### VST3 Support

- Download the [VST3 SDK](https://www.steinberg.net/en/company/developers.html) and extract it to `external/VST3_SDK`.

### MacOS

#### Homebrew as Package Manager (Recommended)

```bash
brew install cmake ninja llvm
```

### Linux

I have not compiled on Linux yet, but I will do so soon.

### Windows

I have not compiled on Windows yet, but I will do so soon.

## Build

```bash
cmake -S . -B build -G Ninja
cmake --build build
```

This will build the plugin to `build/AdaptiveEcho_artefacts/`.

## Run

To run the standalone version of the plugin, run the binary you found in `build/AdaptiveEcho_artefacts/Standalone/`.

To run a VST3/AU version of the plugin, you can find the plugin files in `build/AdaptiveEcho_artefacts/VST3/` or `build/AdaptiveEcho_artefacts/AU/`.

