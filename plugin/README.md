# Adaptive Echo Plugin

## Note

Any paths referenced in this document are relative to the folder containing this README (`adaptive-echo/plugin/`).

## Requirements

### All Platforms

- Download the [VST3 SDK](https://www.steinberg.net/en/company/developers.html) and extract it to `external/VST3_SDK`.

### MacOS

#### Using Homebrew

```bash
brew install cmake ninja llvm
```

### Linux

#### Ubuntu/Debian

```bash
sudo apt update
sudo apt install cmake ninja-build clang build-essential libwebkit2gtk-4.1-dev
```

#### Fedora

```bash
sudo dnf install cmake ninja-build clang
```

#### Arch Linux

```bash
sudo pacman -S cmake ninja clang
```

### Windows

#### Using Chocolatey

```powershell
choco install cmake ninja llvm windows-sdk-10-version-2104-all
choco install visualstudio2022buildtools --package-parameters "--add Microsoft.VisualStudio.Component.VC.Tools.x86.x64 --includeRecommended --includeOptional"
```

### Build Requirements From Source (Harder)

- Install CMake from [here](https://cmake.org/download/)
- Install Ninja from [here](https://ninja-build.org/)
- Install Clang from [here](https://clang.llvm.org/)

## Building

```bash
cmake -S . -B build -G Ninja
cmake --build build
```

This will build the plugin to `build/AdaptiveEcho_artefacts/`.

## Formatting

This project uses clang-format for consistent code style. To format your code:

```bash
clang-format -i src/*.cpp src/*.hpp
```

## Running

To run the standalone version of the plugin, run the binary you found in `build/AdaptiveEcho_artefacts/Standalone/`.

To run a VST3/AU version of the plugin, you can find the plugin files in `build/AdaptiveEcho_artefacts/VST3/` or `build/AdaptiveEcho_artefacts/AU/`.
