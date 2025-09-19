# Instructions for AI Agents

## Project Layout

The project is a CMake project using the JUCE framework for audio processing and audio plugin hosting.

The project is structured as follows:

```
plugin/
  src/
    PluginProcessor.cpp # The plugin processor (DSP)
    PluginProcessor.hpp
    PluginEditor.cpp # The plugin editor (GUI)
    PluginEditor.hpp
  CMakeLists.txt # The CMake build file
  AGENTS.md # Instructions for AI agents
  README.md # Instructions for humans
```

## Building

```bash
cmake -S . -B build -G Ninja
cmake --build build
```

## Formatting

When making changes to source files, ensure code is formatted according to the project's clang-format configuration. Run the following command after editing files:

```bash
clang-format -i src/*.cpp src/*.hpp
```

This will automatically format the code to match the project's style guidelines.

## Common Errors

### Errors Writing to Targets

Any permissions errors related to writing to targets are likely due to the
restricted permissions given to an AI Agent, and are not an issue with the
plugin or build system.
