# Adaptive Echo

Adaptive Echo is a JUCE-based audio generator plugin that learns a sampled sound with the existing CMA-ES optimizer and replays it as a tuned one-shot synthesizer.

## Building

The project builds from the repository root and fetches JUCE automatically through CMake.

```bash
./build.sh
```

Manual build:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --parallel
```

The build produces `VST3` and `Standalone` targets. `COPY_PLUGIN_AFTER_BUILD` is enabled, so JUCE copies the plugin into the default install location after a successful build.

## Plugin Interface

- Load a training sample from disk
- Choose the reference frequency used to tune the learned sound
- Train the synthesizer with the existing optimizer logic
- Play notes from the JUCE on-screen keyboard or host MIDI

Each note renders as a one-shot voice that ignores note-off and continues until the learned envelopes finish.

## Project Structure

- **[cpp/](cpp)**: Core DSP, optimizer, and shared engine helpers.
- **[plugin/](plugin)**: JUCE plugin processor/editor sources.
- **[docs/](docs)**: Project documentation.

## Documentation

See [docs/README.md](docs/README.md) for additional details, including [synthesizer settings](docs/synthesizer-settings.md).

## License

Adaptive Echo is licensed under the MIT License. See [LICENSE](LICENSE) for details.
