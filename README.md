# Adaptive Echo

A digital synthesizer capable of learning from sounds.

## Vision

Adaptive Echo is a synthesizer that can listen to a sound and recreate it using optimization techniques. After recreating a sound, users can modify the settings like a traditional synthesizer. It is licensed under the MIT License, meaning it is free to use for any purpose.

## Project Structure

- **[cpp/](cpp)**: The core C++ synthesizer engine and hybrid evolution optimizer. This is the primary implementation of the synthesizer DSP and the learning algorithm.
- **[docs/](docs)**: Project documentation, including synthesizer settings and architectural details.

## Getting Started (C++ Engine)

The core engine is located in the `cpp` directory. It uses CMake for building and requires a C++17 compliant compiler.

### Prerequisites
- CMake 3.14+
- C++17 compiler (GCC, Clang, or MSVC)
- OpenMP (optional, for parallel optimization)

### Building
You can use the provided build script:
```bash
cd cpp
./build.sh
```

Or manually with CMake:
```bash
cd cpp
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make
```

### Running the Optimizer
```bash
./generate_sound [target_audio.wav]
```

## Documentation

See the [docs](docs/README.md) folder for more information:
- [Synthesizer Settings](docs/synthesizer-settings.md)

## License

Adaptive Echo is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Contributing

See [the contribution guidelines](CONTRIBUTING.md) for details.
