# Adaptive Echo

A fully differentiable synthesizer capable of learning from sounds

## Vision

Adaptive Echo is a synthesizer audio plugin that can listen to a sound and recreate it using gradient descent and machine learning models. After recreating a sound, users can modify the settings like a traditional synthesizer. It is licensed under the MIT License, meaning it is free to use for any purpose.

## C++ Audio Plugin

This is the main implementation of all features. It is a instrument plugin written in C++ using the JUCE framework.

See [Plugin README](plugin/README.md) for details.

## Python Prototype

A prototype implementation of the synthesizer in Python. It uses JAX for differentiation and compilation and JAX's Numpy API.

See [Python README](python/README.md) for details.

## License

Adaptive Echo is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Contributing

See [the contribution guidelines](CONTRIBUTING.md) for details.
