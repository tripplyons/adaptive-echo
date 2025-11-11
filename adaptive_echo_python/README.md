# Adaptive Echo Python Definitions

## Setup

Install uv:

- [uv installation methods](https://docs.astral.sh/uv/getting-started/installation/#installation-methods)

Install dependencies:

```bash
uv sync
```

## Generating Graphs

This command will export the synthesizer and models to TorchScript format and save them within the `plugin` folder for later use in the C++ audio plugin.

```bash
uv run python src/adaptive_echo_python/export_graphs.py
```

## Formatting

```bash
./format.sh
```

## Run Tests

```bash
./test.sh
```
