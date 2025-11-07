#!/usr/bin/env bash

uv run ruff format .
uv run ruff check --select I,RUF022 --fix .
