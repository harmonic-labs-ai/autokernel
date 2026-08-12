#!/bin/bash


# Install uv (if you don't have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and setup
uv sync

# One-time setup: test data + baselines
uv run prepare.py

# Required when specifying hf models
uv sync --extra models

# Install claude
curl -fsSL https://claude.ai/install.sh | bash
