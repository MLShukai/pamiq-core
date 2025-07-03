# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

PAMIQ Core is a framework for building AI agents with real-time adaptive learning capabilities using a parallel train-inference architecture. The project uses Python 3.12+ with modern type annotations and follows strict coding conventions.

## Common Development Commands

The project uses Make commands for development workflow:

```bash
# Core development commands
make venv          # Create Python virtual environment using uv
make format        # Run pre-commit hooks for code formatting
make test          # Run tests with pytest and coverage
make type          # Run type checking with pyright
make run           # Run entire workflow (format + test + type)

# Docker development (recommended)
make docker-build    # Build Docker images
make docker-up       # Start development container
make docker-attach   # Connect to container
make docker-down     # Stop containers

# Documentation
make docs-serve     # Serve documentation locally
make docs-build     # Build documentation
```

## Architecture Overview

PAMIQ Core implements a unique parallel architecture where control, inference, and training threads run concurrently:

1. **Thread Architecture**:

    - Control thread: Manages system coordination
    - Inference thread: Handles agent-environment interactions
    - Training thread: Updates model parameters continuously
    - Thread-safe model synchronization between inference and training

2. **Core Abstractions**:

    - `Agent` and `Environment` interfaces for interaction loop
    - `Model` containers with thread-safe parameter sharing
    - `DataBuffer` for experience collection (implementations: RandomReplacementBuffer, SequentialBuffer)
    - `Trainer` for continuous model updates
    - `LaunchConfig` for system configuration

3. **Key Design Patterns**:

    - Modular component system with clear interfaces
    - Event-driven communication via event mixins
    - State persistence for resumable operation
    - Strict separation between abstract interfaces and implementations

## Testing and Type Checking

- **Run tests**: `make test` (uses pytest with coverage)
- **Run single test**: `pytest tests/path/to/test_file.py::test_function`
- **Type checking**: `make type` (uses pyright with strict settings)

Testing conventions:

- Tests mirror source structure in `tests/` directory
- Create `Impl` dummy classes to test abstract interfaces
- Use mocks for abstract dependencies
- Each test focuses on single functionality

## Code Style Requirements

The project follows strict coding conventions (see CODING_CONVENTIONS.md):

- Python 3.12+ syntax required (use `|` for unions, built-in generics)
- No `TypeVar`/`Generic` - use 3.12+ type parameter syntax
- All public APIs must have type annotations
- Import order: future → standard → third-party → first-party
- No relative imports in src/
- Docstrings required for all public APIs

## Package Management

The project uses `uv` for dependency management. Dependencies are specified in `pyproject.toml` with exact version pinning. Optional dependencies include PyTorch support.

## Important Notes

- Primary development platform is Linux, with Windows/macOS compatibility
- Docker development is recommended for consistency
- The console tool (`pamiq-console`) allows remote control of running systems
- Built-in web API for monitoring and control
- Comprehensive documentation available via MkDocs
