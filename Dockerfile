FROM ubuntu:24.04

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
ENV UV_LINK_MODE=copy
ENV PATH="/root/.local/bin/:$PATH"

# Setup workspace
RUN mkdir /workspace
WORKDIR /workspace
COPY ./ /workspace/

# Install dependencies
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    curl \
    git \
    make \
    && rm -rf /var/lib/apt/lists/* \
    && make venv \
    # Install pre-commit hook.
    && uv run pre-commit install \
    # Shell completion
    && echo 'eval "$(uv generate-shell-completion bash)"' >> ~/.bashrc

# Default command (can be overridden)
CMD ["/bin/bash"]
