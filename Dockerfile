FROM ubuntu:latest

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
ENV UV_LINK_MODE=copy
ENV PATH="/root/.local/bin/:$PATH"

# Setup workspace
RUN mkdir /workspace
WORKDIR /workspace
COPY . .

# Install dependencies
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    curl \
    git \
    make \
    libevdev-dev \
    build-essential \
    bash-completion \
    # Install Nodejs
    && curl -fsSL https://deb.nodesource.com/setup_22.x | bash \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/* \
    # Setup Bash Completion
    && echo '[[ $PS1 && -f /usr/share/bash-completion/bash_completion ]] && \
    . /usr/share/bash-completion/bash_completion' >> ~/.bashrc \
    && make venv \
    # Install pre-commit hook.
    && uv run pre-commit install \
    # Shell completion
    && echo 'eval "$(uv generate-shell-completion bash)"' >> ~/.bashrc


# Setup Claude Code
RUN npm install -g @anthropic-ai/claude-code

# Default command (can be overridden)
CMD ["/bin/bash"]
