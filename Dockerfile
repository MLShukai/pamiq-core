FROM python:3.12-slim

# Setup workspace
RUN mkdir /workspace
WORKDIR /workspace
COPY ./ /workspace/

# Install dependencies
ENV POETRY_VIRTUALENVS_CREATE=false
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    curl \
    git \
    make \
    && rm -rf /var/lib/apt/lists/* \
    && pip install poetry \
    && poetry install

# Default command (can be overridden)
CMD ["/bin/bash"]
