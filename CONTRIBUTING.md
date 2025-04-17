# Contributing

Thank you for your interest in contributing to PAMIQ Core. This guide explains how to set up your development environment.

## 📋 Prerequisites

Please install the following tools in advance:

### Required Tools

- 🐳 **Docker (Docker Compose)**

  - Docker Desktop: <https://www.docker.com/get-started/>
  - Docker Engine (Linux only): <https://docs.docker.com/engine/install/>
  - Verification command:
    ```sh
    docker version && docker compose version
    ```

- 🔨 **make**

  - Windows: Install via [`scoop`](https://scoop.sh) or [`chocolatey`](https://chocolatey.org)
  - macOS: Pre-installed
  - Linux: Use your distribution's package manager (e.g., for Ubuntu: `sudo apt install make`)
  - Verification command:
    ```sh
    make -v
    ```

- 🌲 **git**

  - Download: <https://git-scm.com/downloads>
  - Verification command:
    ```sh
    git -v
    ```

## 🚀 Setting Up the Development Environment

1. Repository Setup

   First, fork the repository by clicking the "Fork" button:

   [![Fork Repository](https://img.shields.io/badge/Fork%20Repository-2ea44f?style=for-the-badge)](https://github.com/MLShukai/pamiq-core/fork)

   After fork, clone your repository:

   ```sh
   git clone https://github.com/your-name/pamiq-core.git
   cd pamiq-core
   ```

2. Building the Docker Environment

   ```sh
   # Build the image
   make docker-build

   # Start the container
   make docker-up

   # Connect to the container
   make docker-attach
   ```

3. Git Initial Configuration

   ```sh
   git config user.name <your GitHub username>
   git config user.email <your GitHub email>
   ```

## 💻 Development Environment Configuration

### Development with VSCode

You can develop by attaching to the container from your preferred editor (VSCode recommended).

📚 Reference: [Attach with VSCode Dev Containers extension](https://code.visualstudio.com/docs/devcontainers/attach-container)

The development container includes the following environment:

- Package manager ([**uv**](https://docs.astral.sh/uv/))
- Git for version control
- Development dependency packages

## 🔄 Development Workflow

Use the following commands for development:

```sh
# Set up Python virtual environment
make venv

# Format code and run pre-commit hooks
make format

# Run tests
make test

# Run type checking
make type

# Run the entire workflow (format, test, type)
make run
```

## ⚙️ Environment Management

### Stopping the Container

```sh
make docker-down
```

### Cleaning Up the Development Environment

```sh
make clean
```

### ⚠️ Complete Deletion (Use Caution)

```sh
# Warning: All work data will be deleted!
make docker-down-volume
```

## 🤝 Contribution Flow

1. Create a new branch for feature additions or bug fixes
2. Make your changes
3. Write tests for new features
4. Run the entire workflow before sending a PR:
   ```shell
   make run
   ```
5. Submit a Pull Request with a clear explanation of your changes

If you have questions or issues, please create an Issue in the GitHub repository.

## 🌲 Git Branch Management

We use a modified version of [Git-Flow](https://nvie.com/posts/a-successful-git-branching-model/) tailored to our project's needs.

### 📌 Core Branches

These are permanent branches in the repository that are never deleted. Direct pushing to these branches is prohibited; changes are only accepted through pull requests.

#### 🎯 main

- The primary development branch
- Always maintained in the latest state
- Integrates changes from `feature/*` and `fix/*` branches

#### 🚀 stable

- Maintains the stable version of the source code after release
- Accepts merges from `release/*` or `hotfix/*` branches with tag issuance
- After tag issuance, merges back to the `main` branch

### 🛠️ Working Branches

These are temporary branches created for feature development or bug fixes.

#### ⚡ Basic Rules

- Branch naming format:
  - With an ISSUE: `<branch type>/#<issue number>/<name>`
    - Example 1: `feature/#123/timer-module`
    - Example 2: `fix/#321/timer-accuracy`
  - Without an issue (for very small tasks): `<branch type>/YYYYMMDD/<feature name>`
    - Example 1: `docs/20240101/fix-readme-typo`
    - Example 2: `hotfix/20240302/security-check`
- Generally branch from `main` and merge back to `main`

**NOTE: `release/*` branches do not follow these rules**

#### ✨ feature

- Branches for adding new functionality

#### 🐛 fix

- Branches for fixing bugs during development

#### 🔧 refactor

- Branches for internal changes that don't affect published functionality

#### 📚 docs

- Branches for adding or updating documentation

#### 📦 release

- Branches for release preparation
- **Branch naming format: `release/<version>`**
  - Example: `release/1.2` (without patch version)
- Branch from `main` and merge to `stable`
- Only for minor release-related fixes
- Version number updates:
  - New features: Minor version update (1.1 → 1.2)
  - Breaking changes: Major version update (1.1 → 2.0)

#### 🚑 hotfix

- Branches for urgent bug fixes
- **Branch from `stable` and merge to `stable`**
- After fixes, update patch version (1.1.0 → 1.1.1)

### 🔄 Branch Flow

```mermaid
gitGraph
    commit
    branch stable order: 2
    checkout main
    commit
    branch "feature/#123/time-module" order: 4
    commit
    commit
    checkout main
    merge "feature/#123/time-module"
    commit
    branch "fix/20230121/set-limit" order: 3
    commit
    checkout main
    merge "fix/20230121/set-limit"
    commit
    branch "release/1.0" order: 2
    checkout "release/1.0"
    commit
    checkout stable
    merge "release/1.0" tag: "1.0.0"
    checkout main
    merge stable
    checkout stable
    branch "hotfix/20230302/value-error" order: 1
    commit
    checkout stable
    merge "hotfix/20230302/value-error" tag: "1.0.1"
    checkout main
    merge stable
```
