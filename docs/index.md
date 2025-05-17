![logo](./images/logo.svg)

# Welcome to PAMIQ-Core Documentation!

**PAMIQ-Core** is a framework for building AI agents with real-time learning capabilities through parallel training and inference.

## ✨ Features

- 🔄 **Parallel Architecture**: Simultaneous inference and training in separate threads
- ⚡ **Real-time Adaptation**: Continuously update models during interaction
- 🧵 **Thread-safe Design**: Robust synchronization mechanisms for parameter sharing and data transfers
- 🔌 **Modular Components**: Easy-to-extend agent, environment, and model interfaces
- 🛠️ **Comprehensive Tools**: Built-in state persistence, time control, and monitoring
- 🌍 **Cross Platform**: Linux is the primary focus, but Windows and macOS are also supported. (However, some older macOS and Windows systems may have significantly less accurate time control.)

![System Architecture](images/system-architecture.png)

## Installation

```bash
# Basic installation
pip install pamiq-core

# With PyTorch support
pip install pamiq-core[torch]
```

## Basic Example

```python
from pamiq_core import launch, Interaction, LaunchConfig
from your_agent import YourAgent
from your_environment import YourEnvironment

# Create agent-environment interaction
interaction = Interaction(YourAgent(), YourEnvironment())

# Launch the system
launch(
    interaction=interaction,
    models=your_models,
    data=your_data_buffers,
    trainers=your_trainers,
    config=LaunchConfig(
        web_api_address=("localhost", 8391),
        max_uptime=300.0,  # 5 minutes
    ),
)
```

See the [samples on GitHub](https://github.com/MLShukai/pamiq-core/tree/main/samples) for complete examples.

### Remote CLI Control

Once the system is running, you can connect and control it remotely via the terminal using `pamiq-console`:

```bash
# Connect to local system
pamiq-console --host localhost --port 8391

# Connect to remote system
pamiq-console --host 192.168.1.100 --port 8391
```

## Documentation

- [**User Guide**](./user-guide/index.md): Describes information for working with PAMIQ-Core.
- [**API Reference**](./api/launch.md)

## Contribution

See [**CONTRIBUTING.md**](https://github.com/MLShukai/pamiq-core/blob/main/CONTRIBUTING.md) on GitHub
