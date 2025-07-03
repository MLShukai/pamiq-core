# 🏋️ Gymnasium Integration

This guide explains how to use Gymnasium environments with PAMIQ Core framework.

## Installation

To use the Gymnasium integration, install PAMIQ Core with the gym extra:

```bash
pip install pamiq-core[gym]
```

## Overview

The `pamiq_core.gym` module provides seamless integration between [Gymnasium](https://gymnasium.farama.org/) (the fork of OpenAI Gym) and PAMIQ Core's interaction system. This allows you to use any Gymnasium environment as a PAMIQ Core environment.

## Key Components

### GymEnvironment

`GymEnvironment` wraps any Gymnasium environment to make it compatible with PAMIQ Core's `Environment` interface:

```python
from pamiq_core.gym import GymEnvironment

# Create from environment ID
env = GymEnvironment("CartPole-v1")

# Or use existing Gymnasium environment
import gymnasium as gym
gym_env = gym.make("CartPole-v1", render_mode="human")
env = GymEnvironment(gym_env)
```

### GymAgent

`GymAgent` is an abstract base class for implementing agents that interact with Gymnasium environments:

```python
from pamiq_core.gym import GymAgent
import numpy as np

class MyCartPoleAgent(GymAgent[np.ndarray, int]):
    def on_reset(self, obs, info):
        """Called when environment resets."""
        return 0  # Initial action

    def on_step(self, obs, reward, terminated, truncated, info):
        """Called at each environment step."""
        # Simple policy: move right if pole is tilting right
        return 1 if obs[2] > 0 else 0
```

## Advanced Features

### Manual Environment Reset

Agents can request environment reset by setting the `need_reset` flag:

```python
class EarlyStoppingAgent(GymAgent[np.ndarray, int]):
    def on_step(self, obs, reward, terminated, truncated, info):
        # Request reset if pole angle is too large
        if abs(obs[2]) > 0.5:
            self.need_reset = True
        return 1 if obs[2] > 0 else 0
```

______________________________________________________________________

## API Reference

More details, Checkout to the [API Reference](../api/gym.md)
