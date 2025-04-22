from __future__ import annotations

from importlib import metadata

from . import data, interaction, model, time, trainer, utils
from .data import DataBuffer, DataCollector, DataUser
from .interaction import (
    Agent,
    Environment,
    FixedIntervalInteraction,
    Interaction,
    IntervalAdjustor,
)
from .launcher import LaunchConfig, launch
from .model import InferenceModel, TrainingModel
from .trainer import Trainer

# pamiq_core to pamiq-core
__version__ = metadata.version(__name__.replace("_", "-"))


__all__ = [
    "data",
    "interaction",
    "model",
    "time",
    "trainer",
    "utils",
    "DataBuffer",
    "DataCollector",
    "DataUser",
    "Interaction",
    "Agent",
    "Environment",
    "FixedIntervalInteraction",
    "IntervalAdjustor",
    "TrainingModel",
    "InferenceModel",
    "Trainer",
    "LaunchConfig",
    "launch",
]
