from collections.abc import Mapping
from typing import Any, NamedTuple

from .data import DataBuffer, DataCollectorsDict, DataUsersDict
from .interaction import Agent
from .model import InferenceModelsDict, TrainingModel, TrainingModelsDict
from .trainer import Trainer, TrainersDict


class ConnectedComponents(NamedTuple):
    """Container for connected PAMIQ components.

    Attributes:
        data_users: Dictionary of data users created from buffers
        data_collectors: Dictionary of data collectors associated with data users
        training_models: Dictionary of training models
        inference_models: Dictionary of inference models derived from training models
    """

    data_users: DataUsersDict
    data_collectors: DataCollectorsDict
    training_models: TrainingModelsDict
    inference_models: InferenceModelsDict


def connect_components(
    agent: Agent[Any, Any] | None = None,
    trainers: Trainer | Mapping[str, Trainer] | None = None,
    buffers: Mapping[str, DataBuffer[Any]] | None = None,
    models: Mapping[str, TrainingModel[Any]] | None = None,
) -> ConnectedComponents:
    """Connect PAMIQ Core components for testing or development.

    This function wires together the core components (agent, trainers, buffers, models)
    by establishing the appropriate connection relationships between them. It handles
    the creation of data users from buffers, extraction of inference models from
    training models, and proper attachment of all related components.

    Args:
        agent: Optional agent to connect with models and data collectors
        trainers: Optional trainer or mapping of trainers to connect with models and data
        buffers: Optional mapping of data buffers to create data users from
        models: Optional mapping of training models to connect with trainers and agent

    Returns:
        ConnectedComponents containing the connected component dictionaries
    """
    if buffers is None:
        buffers = {}
    if models is None:
        models = {}

    if isinstance(trainers, Trainer):
        trainers = {"trainer": trainers}

    data_users = DataUsersDict.from_data_buffers(buffers)
    data_collectors = data_users.data_collectors_dict

    training_models = TrainingModelsDict(models)
    inference_models = training_models.inference_models_dict

    if agent is not None:
        agent.attach_data_collectors(data_collectors)
        agent.attach_inference_models(inference_models)

    if trainers is not None:
        trainers_dict = TrainersDict(trainers)
        trainers_dict.attach_data_users(data_users)
        trainers_dict.attach_training_models(training_models)

    return ConnectedComponents(
        data_users=data_users,
        data_collectors=data_collectors,
        training_models=training_models,
        inference_models=inference_models,
    )
