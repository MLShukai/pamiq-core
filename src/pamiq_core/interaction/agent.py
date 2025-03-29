from abc import ABC, abstractmethod
from typing import Any

from pamiq_core.data import DataCollector, DataCollectorsDict
from pamiq_core.model import InferenceModel, InferenceModelsDict
from pamiq_core.state_persistence import PersistentStateMixin
from pamiq_core.threads import ThreadEventMixin

from .event_mixin import InteractionEventMixin


class Agent[ObsType, ActType](
    ABC, InteractionEventMixin, PersistentStateMixin, ThreadEventMixin
):
    """Base agent class for decision making.

    An agent receives observations from an environment and decides on
    actions to take in response. This abstract class defines the
    interface that all agent implementations must follow.
    """

    _inference_models: InferenceModelsDict
    _data_collectors: DataCollectorsDict

    @abstractmethod
    def step(self, observation: ObsType) -> ActType:
        """Processes an observation and determines the next action.

        Args:
            observation: The current observation from the environment.

        Returns:
            The action to take in response to the observation.
        """
        ...

    def attach_inference_models(self, inference_models: InferenceModelsDict) -> None:
        """Attaches inference models dictionary to this agent.

        This method is called to provide the agent with access to inference models
        for decision making. After attaching, the callback method
        `on_inference_models_attached` is called.

        Args:
            inference_models: Dictionary of inference models to attach.
        """
        self._inference_models = inference_models
        self.on_inference_models_attached()

    def on_inference_models_attached(self) -> None:
        """Callback method for when inference models are attached to the agent.

        Override this method to retrieve DNN models for inference. Use
        `get_inference_model` to retrieve a model.
        """
        pass

    def attach_data_collectors(self, data_collectors: DataCollectorsDict) -> None:
        """Attaches data collectors dictionary to this agent.

        This method is called to provide the agent with access to data collectors
        for collecting experience data. After attaching, the callback method
        `on_data_collectors_attached` is called.

        Args:
            data_collectors: Dictionary of data collectors to attach.
        """
        self._data_collectors = data_collectors
        self.on_data_collectors_attached()

    def on_data_collectors_attached(self) -> None:
        """Callback method for when data collectors are attached to this agent.

        Override this method to set up data collection for the agent. Use
        `get_data_collector` to acquire a collector.
        """
        pass

    def get_inference_model(self, name: str) -> InferenceModel:
        """Retrieves an inference model by name.

        Args:
            name: Name of the inference model to retrieve.

        Returns:
            The requested inference model.

        Raises:
            KeyError: If the model with the specified name does not exist.
        """
        return self._inference_models[name]

    def get_data_collector(self, name: str) -> DataCollector[Any]:
        """Acquires a data collector by name for exclusive use.

        This method acquires a data collector for exclusive use within
        the current step. The collector can only be acquired once at a time.

        Args:
            name: Name of the data collector to acquire.

        Returns:
            The requested data collector.

        Raises:
            KeyError: If the collector is already acquired or not found.
        """
        return self._data_collectors.acquire(name)
