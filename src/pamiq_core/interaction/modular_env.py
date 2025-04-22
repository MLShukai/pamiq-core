from abc import ABC, abstractmethod
from pathlib import Path
from typing import override

from pamiq_core.state_persistence import PersistentStateMixin
from pamiq_core.thread import ThreadEventMixin

from .env import Environment
from .event_mixin import InteractionEventMixin


class Sensor[T](ABC, InteractionEventMixin, PersistentStateMixin, ThreadEventMixin):
    """Abstract base class for sensors that read data from the environment.

    This class provides an interface for reading observations from
    various sensors. Implementations should handle the specific logic
    for acquiring sensor readings.
    """

    @abstractmethod
    def read(self) -> T:
        """Read data from the sensor.

        Returns:
            Sensor reading/observation data.
        """
        ...


class Actuator[T](ABC, InteractionEventMixin, PersistentStateMixin, ThreadEventMixin):
    """Abstract base class for actuators that affect the environment.

    This class provides an interface for operating actuators based on
    action commands. Implementations should handle the specific logic
    for executing actions through hardware or simulation interfaces.
    """

    @abstractmethod
    def operate(self, action: T) -> None:
        """Execute the specified action through the actuator.

        Args:
            action: The action to be executed.
        """
        ...


class ModularEnvironment[ObsType, ActType](Environment[ObsType, ActType]):
    """Environment implementation that uses a Sensor and Actuator.

    This class provides a modular approach to environment implementation
    by separating the sensing (observation) and actuation components.
    """

    @override
    def __init__(self, sensor: Sensor[ObsType], actuator: Actuator[ActType]) -> None:
        """Initialize with a sensor and actuator.

        Args:
            sensor: Component to read observations from the environment.
            actuator: Component to execute actions in the environment.
        """
        self.sensor = sensor
        self.actuator = actuator

    @override
    def observe(self) -> ObsType:
        """Get observations from the environment using the sensor.

        Returns:
            Current observation from the sensor.
        """
        return self.sensor.read()

    @override
    def affect(self, action: ActType) -> None:
        """Apply an action to the environment using the actuator.

        Args:
            action: The action to apply to the environment.
        """
        self.actuator.operate(action)

    @override
    def setup(self) -> None:
        """Set up the environment by initializing sensor and actuator.

        This method is called before starting interaction with the
        environment.
        """
        self.sensor.setup()
        self.actuator.setup()

    @override
    def teardown(self) -> None:
        """Clean up the environment by finalizing sensor and actuator.

        This method is called after finishing interaction with the
        environment.
        """
        self.sensor.teardown()
        self.actuator.teardown()

    @override
    def save_state(self, path: Path) -> None:
        """Save the state of the environment to the specified path.

        Creates a directory at the given path and saves the states of
        the sensor and actuator in subdirectories.

        Args:
            path: Directory path where to save the environment state.
        """
        path.mkdir()
        self.sensor.save_state(path / "sensor")
        self.actuator.save_state(path / "actuator")

    @override
    def load_state(self, path: Path) -> None:
        """Load the state of the environment from the specified path.

        Loads the states of the sensor and actuator from subdirectories
        at the given path.

        Args:
            path: Directory path from where to load the environment state.
        """
        self.sensor.load_state(path / "sensor")
        self.actuator.load_state(path / "actuator")

    @override
    def on_paused(self) -> None:
        """The method to be called when the thread is paused."""
        super().on_paused()
        self.sensor.on_paused()
        self.actuator.on_paused()

    @override
    def on_resumed(self) -> None:
        """The method to be called when the thread is resumed."""
        super().on_resumed()
        self.sensor.on_resumed()
        self.actuator.on_resumed()
