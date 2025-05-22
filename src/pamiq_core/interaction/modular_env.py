from __future__ import annotations

from abc import ABC, abstractmethod
from collections import UserDict
from collections.abc import Mapping
from pathlib import Path
from typing import Any, override

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
        pass


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
        pass


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

    @staticmethod
    def from_dict(
        sensors: Mapping[str, Sensor[Any]], actuators: Mapping[str, Actuator[Any]]
    ) -> ModularEnvironment[Mapping[str, Any], Mapping[str, Any]]:
        """Create a modular environment from dictionaries of sensors and
        actuators.

        Args:
            sensors: A mapping of sensor names to sensor instances.
            actuators: A mapping of actuator names to actuator instances.

        Returns:
            A modular environment that uses composite sensors and actuators.

        Example:
            ```python
            env = ModularEnvironment.from_dict(
                sensors={"camera": CameraSensor(), "lidar": LidarSensor()},
                actuators={"motor": MotorActuator(), "gripper": GripperActuator()}
            )
            ```
        """
        return ModularEnvironment(SensorsDict(sensors), ActuatorsDict(actuators))


class SensorsDict(Sensor[Mapping[str, Any]], UserDict[str, Sensor[Any]]):
    """Dictionary of sensors that acts as a composite sensor.

    This class allows grouping multiple sensors together under a single
    interface. When read, it returns a mapping of sensor names to their
    readings, providing a way to collect data from multiple sensors
    simultaneously.

    All lifecycle events (setup, teardown, pause, resume) and state
    persistence operations are propagated to all contained sensors.
    """

    @override
    def read(self) -> Mapping[str, Any]:
        """Read data from all contained sensors.

        Returns:
            A mapping from sensor names to their respective readings.
        """
        return {k: v.read() for k, v in self.items()}

    @override
    def setup(self) -> None:
        """Set up all contained sensors."""
        super().setup()
        for v in self.values():
            v.setup()

    @override
    def teardown(self) -> None:
        """Clean up resources for all contained sensors."""
        super().teardown()
        for v in self.values():
            v.teardown()

    @override
    def save_state(self, path: Path) -> None:
        """Save the state of all contained sensors.

        Creates a directory at the given path and saves each sensor's state
        in a subdirectory named after its key in this dictionary.

        Args:
            path: Directory path where to save the state.
        """
        super().save_state(path)
        path.mkdir()
        for k, v in self.items():
            v.save_state(path / k)

    @override
    def load_state(self, path: Path) -> None:
        """Load the state of all contained sensors.

        Loads each sensor's state from a subdirectory named after its key
        in this dictionary.

        Args:
            path: Directory path from where to load the state.
        """
        super().load_state(path)
        for k, v in self.items():
            v.load_state(path / k)

    @override
    def on_paused(self) -> None:
        """Handle system pause event for all contained sensors."""
        super().on_paused()
        for v in self.values():
            v.on_paused()

    @override
    def on_resumed(self) -> None:
        """Handle system resume event for all contained sensors."""
        super().on_resumed()
        for v in self.values():
            v.on_resumed()


class ActuatorsDict(Actuator[Mapping[str, Any]], UserDict[str, Actuator[Any]]):
    """Dictionary of actuators that acts as a composite actuator.

    This class allows grouping multiple actuators together under a
    single interface. When operated, it distributes actions to
    individual actuators based on their keys.

    All lifecycle events (setup, teardown, pause, resume) and state
    persistence operations are propagated to all contained actuators.
    """

    @override
    def operate(self, action: Mapping[str, Any]) -> None:
        """Operate all contained actuators with their respective actions.

        Distributes the actions to the appropriate actuators based on their keys.

        Args:
            action: A mapping from actuator names to their respective actions.
                   Each action will be passed to the corresponding actuator.

        Raises:
            KeyError: If action doesn't contain a required actuator key.
        """
        for k, v in self.items():
            v.operate(action[k])

    @override
    def setup(self) -> None:
        """Set up all contained actuators."""
        super().setup()
        for v in self.values():
            v.setup()

    @override
    def teardown(self) -> None:
        """Clean up resources for all contained actuators."""
        super().teardown()
        for v in self.values():
            v.teardown()

    @override
    def save_state(self, path: Path) -> None:
        """Save the state of all contained actuators.

        Creates a directory at the given path and saves each actuator's state
        in a subdirectory named after its key in this dictionary.

        Args:
            path: Directory path where to save the state.
        """
        super().save_state(path)
        path.mkdir()
        for k, v in self.items():
            v.save_state(path / k)

    @override
    def load_state(self, path: Path) -> None:
        """Load the state of all contained actuators.

        Loads each actuator's state from a subdirectory named after its key
        in this dictionary.

        Args:
            path: Directory path from where to load the state.
        """
        super().load_state(path)
        for k, v in self.items():
            v.load_state(path / k)

    @override
    def on_paused(self) -> None:
        """Handle system pause event for all contained actuators."""
        super().on_paused()
        for v in self.values():
            v.on_paused()

    @override
    def on_resumed(self) -> None:
        """Handle system resume event for all contained actuators."""
        super().on_resumed()
        for v in self.values():
            v.on_resumed()
