from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import cast, override

from pamiq_core.interaction.env import Environment
from pamiq_core.interaction.event_mixin import InteractionEventMixin
from pamiq_core.interaction.modular_env import Actuator, Sensor
from pamiq_core.state_persistence import PersistentStateMixin
from pamiq_core.thread import ThreadEventMixin


class Wrapper[T, W](ABC, InteractionEventMixin, PersistentStateMixin, ThreadEventMixin):
    """Base wrapper class for transforming values.

    This abstract class provides an interface for wrapping and
    transforming values from one type to another. Subclasses should
    implement the wrap method to define the specific transformation.
    """

    @abstractmethod
    def wrap(self, value: T) -> W:
        """Transform the input value to another value.

        Args:
            value: Input value to be transformed.

        Returns:
            Transformed value.
        """
        pass

    def __call__(self, value: T) -> W:
        """Enable calling the wrapper as a function.

        Args:
            value: Input value to be transformed.

        Returns:
            Transformed value.
        """
        return self.wrap(value)

    def wrap_sensor(self, sensor: Sensor[T]) -> SensorWrapper[T, W]:
        """Create a SensorWrapper that combines this wrapper with the provided
        sensor.

        This is a convenience method that applies the current wrapper to a sensor,
        creating a SensorWrapper that will transform the sensor's readings using
        this wrapper's transformation logic.

        Args:
            sensor: The sensor to wrap with this wrapper.

        Returns:
            A new SensorWrapper that applies this wrapper's transformation to the sensor.
        """
        return SensorWrapper(sensor, self)

    def wrap_actuator(self, actuator: Actuator[W]) -> ActuatorWrapper[W, T]:
        """Create an ActuatorWrapper that combines this wrapper with the
        provided actuator.

        This is a convenience method that applies the current wrapper to an actuator,
        creating an ActuatorWrapper that will transform actions before passing them
        to the actuator using this wrapper's transformation logic.

        Args:
            actuator: The actuator to wrap with this wrapper.

        Returns:
            A new ActuatorWrapper that applies this wrapper's transformation to actions.
        """
        return ActuatorWrapper(actuator, self)


class LambdaWrapper[T, W](Wrapper[T, W]):
    """Wrapper that uses a callable function to transform values.

    This wrapper enables using lambda functions or any callable to
    perform the transformation.
    """

    def __init__(self, func: Callable[[T], W]) -> None:
        """Initialize with a transformation function.

        Args:
            func: Function that transforms input values of type T to type W.
        """
        super().__init__()
        self._func = func

    @override
    def wrap(self, value: T) -> W:
        """Transform the value using the provided function.

        Args:
            value: Input value to be transformed.

        Returns:
            Transformed value.
        """
        return self._func(value)


def _ensure_wrapper[T, W](wrapper: Wrapper[T, W] | Callable[[T], W]) -> Wrapper[T, W]:
    """Ensure the given object is a Wrapper instance.

    Args:
        wrapper: A wrapper instance or function.

    Returns:
        A Wrapper instance.
    """
    if isinstance(wrapper, Wrapper):
        return cast(Wrapper[T, W], wrapper)  # Avoid unknown member type error.
    return LambdaWrapper(wrapper)


class EnvironmentWrapper[ObsType, WrappedObsType, ActType, WrappedActType](
    Environment[WrappedObsType, WrappedActType]
):
    """Wrapper for Environment that transforms observations and actions.

    This wrapper transforms the observations returned by the wrapped
    environment and the actions passed to it.
    """

    def __init__(
        self,
        env: Environment[ObsType, ActType],
        obs_wrapper: Wrapper[ObsType, WrappedObsType]
        | Callable[[ObsType], WrappedObsType],
        act_wrapper: Wrapper[WrappedActType, ActType]
        | Callable[[WrappedActType], ActType],
    ) -> None:
        """Initialize with an environment and wrappers.

        Args:
            env: The environment to wrap.
            obs_wrapper: Wrapper for transforming observations from the environment.
            act_wrapper: Wrapper for transforming actions before passing to the environment.
        """
        self.env = env
        self._obs_wrapper = _ensure_wrapper(obs_wrapper)
        self._act_wrapper = _ensure_wrapper(act_wrapper)

    @override
    def observe(self) -> WrappedObsType:
        """Get wrapped observation from the environment.

        Returns:
            Transformed observation.
        """
        return self._obs_wrapper(self.env.observe())

    @override
    def affect(self, action: WrappedActType) -> None:
        """Apply wrapped action to the environment.

        Args:
            action: Action to be transformed and applied.
        """
        self.env.affect(self._act_wrapper(action))

    @override
    def setup(self) -> None:
        """Set up the wrapped environment and wrappers."""
        self.env.setup()
        self._obs_wrapper.setup()
        self._act_wrapper.setup()

    @override
    def teardown(self) -> None:
        """Clean up the wrapped environment and wrappers."""
        self.env.teardown()
        self._obs_wrapper.teardown()
        self._act_wrapper.teardown()

    @override
    def save_state(self, path: Path) -> None:
        """Save state of the wrapped environment and wrappers.

        Args:
            path: Directory path where to save the state.
        """
        path.mkdir()
        self.env.save_state(path / "env")
        self._obs_wrapper.save_state(path / "obs_wrapper")
        self._act_wrapper.save_state(path / "act_wrapper")

    @override
    def load_state(self, path: Path) -> None:
        """Load state of the wrapped environment and wrappers.

        Args:
            path: Directory path from where to load the state.
        """
        self.env.load_state(path / "env")
        self._obs_wrapper.load_state(path / "obs_wrapper")
        self._act_wrapper.load_state(path / "act_wrapper")

    @override
    def on_paused(self) -> None:
        """The method to be called when the thread is paused."""
        super().on_paused()
        self.env.on_paused()
        self._obs_wrapper.on_paused()
        self._act_wrapper.on_paused()

    @override
    def on_resumed(self) -> None:
        """The method to be called when the thread is on_resumed."""
        super().on_resumed()
        self.env.on_resumed()
        self._obs_wrapper.on_resumed()
        self._act_wrapper.on_resumed()


class SensorWrapper[T, W](Sensor[W]):
    """Wrapper for Sensor that transforms sensor readings.

    This wrapper applies a transformation to the readings from the
    wrapped sensor.
    """

    def __init__(
        self, sensor: Sensor[T], wrapper: Wrapper[T, W] | Callable[[T], W]
    ) -> None:
        """Initialize with a sensor and a wrapper.

        Args:
            sensor: The sensor to wrap.
            wrapper: Wrapper for transforming sensor readings.
        """
        self.sensor = sensor
        self._wrapper = _ensure_wrapper(wrapper)

    @override
    def read(self) -> W:
        """Get transformed reading from the sensor.

        Returns:
            Transformed sensor reading.
        """
        return self._wrapper(self.sensor.read())

    @override
    def setup(self) -> None:
        """Set up the wrapped sensor and wrapper."""
        self.sensor.setup()
        self._wrapper.setup()

    @override
    def teardown(self) -> None:
        """Clean up the wrapped sensor and wrapper."""
        self.sensor.teardown()
        self._wrapper.teardown()

    @override
    def save_state(self, path: Path) -> None:
        """Save state of the wrapped sensor and wrapper.

        Args:
            path: Directory path where to save the state.
        """
        path.mkdir()
        self.sensor.save_state(path / "sensor")
        self._wrapper.save_state(path / "wrapper")

    @override
    def load_state(self, path: Path) -> None:
        """Load state of the wrapped sensor and wrapper.

        Args:
            path: Directory path from where to load the state.
        """
        self.sensor.load_state(path / "sensor")
        self._wrapper.load_state(path / "wrapper")

    @override
    def on_paused(self) -> None:
        """The method to be called when the thread is paused."""
        super().on_paused()
        self.sensor.on_paused()
        self._wrapper.on_paused()

    @override
    def on_resumed(self) -> None:
        """The method to be called when the thread is resumed."""
        super().on_resumed()
        self.sensor.on_resumed()
        self._wrapper.on_resumed()


class ActuatorWrapper[T, W](Actuator[W]):
    """Wrapper for Actuator that transforms actions.

    This wrapper applies a transformation to actions before passing them
    to the wrapped actuator.
    """

    def __init__(
        self, actuator: Actuator[T], wrapper: Wrapper[W, T] | Callable[[W], T]
    ) -> None:
        """Initialize with an actuator and a wrapper.

        Args:
            actuator: The actuator to wrap.
            wrapper: Wrapper for transforming actions.
        """
        self.actuator = actuator
        self._wrapper = _ensure_wrapper(wrapper)

    @override
    def operate(self, action: W) -> None:
        """Apply transformed action to the actuator.

        Args:
            action: Action to be transformed and applied.
        """
        self.actuator.operate(self._wrapper(action))

    @override
    def setup(self) -> None:
        """Set up the wrapped actuator and wrapper."""
        self.actuator.setup()
        self._wrapper.setup()

    @override
    def teardown(self) -> None:
        """Clean up the wrapped actuator and wrapper."""
        self.actuator.teardown()
        self._wrapper.teardown()

    @override
    def save_state(self, path: Path) -> None:
        """Save state of the wrapped actuator and wrapper.

        Args:
            path: Directory path where to save the state.
        """
        path.mkdir()
        self.actuator.save_state(path / "actuator")
        self._wrapper.save_state(path / "wrapper")

    @override
    def load_state(self, path: Path) -> None:
        """Load state of the wrapped actuator and wrapper.

        Args:
            path: Directory path from where to load the state.
        """
        self.actuator.load_state(path / "actuator")
        self._wrapper.load_state(path / "wrapper")

    @override
    def on_paused(self) -> None:
        """The method to be called when the thread is paused."""
        super().on_paused()
        self.actuator.on_paused()
        self._wrapper.on_paused()

    @override
    def on_resumed(self) -> None:
        """The method to be called when the thread is resumed."""
        super().on_resumed()
        self.actuator.on_resumed()
        self._wrapper.on_resumed()
