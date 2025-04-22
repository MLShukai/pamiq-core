from abc import ABC
from pathlib import Path

import pytest

from pamiq_core.interaction.env import Environment
from pamiq_core.interaction.event_mixin import InteractionEventMixin
from pamiq_core.interaction.modular_env import Actuator, ModularEnvironment, Sensor
from pamiq_core.state_persistence import PersistentStateMixin
from pamiq_core.thread import ThreadEventMixin


class TestSensor:
    """Test suite for Sensor abstract base class."""

    def test_sensor_inheritance(self):
        """Test that Sensor inherits from correct base classes."""
        assert issubclass(Sensor, ABC)
        assert issubclass(Sensor, InteractionEventMixin)
        assert issubclass(Sensor, PersistentStateMixin)
        assert issubclass(Sensor, ThreadEventMixin)

    def test_abstract_methods(self):
        """Test that Sensor has the correct abstract methods."""
        assert Sensor.__abstractmethods__ == frozenset({"read"})


class TestActuator:
    """Test suite for Actuator abstract base class."""

    def test_actuator_inheritance(self):
        """Test that Actuator inherits from correct base classes."""
        assert issubclass(Actuator, ABC)
        assert issubclass(Actuator, InteractionEventMixin)
        assert issubclass(Actuator, PersistentStateMixin)
        assert issubclass(Actuator, ThreadEventMixin)

    def test_abstract_methods(self):
        """Test that Actuator has the correct abstract methods."""
        assert Actuator.__abstractmethods__ == frozenset({"operate"})


class TestModularEnvironment:
    """Test suite for ModularEnvironment class."""

    @pytest.fixture
    def mock_sensor(self, mocker):
        """Fixture providing a mock sensor."""
        sensor = mocker.Mock(spec=Sensor)
        sensor.read.return_value = "test_observation"
        return sensor

    @pytest.fixture
    def mock_actuator(self, mocker):
        """Fixture providing a mock actuator."""
        return mocker.Mock(spec=Actuator)

    @pytest.fixture
    def env(self, mock_sensor, mock_actuator):
        """Fixture providing a ModularEnvironment with mock components."""
        return ModularEnvironment(mock_sensor, mock_actuator)

    def test_inheritance(self):
        """Test that ModularEnvironment inherits from Environment."""
        assert issubclass(ModularEnvironment, Environment)

    def test_init(self, env: ModularEnvironment, mock_sensor, mock_actuator):
        """Test ModularEnvironment initialization."""
        assert env.sensor == mock_sensor
        assert env.actuator == mock_actuator

    def test_observe(self, env, mock_sensor):
        """Test that observe calls sensor.read()."""
        mock_sensor.read.return_value = "mocked_observation"
        result = env.observe()

        mock_sensor.read.assert_called_once()
        assert result == "mocked_observation"

    def test_affect(self, env, mock_actuator):
        """Test that affect calls actuator.operate()."""
        action = "test_action"
        env.affect(action)

        mock_actuator.operate.assert_called_once_with(action)

    def test_setup(self, env, mock_sensor, mock_actuator):
        """Test that setup calls setup on both sensor and actuator."""
        env.setup()

        mock_sensor.setup.assert_called_once()
        mock_actuator.setup.assert_called_once()

    def test_teardown(self, env, mock_sensor, mock_actuator):
        """Test that teardown calls teardown on both sensor and actuator."""
        env.teardown()

        mock_sensor.teardown.assert_called_once()
        mock_actuator.teardown.assert_called_once()

    def test_save_state(self, env, mock_sensor, mock_actuator, tmp_path: Path):
        """Test that save_state creates directory and calls save_state on
        components."""
        save_path = tmp_path / "test_save"
        env.save_state(save_path)

        assert save_path.is_dir()

        mock_sensor.save_state.assert_called_once_with(save_path / "sensor")
        mock_actuator.save_state.assert_called_once_with(save_path / "actuator")

    def test_load_state(self, env, mock_sensor, mock_actuator, tmp_path: Path):
        """Test that load_state calls load_state on both sensor and
        actuator."""
        load_path = tmp_path / "test_load"

        env.load_state(load_path)

        mock_sensor.load_state.assert_called_once_with(load_path / "sensor")
        mock_actuator.load_state.assert_called_once_with(load_path / "actuator")

    def test_on_paused(self, env, mock_sensor, mock_actuator):
        """Test that on_paused() calls on_paused on both sensor and
        actuator."""
        env.on_paused()
        mock_sensor.on_paused.assert_called_once_with()
        mock_actuator.on_paused.assert_called_once_with()

    def test_on_resumed(self, env, mock_sensor, mock_actuator):
        """Test that on_resumed() calls on_resumed on both sensor and
        actuator."""
        env.on_resumed()
        mock_sensor.on_resumed.assert_called_once_with()
        mock_actuator.on_resumed.assert_called_once_with()
