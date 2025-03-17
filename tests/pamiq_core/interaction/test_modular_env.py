from abc import ABC
from pathlib import Path

import pytest

from pamiq_core.interaction.environment import Environment
from pamiq_core.interaction.event_mixin import InteractionEventMixin
from pamiq_core.interaction.modular_env import Actuator, ModularEnv, Sensor
from pamiq_core.state_persistence import PersistentStateMixin


class TestSensor:
    """Test suite for Sensor abstract base class."""

    def test_sensor_inheritance(self):
        """Test that Sensor inherits from correct base classes."""
        assert issubclass(Sensor, ABC)
        assert issubclass(Sensor, InteractionEventMixin)
        assert issubclass(Sensor, PersistentStateMixin)

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

    def test_abstract_methods(self):
        """Test that Actuator has the correct abstract methods."""
        assert Actuator.__abstractmethods__ == frozenset({"operate"})


class TestModularEnv:
    """Test suite for ModularEnv class."""

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
        """Fixture providing a ModularEnv with mock components."""
        return ModularEnv(mock_sensor, mock_actuator)

    def test_inheritance(self):
        """Test that ModularEnv inherits from Environment."""
        assert issubclass(ModularEnv, Environment)

    def test_init(self, env: ModularEnv, mock_sensor, mock_actuator):
        """Test ModularEnv initialization."""
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
