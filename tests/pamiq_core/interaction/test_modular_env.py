from abc import ABC
from collections.abc import Mapping
from pathlib import Path
from typing import override

import pytest
from pytest_mock import MockerFixture

from pamiq_core.interaction.env import Environment
from pamiq_core.interaction.event_mixin import InteractionEventMixin
from pamiq_core.interaction.modular_env import (
    Actuator,
    ActuatorsDict,
    ModularEnvironment,
    Sensor,
    SensorsDict,
)
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

    def test_from_dict(self, mock_sensor, mock_actuator):
        env = ModularEnvironment.from_dict(
            {"sensor": mock_sensor}, {"actuator": mock_actuator}
        )

        assert env.observe() == {"sensor": "test_observation"}
        env.affect({"actuator": "action"})
        mock_actuator.operate.assert_called_once_with("action")


class DummySensor(Sensor[int]):
    """Simple sensor implementation for testing."""

    def __init__(self, value: int = 0):
        self.value = value

    @override
    def read(self) -> int:
        return self.value


class DummyActuator(Actuator[int]):
    """Simple actuator implementation for testing."""

    def __init__(self):
        self.last_action = None

    @override
    def operate(self, action: int) -> None:
        self.last_action = action


class TestSensorsDict:
    """Test suite for SensorsDict."""

    @pytest.fixture
    def sensors_dict(self) -> SensorsDict:
        """Fixture providing a SensorsDict with dummy sensors."""
        sensors = SensorsDict()
        sensors["sensor1"] = DummySensor(1)
        sensors["sensor2"] = DummySensor(2)
        return sensors

    def test_read(self, sensors_dict: SensorsDict) -> None:
        """Test that read() returns readings from all sensors."""
        readings = sensors_dict.read()

        assert isinstance(readings, Mapping)
        assert readings["sensor1"] == 1
        assert readings["sensor2"] == 2

    def test_setitem_getitem(self) -> None:
        """Test dictionary-like operations."""
        sensors = SensorsDict()
        sensor = DummySensor(5)

        sensors["new_sensor"] = sensor

        assert "new_sensor" in sensors
        assert sensors["new_sensor"] is sensor

    def test_setup_propagates(
        self, sensors_dict: SensorsDict, mocker: MockerFixture
    ) -> None:
        """Test that setup() propagates to all contained sensors."""
        # Spy on the contained sensors' setup methods
        spy1 = mocker.spy(sensors_dict["sensor1"], "setup")
        spy2 = mocker.spy(sensors_dict["sensor2"], "setup")

        sensors_dict.setup()

        spy1.assert_called_once_with()
        spy2.assert_called_once_with()

    def test_teardown_propagates(
        self, sensors_dict: SensorsDict, mocker: MockerFixture
    ) -> None:
        """Test that teardown() propagates to all contained sensors."""
        # Spy on the contained sensors' teardown methods
        spy1 = mocker.spy(sensors_dict["sensor1"], "teardown")
        spy2 = mocker.spy(sensors_dict["sensor2"], "teardown")

        sensors_dict.teardown()

        spy1.assert_called_once_with()
        spy2.assert_called_once_with()

    def test_on_paused_propagates(
        self, sensors_dict: SensorsDict, mocker: MockerFixture
    ) -> None:
        """Test that on_paused() propagates to all contained sensors."""
        # Spy on the contained sensors' on_paused methods
        spy1 = mocker.spy(sensors_dict["sensor1"], "on_paused")
        spy2 = mocker.spy(sensors_dict["sensor2"], "on_paused")

        sensors_dict.on_paused()

        spy1.assert_called_once_with()
        spy2.assert_called_once_with()

    def test_on_resumed_propagates(
        self, sensors_dict: SensorsDict, mocker: MockerFixture
    ) -> None:
        """Test that on_resumed() propagates to all contained sensors."""
        # Spy on the contained sensors' on_resumed methods
        spy1 = mocker.spy(sensors_dict["sensor1"], "on_resumed")
        spy2 = mocker.spy(sensors_dict["sensor2"], "on_resumed")

        sensors_dict.on_resumed()

        spy1.assert_called_once_with()
        spy2.assert_called_once_with()

    def test_save_load_state(
        self, sensors_dict: SensorsDict, mocker: MockerFixture, tmp_path: Path
    ) -> None:
        """Test saving and loading state."""
        # Spy on the contained sensors' save_state methods
        spy_save1 = mocker.spy(sensors_dict["sensor1"], "save_state")
        spy_save2 = mocker.spy(sensors_dict["sensor2"], "save_state")

        # Spy on the contained sensors' load_state methods
        spy_load1 = mocker.spy(sensors_dict["sensor1"], "load_state")
        spy_load2 = mocker.spy(sensors_dict["sensor2"], "load_state")

        # Create a path for saving/loading
        save_path = tmp_path / "sensors_state"

        # Save state
        sensors_dict.save_state(save_path)

        # Check that save_state was called on all sensors with correct paths
        # Check that save_state was called on all actuators with correct paths
        spy_save1.assert_called_once_with(save_path / "sensor1")
        spy_save2.assert_called_once_with(save_path / "sensor2")

        # Load state
        sensors_dict.load_state(save_path)

        # Check that load_state was called on all actuators with correct paths
        spy_load1.assert_called_once_with(save_path / "sensor1")
        spy_load2.assert_called_once_with(save_path / "sensor2")


class TestActuatorsDict:
    """Test suite for ActuatorsDict."""

    @pytest.fixture
    def actuators_dict(self) -> ActuatorsDict:
        """Fixture providing an ActuatorsDict with dummy actuators."""
        actuators = ActuatorsDict()
        actuators["actuator1"] = DummyActuator()
        actuators["actuator2"] = DummyActuator()
        return actuators

    def test_operate(self, actuators_dict) -> None:
        """Test that operate() distributes actions to all actuators."""
        actions = {"actuator1": 10, "actuator2": 20}

        actuators_dict.operate(actions)

        # Check that each actuator received its respective action
        assert actuators_dict["actuator1"].last_action == 10
        assert actuators_dict["actuator2"].last_action == 20

    def test_operate_missing_key(self, actuators_dict: ActuatorsDict) -> None:
        """Test that operate() raises KeyError for missing actions."""
        actions = {
            "actuator1": 10
            # actuator2 is missing
        }

        with pytest.raises(KeyError):
            actuators_dict.operate(actions)

    def test_setitem_getitem(self) -> None:
        """Test dictionary-like operations."""
        actuators = ActuatorsDict()
        actuator = DummyActuator()

        actuators["new_actuator"] = actuator

        assert "new_actuator" in actuators
        assert actuators["new_actuator"] is actuator

    def test_setup_propagates(
        self, actuators_dict: ActuatorsDict, mocker: MockerFixture
    ) -> None:
        """Test that setup() propagates to all contained actuators."""
        # Spy on the contained actuators' setup methods
        spy1 = mocker.spy(actuators_dict["actuator1"], "setup")
        spy2 = mocker.spy(actuators_dict["actuator2"], "setup")

        actuators_dict.setup()

        spy1.assert_called_once_with()
        spy2.assert_called_once_with()

    def test_teardown_propagates(
        self, actuators_dict: ActuatorsDict, mocker: MockerFixture
    ) -> None:
        """Test that teardown() propagates to all contained actuators."""
        # Spy on the contained actuators' teardown methods
        spy1 = mocker.spy(actuators_dict["actuator1"], "teardown")
        spy2 = mocker.spy(actuators_dict["actuator2"], "teardown")

        actuators_dict.teardown()

        spy1.assert_called_once_with()
        spy2.assert_called_once_with()

    def test_on_paused_propagates(
        self, actuators_dict: ActuatorsDict, mocker: MockerFixture
    ) -> None:
        """Test that on_paused() propagates to all contained actuators."""
        # Spy on the contained actuators' on_paused methods
        spy1 = mocker.spy(actuators_dict["actuator1"], "on_paused")
        spy2 = mocker.spy(actuators_dict["actuator2"], "on_paused")

        actuators_dict.on_paused()

        spy1.assert_called_once_with()
        spy2.assert_called_once_with()

    def test_on_resumed_propagates(
        self, actuators_dict: ActuatorsDict, mocker: MockerFixture
    ) -> None:
        """Test that on_resumed() propagates to all contained actuators."""
        # Spy on the contained actuators' on_resumed methods
        spy1 = mocker.spy(actuators_dict["actuator1"], "on_resumed")
        spy2 = mocker.spy(actuators_dict["actuator2"], "on_resumed")

        actuators_dict.on_resumed()

        spy1.assert_called_once_with()
        spy2.assert_called_once()

    def test_save_load_state(
        self, actuators_dict: ActuatorsDict, mocker: MockerFixture, tmp_path: Path
    ) -> None:
        """Test saving and loading state."""
        # Spy on the contained actuators' save_state methods
        spy_save1 = mocker.spy(actuators_dict["actuator1"], "save_state")
        spy_save2 = mocker.spy(actuators_dict["actuator2"], "save_state")

        # Spy on the contained actuators' load_state methods
        spy_load1 = mocker.spy(actuators_dict["actuator1"], "load_state")
        spy_load2 = mocker.spy(actuators_dict["actuator2"], "load_state")

        # Create a path for saving/loading
        save_path = tmp_path / "actuators_state"

        # Save state
        actuators_dict.save_state(save_path)

        # Check that save_state was called on all actuators with correct paths
        spy_save1.assert_called_once_with(save_path / "actuator1")
        spy_save2.assert_called_once_with(save_path / "actuator2")

        # Load state
        actuators_dict.load_state(save_path)

        # Check that load_state was called on all actuators with correct paths
        spy_load1.assert_called_once_with(save_path / "actuator1")
        spy_load2.assert_called_once_with(save_path / "actuator2")
