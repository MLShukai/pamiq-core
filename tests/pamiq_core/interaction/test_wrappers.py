from typing import override

import pytest
from pytest_mock import MockerFixture

from pamiq_core.interaction.env import Environment
from pamiq_core.interaction.event_mixin import InteractionEventMixin
from pamiq_core.interaction.modular_env import Actuator, Sensor
from pamiq_core.interaction.wrappers import (
    ActuatorWrapper,
    EnvironmentWrapper,
    LambdaWrapper,
    SensorWrapper,
    Wrapper,
)
from pamiq_core.state_persistence import PersistentStateMixin
from pamiq_core.thread import ThreadEventMixin


class WrapperImpl(Wrapper[int, str]):
    """Dummy wrapper implementation for testing."""

    @override
    def wrap(self, value: int) -> str:
        return str(value * 2)


class TestWrapper:
    """Test suite for the Wrapper base class."""

    def test_inheritance(self):
        """Test that Wrapper inherits from correct base classes."""
        assert issubclass(Wrapper, InteractionEventMixin)
        assert issubclass(Wrapper, PersistentStateMixin)
        assert issubclass(Wrapper, ThreadEventMixin)

    def test_abstract_methods(self):
        """Test that Wrapper has the correct abstract methods."""
        assert "wrap" in Wrapper.__abstractmethods__

    def test_call_method(self):
        """Test that the __call__ method calls wrap."""
        wrapper = WrapperImpl()
        assert wrapper(5) == "10"  # wrap doubles value and converts to string

    def test_wrap_sensor(self, mocker: MockerFixture):
        """Test that wrap_sensor correctly creates a SensorWrapper."""
        # Create a mock sensor
        mock_sensor = mocker.Mock(spec=Sensor)

        # Create a concrete wrapper implementation
        wrapper = WrapperImpl()

        # Test the method
        result = wrapper.wrap_sensor(mock_sensor)

        # Verify correct SensorWrapper was created
        assert isinstance(result, SensorWrapper)
        assert result.sensor is mock_sensor
        assert result._wrapper is wrapper

    def test_wrap_actuator(self, mocker: MockerFixture):
        """Test that wrap_actuator correctly creates an ActuatorWrapper."""
        # Create a mock actuator
        mock_actuator = mocker.Mock(spec=Actuator)

        # Create a concrete wrapper implementation
        wrapper = WrapperImpl()

        # Test the method
        result = wrapper.wrap_actuator(mock_actuator)

        # Verify correct ActuatorWrapper was created
        assert isinstance(result, ActuatorWrapper)
        assert result.actuator is mock_actuator
        assert result._wrapper is wrapper


class TestLambdaWrapper:
    """Test suite for the LambdaWrapper class."""

    def test_inheritance(self):
        """Test that LambdaWrapper inherits from Wrapper."""
        assert issubclass(LambdaWrapper, Wrapper)

    def test_wrap_with_function(self):
        """Test wrapping with a named function."""

        def double(x):
            return x * 2

        wrapper = LambdaWrapper(double)
        assert wrapper.wrap(5) == 10


class TestEnvironmentWrapper:
    """Test suite for the EnvironmentWrapper class."""

    @pytest.fixture
    def mock_env(self, mocker):
        """Fixture providing a mock environment."""
        env = mocker.Mock(spec=Environment)
        env.observe.return_value = "raw_observation"
        return env

    @pytest.fixture
    def obs_wrapper(self):
        """Fixture providing an observation wrapper."""
        return LambdaWrapper(lambda x: f"wrapped_{x}")

    @pytest.fixture
    def act_wrapper(self):
        """Fixture providing an action wrapper."""
        return LambdaWrapper(lambda x: f"unwrapped_{x}")

    @pytest.fixture
    def env_wrapper(self, mock_env, obs_wrapper, act_wrapper):
        """Fixture providing an EnvironmentWrapper instance."""
        return EnvironmentWrapper(mock_env, obs_wrapper, act_wrapper)

    def test_inheritance(self):
        """Test that EnvironmentWrapper inherits from Environment."""
        assert issubclass(EnvironmentWrapper, Environment)

    def test_observe(self, env_wrapper):
        """Test that observe calls the wrapped environment's observe and
        applies the wrapper."""
        result = env_wrapper.observe()
        assert result == "wrapped_raw_observation"

    def test_affect(self, env_wrapper, mock_env):
        """Test that affect calls the wrapped environment's affect with wrapped
        action."""
        env_wrapper.affect("wrapped_action")
        mock_env.affect.assert_called_once_with("unwrapped_wrapped_action")

    def test_setup(self, mocker, env_wrapper, mock_env, obs_wrapper, act_wrapper):
        """Test that setup calls setup on all components."""
        # Spy on the setup methods
        obs_setup_spy = mocker.spy(obs_wrapper, "setup")
        act_setup_spy = mocker.spy(act_wrapper, "setup")

        env_wrapper.setup()

        mock_env.setup.assert_called_once()
        obs_setup_spy.assert_called_once()
        act_setup_spy.assert_called_once()

    def test_teardown(self, env_wrapper, mock_env, obs_wrapper, act_wrapper, mocker):
        """Test that teardown calls teardown on all components."""
        # Spy on the teardown methods
        obs_teardown_spy = mocker.spy(obs_wrapper, "teardown")
        act_teardown_spy = mocker.spy(act_wrapper, "teardown")

        env_wrapper.teardown()

        mock_env.teardown.assert_called_once()
        obs_teardown_spy.assert_called_once()
        act_teardown_spy.assert_called_once()

    def test_save_state(
        self, env_wrapper, mock_env, obs_wrapper, act_wrapper, mocker, tmp_path
    ):
        """Test that save_state creates directory and calls save_state on all
        components."""
        # Spy on the save_state methods
        obs_save_spy = mocker.spy(obs_wrapper, "save_state")
        act_save_spy = mocker.spy(act_wrapper, "save_state")

        save_path = tmp_path / "test_save"
        env_wrapper.save_state(save_path)

        assert save_path.is_dir()
        mock_env.save_state.assert_called_once_with(save_path / "env")
        obs_save_spy.assert_called_once_with(save_path / "obs_wrapper")
        act_save_spy.assert_called_once_with(save_path / "act_wrapper")

    def test_load_state(
        self, env_wrapper, mock_env, obs_wrapper, act_wrapper, mocker, tmp_path
    ):
        """Test that load_state calls load_state on all components."""
        # Spy on the load_state methods
        obs_load_spy = mocker.spy(obs_wrapper, "load_state")
        act_load_spy = mocker.spy(act_wrapper, "load_state")

        load_path = tmp_path / "test_load"
        env_wrapper.load_state(load_path)

        mock_env.load_state.assert_called_once_with(load_path / "env")
        obs_load_spy.assert_called_once_with(load_path / "obs_wrapper")
        act_load_spy.assert_called_once_with(load_path / "act_wrapper")

    def test_with_function_wrappers(self, mock_env):
        """Test creating EnvironmentWrapper with direct function wrappers."""
        # Create wrapper with functions directly
        env_wrapper = EnvironmentWrapper(
            mock_env, lambda x: f"func_wrapped_{x}", lambda x: f"func_unwrapped_{x}"
        )

        # Test observation wrapping
        result = env_wrapper.observe()
        assert result == "func_wrapped_raw_observation"

        # Test action wrapping
        env_wrapper.affect("direct_action")
        mock_env.affect.assert_called_with("func_unwrapped_direct_action")

    def test_on_paused(self, env_wrapper, mock_env, obs_wrapper, act_wrapper, mocker):
        """Test that on_paused calls on_paused on all components."""
        # Spy on the on_paused methods
        obs_on_paused_spy = mocker.spy(obs_wrapper, "on_paused")
        act_on_paused_spy = mocker.spy(act_wrapper, "on_paused")

        env_wrapper.on_paused()

        mock_env.on_paused.assert_called_once_with()
        obs_on_paused_spy.assert_called_once_with()
        act_on_paused_spy.assert_called_once_with()

    def test_on_resumed(self, env_wrapper, mock_env, obs_wrapper, act_wrapper, mocker):
        """Test that on_resumed calls on_resumed on all components."""
        # Spy on the on_resumed methods
        obs_on_resumed_spy = mocker.spy(obs_wrapper, "on_resumed")
        act_on_resumed_spy = mocker.spy(act_wrapper, "on_resumed")

        env_wrapper.on_resumed()

        mock_env.on_resumed.assert_called_once_with()
        obs_on_resumed_spy.assert_called_once_with()
        act_on_resumed_spy.assert_called_once_with()


class TestSensorWrapper:
    """Test suite for the SensorWrapper class."""

    @pytest.fixture
    def mock_sensor(self, mocker):
        """Fixture providing a mock sensor."""
        sensor = mocker.Mock(spec=Sensor)
        sensor.read.return_value = "raw_sensor_data"
        return sensor

    @pytest.fixture
    def wrapper(self):
        """Fixture providing a wrapper."""
        return LambdaWrapper(lambda x: f"processed_{x}")

    @pytest.fixture
    def sensor_wrapper(self, mock_sensor, wrapper):
        """Fixture providing a SensorWrapper instance."""
        return SensorWrapper(mock_sensor, wrapper)

    def test_inheritance(self):
        """Test that SensorWrapper inherits from Sensor."""
        assert issubclass(SensorWrapper, Sensor)

    def test_read(self, sensor_wrapper):
        """Test that read calls the wrapped sensor's read and applies the
        wrapper."""
        result = sensor_wrapper.read()
        assert result == "processed_raw_sensor_data"

    def test_setup(self, sensor_wrapper, mock_sensor, wrapper, mocker):
        """Test that setup calls setup on all components."""
        # Spy on the setup method
        wrapper_setup_spy = mocker.spy(wrapper, "setup")

        sensor_wrapper.setup()

        mock_sensor.setup.assert_called_once()
        wrapper_setup_spy.assert_called_once()

    def test_teardown(self, sensor_wrapper, mock_sensor, wrapper, mocker):
        """Test that teardown calls teardown on all components."""
        # Spy on the teardown method
        wrapper_teardown_spy = mocker.spy(wrapper, "teardown")

        sensor_wrapper.teardown()

        mock_sensor.teardown.assert_called_once()
        wrapper_teardown_spy.assert_called_once()

    def test_save_state(self, sensor_wrapper, mock_sensor, wrapper, mocker, tmp_path):
        """Test that save_state creates directory and calls save_state on all
        components."""
        # Spy on the save_state method
        wrapper_save_spy = mocker.spy(wrapper, "save_state")

        save_path = tmp_path / "test_save"
        sensor_wrapper.save_state(save_path)

        assert save_path.is_dir()
        mock_sensor.save_state.assert_called_once_with(save_path / "sensor")
        wrapper_save_spy.assert_called_once_with(save_path / "wrapper")

    def test_load_state(self, sensor_wrapper, mock_sensor, wrapper, mocker, tmp_path):
        """Test that load_state calls load_state on all components."""
        # Spy on the load_state method
        wrapper_load_spy = mocker.spy(wrapper, "load_state")

        load_path = tmp_path / "test_load"
        sensor_wrapper.load_state(load_path)

        mock_sensor.load_state.assert_called_once_with(load_path / "sensor")
        wrapper_load_spy.assert_called_once_with(load_path / "wrapper")

    def test_with_function_wrapper(self, mock_sensor):
        """Test creating SensorWrapper with a direct function wrapper."""
        # Create wrapper with function directly
        sensor_wrapper = SensorWrapper(mock_sensor, lambda x: f"func_processed_{x}")

        # Test reading
        result = sensor_wrapper.read()
        assert result == "func_processed_raw_sensor_data"

    def test_on_paused(self, sensor_wrapper, mock_sensor, wrapper, mocker):
        """Test that on_paused calls on_paused on all components."""
        # Spy on the on_paused method
        wrapper_on_paused_spy = mocker.spy(wrapper, "on_paused")

        sensor_wrapper.on_paused()

        mock_sensor.on_paused.assert_called_once_with()
        wrapper_on_paused_spy.assert_called_once_with()

    def test_on_resumed(self, sensor_wrapper, mock_sensor, wrapper, mocker):
        """Test that on_resumed calls on_resumed on all components."""
        # Spy on the on_resumed method
        wrapper_on_resumed_spy = mocker.spy(wrapper, "on_resumed")

        sensor_wrapper.on_resumed()

        mock_sensor.on_resumed.assert_called_once_with()
        wrapper_on_resumed_spy.assert_called_once_with()


class TestActuatorWrapper:
    """Test suite for the ActuatorWrapper class."""

    @pytest.fixture
    def mock_actuator(self, mocker):
        """Fixture providing a mock actuator."""
        return mocker.Mock(spec=Actuator)

    @pytest.fixture
    def wrapper(self):
        """Fixture providing a wrapper."""
        return LambdaWrapper(lambda x: f"transformed_{x}")

    @pytest.fixture
    def actuator_wrapper(self, mock_actuator, wrapper):
        """Fixture providing an ActuatorWrapper instance."""
        return ActuatorWrapper(mock_actuator, wrapper)

    def test_inheritance(self):
        """Test that ActuatorWrapper inherits from Actuator."""
        assert issubclass(ActuatorWrapper, Actuator)

    def test_operate(self, actuator_wrapper, mock_actuator):
        """Test that operate calls the wrapped actuator's operate with wrapped
        action."""
        actuator_wrapper.operate("input_action")
        mock_actuator.operate.assert_called_once_with("transformed_input_action")

    def test_setup(self, actuator_wrapper, mock_actuator, wrapper, mocker):
        """Test that setup calls setup on all components."""
        # Spy on the setup method
        wrapper_setup_spy = mocker.spy(wrapper, "setup")

        actuator_wrapper.setup()

        mock_actuator.setup.assert_called_once()
        wrapper_setup_spy.assert_called_once()

    def test_teardown(self, actuator_wrapper, mock_actuator, wrapper, mocker):
        """Test that teardown calls teardown on all components."""
        # Spy on the teardown method
        wrapper_teardown_spy = mocker.spy(wrapper, "teardown")

        actuator_wrapper.teardown()

        mock_actuator.teardown.assert_called_once()
        wrapper_teardown_spy.assert_called_once()

    def test_save_state(
        self, actuator_wrapper, mock_actuator, wrapper, mocker, tmp_path
    ):
        """Test that save_state creates directory and calls save_state on all
        components."""
        # Spy on the save_state method
        wrapper_save_spy = mocker.spy(wrapper, "save_state")

        save_path = tmp_path / "test_save"
        actuator_wrapper.save_state(save_path)

        assert save_path.is_dir()
        mock_actuator.save_state.assert_called_once_with(save_path / "actuator")
        wrapper_save_spy.assert_called_once_with(save_path / "wrapper")

    def test_load_state(
        self, actuator_wrapper, mock_actuator, wrapper, mocker, tmp_path
    ):
        """Test that load_state calls load_state on all components."""
        # Spy on the load_state method
        wrapper_load_spy = mocker.spy(wrapper, "load_state")

        load_path = tmp_path / "test_load"
        actuator_wrapper.load_state(load_path)

        mock_actuator.load_state.assert_called_once_with(load_path / "actuator")
        wrapper_load_spy.assert_called_once_with(load_path / "wrapper")

    def test_with_function_wrapper(self, mock_actuator):
        """Test creating ActuatorWrapper with a direct function wrapper."""
        # Create wrapper with function directly
        actuator_wrapper = ActuatorWrapper(
            mock_actuator, lambda x: f"func_transformed_{x}"
        )

        # Test operation
        actuator_wrapper.operate("direct_action")
        mock_actuator.operate.assert_called_once_with("func_transformed_direct_action")

    def test_on_paused(self, actuator_wrapper, mock_actuator, wrapper, mocker):
        """Test that on_paused calls on_paused on all components."""
        # Spy on the on_paused method
        wrapper_on_paused_spy = mocker.spy(wrapper, "on_paused")

        actuator_wrapper.on_paused()

        mock_actuator.on_paused.assert_called_once_with()
        wrapper_on_paused_spy.assert_called_once_with()

    def test_on_resumed(self, actuator_wrapper, mock_actuator, wrapper, mocker):
        """Test that on_resumed calls on_resumed on all components."""
        # Spy on the on_resumed method
        wrapper_on_resumed_spy = mocker.spy(wrapper, "on_resumed")

        actuator_wrapper.on_resumed()

        mock_actuator.on_resumed.assert_called_once_with()
        wrapper_on_resumed_spy.assert_called_once_with()
