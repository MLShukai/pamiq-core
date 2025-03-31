from pathlib import Path

import pytest
from pytest_mock import MockerFixture

from pamiq_core.interaction.agent import Agent
from pamiq_core.interaction.env import Environment
from pamiq_core.interaction.interactions import FixedIntervalInteraction, Interaction
from pamiq_core.interaction.interval_adjustors import (
    IntervalAdjustor,
    SleepIntervalAdjustor,
)


class TestInteraction:
    """Tests for the Interaction class."""

    @pytest.fixture
    def mock_agent(self, mocker: MockerFixture):
        """Fixture providing a mock agent."""
        return mocker.Mock(spec=Agent)

    @pytest.fixture
    def mock_environment(self, mocker: MockerFixture):
        """Fixture providing a mock environment."""
        return mocker.Mock(spec=Environment)

    @pytest.fixture
    def interaction(self, mock_agent, mock_environment) -> Interaction:
        """Fixture providing an Interaction instance with mock components."""
        return Interaction(mock_agent, mock_environment)

    def test_init(
        self, interaction: Interaction, mock_agent: Agent, mock_environment: Environment
    ):
        """Test that initialization correctly sets the agent and
        environment."""
        assert interaction.agent == mock_agent
        assert interaction.environment == mock_environment

    def test_step(self, interaction: Interaction, mock_agent, mock_environment):
        """Test that step() correctly coordinates the observation-action
        loop."""
        # Setup mock behavior
        mock_environment.observe.return_value = "test_observation"
        mock_agent.step.return_value = "test_action"

        # Call the method under test
        interaction.step()

        # Verify the expected interactions
        mock_environment.observe.assert_called_once()
        mock_agent.step.assert_called_once_with("test_observation")
        mock_environment.affect.assert_called_once_with("test_action")

    def test_setup(self, interaction: Interaction, mock_agent, mock_environment):
        """Test that setup() calls setup on both agent and environment."""
        interaction.setup()

        mock_agent.setup.assert_called_once()
        mock_environment.setup.assert_called_once()

    def test_teardown(self, interaction: Interaction, mock_agent, mock_environment):
        """Test that teardown() calls teardown on both agent and
        environment."""
        interaction.teardown()

        mock_agent.teardown.assert_called_once()
        mock_environment.teardown.assert_called_once()

    def test_save_state(
        self, interaction: Interaction, mock_agent, mock_environment, tmp_path: Path
    ):
        """Test that save_state() creates directory and calls save_state on
        components."""
        save_path = tmp_path / "test_save"
        interaction.save_state(save_path)

        assert save_path.is_dir()
        mock_agent.save_state.assert_called_once_with(save_path / "agent")
        mock_environment.save_state.assert_called_once_with(save_path / "environment")

    def test_load_state(
        self, interaction: Interaction, mock_agent, mock_environment, tmp_path: Path
    ):
        """Test that load_state() calls load_state on both agent and
        environment."""
        load_path = tmp_path / "test_load"
        interaction.load_state(load_path)

        mock_agent.load_state.assert_called_once_with(load_path / "agent")
        mock_environment.load_state.assert_called_once_with(load_path / "environment")

    def test_on_paused(self, interaction: Interaction, mock_agent, mock_environment):
        """Test that on_paused() calls on_paused on both agent and
        environment."""
        interaction.on_paused()
        mock_agent.on_paused.assert_called_once_with()
        mock_environment.on_paused.assert_called_once_with()

    def test_on_resumed(self, interaction: Interaction, mock_agent, mock_environment):
        """Test that on_resumed() calls on_resumed on both agent and
        environment."""
        interaction.on_resumed()
        mock_agent.on_resumed.assert_called_once_with()
        mock_environment.on_resumed.assert_called_once_with()


class TestFixedIntervalInteraction:
    """Tests for FixedIntervalInteraction class."""

    @pytest.fixture
    def mock_agent(self, mocker: MockerFixture):
        """Fixture providing a mock agent."""
        return mocker.Mock(spec=Agent)

    @pytest.fixture
    def mock_environment(self, mocker: MockerFixture):
        """Fixture providing a mock environment."""
        return mocker.Mock(spec=Environment)

    @pytest.fixture
    def mock_adjustor(self, mocker: MockerFixture):
        """Fixture providing a mock interval adjustor."""
        return mocker.Mock(spec=IntervalAdjustor)

    @pytest.fixture
    def fixed_interval_interaction(self, mock_agent, mock_environment, mock_adjustor):
        """Fixture providing a FixedIntervalInteraction with mock
        components."""
        return FixedIntervalInteraction(mock_agent, mock_environment, mock_adjustor)

    def test_init(
        self, fixed_interval_interaction, mock_agent, mock_environment, mock_adjustor
    ):
        """Test that initialization correctly sets components."""
        assert fixed_interval_interaction.agent == mock_agent
        assert fixed_interval_interaction.environment == mock_environment
        assert fixed_interval_interaction._adjustor == mock_adjustor

    def test_setup(
        self, fixed_interval_interaction, mock_adjustor, mocker: MockerFixture
    ):
        """Test that setup calls parent setup and resets the adjustor."""
        spy_super_setup = mocker.spy(Interaction, "setup")

        fixed_interval_interaction.setup()

        spy_super_setup.assert_called_once_with(fixed_interval_interaction)
        mock_adjustor.reset.assert_called_once_with()

    def test_step(
        self, fixed_interval_interaction, mock_adjustor, mocker: MockerFixture
    ):
        """Test that step calls parent step and adjusts the interval."""
        # Spy on the parent step method
        spy_super_step = mocker.spy(Interaction, "step")

        fixed_interval_interaction.step()

        spy_super_step.assert_called_once_with(fixed_interval_interaction)
        mock_adjustor.adjust.assert_called_once()

    def test_interaction_inheritance(self):
        """Test that FixedIntervalInteraction inherits from Interaction."""
        assert issubclass(FixedIntervalInteraction, Interaction)

    def test_with_sleep_adjustor(self, mock_agent, mock_environment):
        """Test that creating FixedIntervalInteraction instance with
        SleepIntervalAdjustor."""
        interaction = FixedIntervalInteraction.with_sleep_adjustor(
            mock_agent, mock_environment, 1.0
        )
        assert isinstance(interaction._adjustor, SleepIntervalAdjustor)
