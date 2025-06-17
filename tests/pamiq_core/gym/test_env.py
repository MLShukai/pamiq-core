"""Tests for GymEnvironment class."""

import gymnasium as gym
import numpy as np
import pytest
from pytest_mock import MockerFixture

from pamiq_core.gym import GymEnvironment
from pamiq_core.gym.types import EnvReset, EnvStep, GymAction
from pamiq_core.interaction import Environment


class TestGymEnvironment:
    """Tests for GymEnvironment class."""

    @pytest.fixture
    def mock_gym_env(self, mocker: MockerFixture):
        """Fixture providing a mock Gymnasium environment."""
        return mocker.Mock(spec=gym.Env)

    def test_inherits_from_environment(self):
        """Test that GymEnvironment inherits from Environment."""
        assert issubclass(GymEnvironment, Environment)

    def test_init_with_env_instance(self, mock_gym_env):
        """Test initialization with a Gym environment instance."""
        gym_env = GymEnvironment(mock_gym_env)
        assert gym_env.env is mock_gym_env

    def test_init_with_env_string(self, mocker: MockerFixture):
        """Test initialization with environment ID string."""
        mock_env_instance = mocker.Mock(spec=gym.Env)
        mock_make = mocker.patch("gymnasium.make", return_value=mock_env_instance)

        gym_env = GymEnvironment("CartPole-v1")

        mock_make.assert_called_once_with("CartPole-v1")
        assert gym_env.env is mock_env_instance

    def test_init_with_env_string_and_kwargs(self, mocker: MockerFixture):
        """Test initialization with environment ID and additional kwargs."""
        mock_env_instance = mocker.Mock(spec=gym.Env)
        mock_make = mocker.patch("gymnasium.make", return_value=mock_env_instance)

        gym_env = GymEnvironment("CartPole-v1", render_mode="rgb_array", max_steps=100)

        mock_make.assert_called_once_with(
            "CartPole-v1", render_mode="rgb_array", max_steps=100
        )
        assert gym_env.env is mock_env_instance

    def test_setup_resets_environment(self, mock_gym_env):
        """Test that setup() resets the environment and stores initial
        observation."""
        mock_obs = np.array([0.0, 0.0, 0.0, 0.0])
        mock_info = {"reset_info": True}
        mock_gym_env.reset.return_value = (mock_obs, mock_info)

        gym_env = GymEnvironment(mock_gym_env)
        gym_env.setup()

        mock_gym_env.reset.assert_called_once()
        obs = gym_env.observe()
        assert isinstance(obs, EnvReset)
        assert np.array_equal(obs.obs, mock_obs)
        assert obs.info == mock_info

    def test_observe_returns_current_observation(self, mock_gym_env):
        """Test that observe() returns the current observation."""
        mock_gym_env.reset.return_value = (np.zeros(4), {})

        gym_env = GymEnvironment(mock_gym_env)
        gym_env.setup()

        # Initial observation should be EnvReset
        obs1 = gym_env.observe()
        assert isinstance(obs1, EnvReset)

        # Set up step return value
        mock_gym_env.step.return_value = (np.ones(4), 1.0, False, False, {"step": 1})

        # After a step, should be EnvStep
        gym_env.affect(GymAction(0, False))
        obs2 = gym_env.observe()
        assert isinstance(obs2, EnvStep)

    def test_affect_performs_step(self, mock_gym_env):
        """Test that affect() performs a step in the environment."""
        mock_gym_env.reset.return_value = (np.zeros(4), {})

        # Set up step return values
        step_obs = np.array([0.1, 0.2, 0.3, 0.4])
        step_reward = 1.0
        step_terminated = False
        step_truncated = False
        step_info = {"step": 1}
        mock_gym_env.step.return_value = (
            step_obs,
            step_reward,
            step_terminated,
            step_truncated,
            step_info,
        )

        gym_env = GymEnvironment(mock_gym_env)
        gym_env.setup()

        # Perform a step
        action = GymAction(1, False)
        gym_env.affect(action)

        mock_gym_env.step.assert_called_once_with(1)

        obs = gym_env.observe()
        assert isinstance(obs, EnvStep)
        assert np.array_equal(obs.obs, step_obs)
        assert obs.reward == step_reward
        assert obs.terminated is step_terminated
        assert obs.truncated is step_truncated
        assert obs.info == step_info

    def test_affect_handles_episode_end_terminated(self, mock_gym_env):
        """Test that affect() handles episode termination."""
        mock_gym_env.reset.return_value = (np.zeros(4), {"reset": True})

        # Set up step to return terminated=True
        final_obs = np.array([1.0, 2.0, 3.0, 4.0])
        mock_gym_env.step.return_value = (final_obs, 10.0, True, False, {"final": True})

        gym_env = GymEnvironment(mock_gym_env)
        gym_env.setup()
        gym_env.affect(GymAction(0, False))

        obs = gym_env.observe()
        assert isinstance(obs, tuple)
        assert len(obs) == 2

        # First element should be the final step
        final_step = obs[0]
        assert isinstance(final_step, EnvStep)
        assert final_step.terminated is True
        assert final_step.done is True
        assert np.array_equal(final_step.obs, final_obs)

        # Second element should be the reset
        reset_obs = obs[1]
        assert isinstance(reset_obs, EnvReset)
        assert np.array_equal(reset_obs.obs, np.zeros(4))

    def test_affect_handles_episode_end_truncated(self, mock_gym_env):
        """Test that affect() handles episode truncation."""
        mock_gym_env.reset.return_value = (np.zeros(4), {"reset": True})

        # Set up step to return truncated=True
        mock_gym_env.step.return_value = (
            np.ones(4),
            5.0,
            False,
            True,
            {"truncated": True},
        )

        gym_env = GymEnvironment(mock_gym_env)
        gym_env.setup()
        gym_env.affect(GymAction(0, False))

        obs = gym_env.observe()
        assert isinstance(obs, tuple)
        assert obs[0].truncated is True
        assert obs[0].done is True

    def test_affect_handles_manual_reset_request(self, mock_gym_env):
        """Test that affect() handles manual reset request from agent."""
        mock_gym_env.reset.return_value = (np.zeros(4), {"reset": True})
        mock_gym_env.step.return_value = (np.ones(4), 1.0, False, False, {"step": 1})

        gym_env = GymEnvironment(mock_gym_env)
        gym_env.setup()

        # Request reset even though episode hasn't ended
        gym_env.affect(GymAction(0, need_reset=True))

        obs = gym_env.observe()
        assert isinstance(obs, tuple)
        assert isinstance(obs[0], EnvStep)
        assert isinstance(obs[1], EnvReset)

        # Should have called reset twice (once in setup, once for manual reset)
        assert mock_gym_env.reset.call_count == 2

    def test_del_closes_environment(self, mock_gym_env):
        """Test that __del__ closes the environment."""
        gym_env = GymEnvironment(mock_gym_env)

        # Manually call __del__
        gym_env.__del__()
        mock_gym_env.close.assert_called_once()
