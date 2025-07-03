"""Tests for GymAgent class."""

from typing import Any, override

import numpy as np
import pytest

from pamiq_core.gym import GymAgent
from pamiq_core.gym.types import EnvReset, EnvStep, GymAction
from pamiq_core.interaction import Agent


class GymAgentImpl(GymAgent[np.ndarray, int]):
    """Concrete implementation of GymAgent for testing."""

    def __init__(self):
        super().__init__()
        self.reset_count = 0
        self.step_count = 0
        self.last_obs = None
        self.last_reward = None

    @override
    def on_reset(self, obs: np.ndarray, info: dict[str, Any]) -> int:
        self.reset_count += 1
        self.last_obs = obs
        return 0  # Return action 0 on reset

    @override
    def on_step(
        self,
        obs: np.ndarray,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict[str, Any],
    ) -> int:
        self.step_count += 1
        self.last_obs = obs
        self.last_reward = reward
        # Simple policy: return 1 if first element is positive, else 0
        return 1 if obs[0] > 0 else 0


class TestGymAgent:
    """Tests for GymAgent class."""

    def test_inherits_from_agent(self):
        """Test that GymAgent inherits from Agent."""

        assert issubclass(GymAgent, Agent)

    @pytest.mark.parametrize("method", ["on_reset", "on_step"])
    def test_abstract_methods(self, method):
        """Test that GymAgent has correct abstract methods."""
        assert method in GymAgent.__abstractmethods__

    def test_setup_initializes_need_reset(self):
        """Test that setup() initializes need_reset flag."""
        agent = GymAgentImpl()
        agent.setup()
        assert agent.need_reset is False

    def test_step_with_env_reset(self):
        """Test step() method with EnvReset observation."""
        agent = GymAgentImpl()
        agent.setup()

        obs = np.array([1.0, 2.0, 3.0, 4.0])
        info = {"reset": True}
        env_reset = EnvReset(obs, info)

        action = agent.step(env_reset)

        assert isinstance(action, GymAction)
        assert action.action == 0  # on_reset returns 0
        assert action.need_reset is False
        assert agent.reset_count == 1
        assert agent.step_count == 0
        assert agent.last_obs is not None
        assert np.array_equal(agent.last_obs, obs)

    def test_step_with_env_step(self):
        """Test step() method with EnvStep observation."""
        agent = GymAgentImpl()
        agent.setup()

        obs = np.array([1.0, 2.0, 3.0, 4.0])
        reward = 1.0
        truncated = False
        terminated = False
        info = {"step": 1}
        env_step = EnvStep(obs, reward, terminated, truncated, info)

        action = agent.step(env_step)

        assert isinstance(action, GymAction)
        assert action.action == 1  # on_step returns 1 because obs[0] > 0
        assert action.need_reset is False
        assert agent.reset_count == 0
        assert agent.step_count == 1
        assert agent.last_obs is not None
        assert np.array_equal(agent.last_obs, obs)
        assert agent.last_reward == reward

    def test_step_with_tuple_observation(self):
        """Test step() method with tuple observation (step + reset)."""
        agent = GymAgentImpl()
        agent.setup()

        # Create final step
        final_obs = np.array([2.0, 3.0, 4.0, 5.0])
        final_step = EnvStep(final_obs, 10.0, True, False, {"final": True})

        # Create reset observation
        reset_obs = np.array([0.0, 0.0, 0.0, 0.0])
        reset_info = EnvReset(reset_obs, {"reset": True})

        # Tuple observation
        observation = (final_step, reset_info)

        action = agent.step(observation)

        assert isinstance(action, GymAction)
        assert action.action == 0  # on_reset returns 0
        assert action.need_reset is False
        assert agent.reset_count == 1
        assert agent.step_count == 1  # on_step should have been called once
        # Last obs should be from reset, not step
        assert agent.last_obs is not None
        assert np.array_equal(agent.last_obs, reset_obs)

    def test_need_reset_propagated_to_action(self):
        """Test that need_reset flag is propagated to GymAction."""
        agent = GymAgentImpl()
        agent.setup()

        # Set need_reset flag
        agent.need_reset = True

        obs = np.array([1.0, 2.0, 3.0, 4.0])
        env_step = EnvStep(obs, 1.0, False, False, {})

        action = agent.step(env_step)

        assert action.need_reset is True

    def test_alternating_steps_and_resets(self):
        """Test alternating between steps and resets."""
        agent = GymAgentImpl()
        agent.setup()

        # Step
        env_step = EnvStep(np.ones(4), 1.0, False, False, {})
        agent.step(env_step)

        # Reset
        env_reset = EnvReset(np.zeros(4), {})
        agent.step(env_reset)

        # Step again
        agent.step(env_step)

        assert agent.step_count == 2
        assert agent.reset_count == 1

    def test_need_reset_cleared_after_reset(self):
        """Test that need_reset is cleared after processing a reset."""
        agent = GymAgentImpl()
        agent.setup()

        # Set need_reset flag
        agent.need_reset = True

        # Process a reset
        env_reset = EnvReset(np.zeros(4), {})
        action = agent.step(env_reset)

        # need_reset should be False after reset
        assert agent.need_reset is False
        assert action.need_reset is False
