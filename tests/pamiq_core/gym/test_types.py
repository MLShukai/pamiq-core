"""Tests for gym type definitions."""

import pytest

from pamiq_core.gym.types import EnvStep


class TestEnvStep:
    """Tests for EnvStep dataclass."""

    @pytest.mark.parametrize(
        "terminated,truncated,expected",
        [
            (False, False, False),
            (True, False, True),
            (False, True, True),
            (True, True, True),
        ],
    )
    def test_done_property(self, terminated, truncated, expected):
        """Test done property returns correct value based on terminated and
        truncated flags."""
        env_step = EnvStep(
            obs="test_obs",
            reward=1.0,
            terminated=terminated,
            truncated=truncated,
            info={},
        )
        assert env_step.done is expected
