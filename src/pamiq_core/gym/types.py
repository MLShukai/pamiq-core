from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EnvReset[T]:
    """Observation data after environment reset."""

    obs: T
    info: dict[str, Any]


@dataclass(frozen=True)
class EnvStep[T]:
    """Observation data after environment step."""

    obs: T
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any]

    @property
    def done(self) -> bool:
        """Check if episode has ended (either truncated or terminated)."""
        return self.truncated or self.terminated


# Single output from environment (either reset or step)
type EnvOutput[T] = EnvReset[T] | EnvStep[T]
# Full observation type including combined step+reset for episode end
type GymObs[T] = EnvOutput[T] | tuple[EnvStep[T], EnvReset[T]]


@dataclass(frozen=True)
class GymAction[T]:
    """Action wrapper with reset request flag."""

    action: T
    need_reset: bool
