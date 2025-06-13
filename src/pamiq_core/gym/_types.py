from dataclasses import dataclass
from typing import Any, SupportsFloat


@dataclass(frozen=True)
class EnvReset[T]:
    obs: T
    info: dict[str, Any]


@dataclass(frozen=True)
class EnvStep[T]:
    obs: T
    reward: SupportsFloat
    truncated: bool
    terminated: bool
    info: dict[str, Any]

    @property
    def done(self) -> bool:
        return self.truncated or self.terminated


type EnvOutput[T] = EnvReset[T] | EnvStep[T]
type GymObs[T] = EnvOutput[T] | tuple[EnvStep[T], EnvReset[T]]


@dataclass(frozen=True)
class GymAction[T]:
    action: T
    need_reset: bool
