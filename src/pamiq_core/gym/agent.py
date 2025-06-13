from abc import abstractmethod
from dataclasses import asdict
from typing import Any, SupportsFloat, override

from pamiq_core.interaction import Agent

from ._types import EnvReset, EnvStep, GymAction, GymObs


class GymAgent[O, A](Agent[GymObs[O], GymAction[A]]):
    need_reset: bool = False

    @override
    def setup(self) -> None:
        super().setup()
        self.need_reset = False

    @abstractmethod
    def on_reset(self, obs: O, info: dict[str, Any]) -> A:
        pass

    @abstractmethod
    def on_step(
        self,
        obs: O,
        reward: SupportsFloat,
        truncated: bool,
        terminated: bool,
        info: dict[str, Any],
    ) -> A:
        pass

    def _on_reset(self, obs: O, info: dict[str, Any]) -> A:
        self.need_reset = False
        return self.on_reset(obs, info)

    @override
    def step(self, observation: GymObs[O]) -> GymAction[A]:
        match observation:
            case EnvReset():
                action = self._on_reset(**asdict(observation))
            case EnvStep():
                action = self.on_step(**asdict(observation))
            case tuple():
                self.on_step(**asdict(observation[0]))
                action = self._on_reset(**asdict(observation[1]))
        return GymAction(action, self.need_reset)
