from typing import Any, override

import gymnasium as gym

from pamiq_core.interaction import Environment

from ._types import EnvReset, EnvStep, GymAction, GymObs


class GymEnvironment[O, A](Environment[GymObs[O], GymAction[A]]):
    _obs: GymObs[O]

    def __init__(self, env: gym.Env[O, A] | str, **gym_make_kwds: Any) -> None:
        super().__init__()
        if isinstance(env, str):
            self.env: gym.Env[O, A] = gym.make(env, **gym_make_kwds)  # pyright: ignore[reportUnknownMemberType, ]
        else:
            self.env = env

    @override
    def setup(self) -> None:
        super().setup()
        self._obs = EnvReset(*self.env.reset())

    @override
    def observe(self) -> GymObs[O]:
        return self._obs

    @override
    def affect(self, action: GymAction[A]) -> None:
        obs = EnvStep(*self.env.step(action.action))
        if obs.done or action.need_reset:
            obs = (obs, EnvReset(*self.env.reset()))
        self._obs = obs

    def __del__(self) -> None:
        if hasattr(self, "env"):
            self.env.close()
