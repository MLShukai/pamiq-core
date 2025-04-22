from enum import Enum, auto


class ThreadTypes(Enum):
    """Declares all thread types."""

    CONTROL = auto()
    INFERENCE = auto()
    TRAINING = auto()

    @property
    def thread_name(self) -> str:
        """Returns the name of thread type."""

        return _THREAD_NAMES[self]


_THREAD_NAMES = {
    ThreadTypes.CONTROL: "control",
    ThreadTypes.INFERENCE: "inference",
    ThreadTypes.TRAINING: "training",
}
