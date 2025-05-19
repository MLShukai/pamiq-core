"""`IntervalAdjustor` verifies its performance based on the variance and mean
of multiple `adjust` calls."""

import statistics

import pytest

from pamiq_core.interaction.interval_adjustors import (
    IntervalAdjustor,
    SleepIntervalAdjustor,
)
from tests.helpers import skip_if_kernel_is_linuxkit, skip_if_platform_is_darwin


def compute_adjustor_spec(
    adjustor: IntervalAdjustor, num_trial: int
) -> tuple[float, float]:
    """Executes `adjust` for the number of times specified by `num_trial` and
    returns the mean and standard deviation of the elapsed time."""

    delta_times = []
    adjustor.reset()
    for _ in range(num_trial):
        delta_times.append(adjustor.adjust())

    return statistics.mean(delta_times), statistics.stdev(delta_times)


class TestSleepIntervalAdjustor:
    @skip_if_kernel_is_linuxkit()
    @skip_if_platform_is_darwin()
    @pytest.mark.parametrize(
        """interval,num_trial""",
        [
            (0.01, 50),
            (0.05, 10),
        ],
    )
    def test_adjust(self, interval: float, num_trial: int) -> None:
        adjustor = SleepIntervalAdjustor(interval)
        mean, std = compute_adjustor_spec(adjustor, num_trial)
        print(mean, std)
        assert mean == pytest.approx(
            interval, abs=0.001
        )  # Allowable error: 0.001 seconds
        assert std < interval * 0.1
