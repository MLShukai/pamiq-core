import time as original_time
from pathlib import Path

import pytest

from pamiq_core import time as pamiq_time
from pamiq_core.time import TimeController, get_global_time_controller
from tests.helpers import (
    skip_if_kernel_is_linuxkit,
    skip_if_platform_is_darwin,
    skip_if_platform_is_windows,
)


def test_module_global_values():
    controller = pamiq_time._time_controller
    assert pamiq_time.is_paused == controller.is_paused
    assert pamiq_time.sleep == controller.sleep
    assert pamiq_time.time == controller.time
    assert pamiq_time.monotonic == controller.monotonic
    assert pamiq_time.set_time_scale == controller.set_time_scale
    assert pamiq_time.get_time_scale == controller.get_time_scale
    assert pamiq_time.pause == controller.pause
    assert pamiq_time.resume == controller.resume
    assert pamiq_time.state_dict == controller.state_dict
    assert pamiq_time.load_state_dict == controller.load_state_dict

    assert pamiq_time.fixed_sleep == original_time.sleep
    assert pamiq_time.fixed_time == original_time.time


def test_get_global_time_controller():
    assert get_global_time_controller() is pamiq_time._time_controller


@pytest.fixture
def controller():
    """Provide a fresh TimeController instance for each test."""
    return TimeController()


def test_default_time_scale(controller):
    """Verify the default time scale is 1.0."""
    assert controller.get_time_scale() == 1.0


def test_set_get_time_scale(controller):
    """Verify time scale can be set and retrieved."""
    test_scales = [0.5, 1.0, 2.0, 10.0]
    for scale in test_scales:
        controller.set_time_scale(scale)
        assert controller.get_time_scale() == scale


def test_set_invalid_time_scale(controller):
    """Verify setting invalid time scales raises an error."""
    invalid_scales = [-1.0, 0.0]
    for scale in invalid_scales:
        with pytest.raises(AssertionError):
            controller.set_time_scale(scale)


def test_time_monotonicity(controller):
    """Verify time values are strictly increasing."""
    times = [controller.time() for _ in range(5)]
    assert all(t2 >= t1 for t1, t2 in zip(times, times[1:]))


def test_perf_counter_monotonicity(controller):
    """Verify perf_counter values are strictly increasing."""
    times = [controller.perf_counter() for _ in range(5)]
    assert all(t2 >= t1 for t1, t2 in zip(times, times[1:]))


def test_monotonic_monotonicity(controller):
    """Verify monotonic values are strictly increasing."""
    times = [controller.monotonic() for _ in range(5)]
    assert all(t2 >= t1 for t1, t2 in zip(times, times[1:]))


@skip_if_platform_is_windows()
@skip_if_platform_is_darwin()
def test_sleep_duration(controller):
    """Verify sleep duration at normal speed."""
    sleep_time = 0.1
    start = original_time.time()
    controller.sleep(sleep_time)
    elapsed = original_time.time() - start
    assert elapsed == pytest.approx(0.1, abs=0.01)


@skip_if_platform_is_windows()
@skip_if_platform_is_darwin()
@skip_if_kernel_is_linuxkit()
def test_sleep_with_time_scale(controller):
    """Verify sleep duration is affected by time scale."""
    controller.set_time_scale(2.0)
    sleep_time = 0.1
    start = original_time.time()
    controller.sleep(sleep_time)
    elapsed = original_time.time() - start
    assert elapsed == pytest.approx(0.05, abs=0.01)


@skip_if_platform_is_windows()
@skip_if_platform_is_darwin()
@skip_if_kernel_is_linuxkit()
def test_time_functions_scale_properly(controller):
    """Verify all time functions respect time scale."""
    scale = 2.0
    controller.set_time_scale(scale)

    funcs = [controller.time, controller.perf_counter, controller.monotonic]
    for func in funcs:
        start = func()
        original_time.sleep(0.1)
        elapsed = func() - start
        assert elapsed == pytest.approx(0.2, abs=0.01)  # 0.1 * 2.0 scale


def test_pause_freezes_all_time_functions(controller):
    """Verify all time functions freeze when paused."""
    funcs = [
        (controller.time, "time"),
        (controller.perf_counter, "perf_counter"),
        (controller.monotonic, "monotonic"),
    ]

    for func, name in funcs:
        start_value = func()
        controller.pause()
        original_time.sleep(0.1)
        paused_value = func()
        assert start_value == pytest.approx(
            paused_value, abs=0.001
        ), f"{name} not frozen during pause"
        controller.resume()


@skip_if_platform_is_darwin()
@skip_if_kernel_is_linuxkit()
def test_resume_continues_time_properly(controller):
    """Verify time continues correctly after resume."""
    start_time = controller.time()
    controller.pause()
    original_time.sleep(0.1)
    controller.resume()
    original_time.sleep(0.1)
    end_time = controller.time()
    assert end_time - start_time == pytest.approx(0.1, abs=0.01)


@skip_if_platform_is_darwin()
@skip_if_kernel_is_linuxkit()
def test_pause_resume_with_time_scale(controller):
    """Verify pause/resume works correctly with different time scales."""
    controller.set_time_scale(2.0)
    start_time = controller.time()

    controller.pause()
    original_time.sleep(0.1)
    controller.resume()
    original_time.sleep(0.1)
    end_time = controller.time()

    assert end_time - start_time == pytest.approx(0.2, abs=0.01)


@skip_if_platform_is_darwin()
@skip_if_kernel_is_linuxkit()
def test_multiple_pause_resume_cycles(controller):
    """Verify multiple pause/resume cycles work correctly."""
    start_time = controller.time()
    total_unpaused_sleep = 0

    # First cycle
    controller.pause()
    original_time.sleep(0.1)  # Should not count
    controller.resume()
    original_time.sleep(0.1)  # Should count
    total_unpaused_sleep += 0.1

    # Second cycle
    controller.pause()
    original_time.sleep(0.1)  # Should not count
    controller.resume()
    original_time.sleep(0.1)  # Should count
    total_unpaused_sleep += 0.1

    end_time = controller.time()
    assert end_time - start_time == pytest.approx(total_unpaused_sleep, abs=0.02)


def test_pause_resume_preserves_monotonicity(controller):
    """Verify time remains monotonic through pause/resume cycles."""
    times = []

    times.append(controller.time())
    controller.pause()
    times.append(controller.time())
    original_time.sleep(0.1)
    times.append(controller.time())
    controller.resume()
    times.append(controller.time())
    original_time.sleep(0.1)
    times.append(controller.time())

    assert all(t2 >= t1 for t1, t2 in zip(times, times[1:]))


def test_sleep_during_pause(controller):
    """Verify sleep returns immediately when system is paused."""
    controller.pause()
    start_time = original_time.time()
    controller.sleep(1.0)  # Should return immediately
    elapsed = original_time.time() - start_time
    assert elapsed < 0.001


def test_pause_idempotent(controller):
    """Verify multiple pauses don't affect the behavior."""
    controller.pause()
    first_pause_time = controller.time()
    controller.pause()  # Second pause should have no effect
    second_pause_time = controller.time()

    assert first_pause_time == pytest.approx(second_pause_time, abs=0.001)


def test_resume_idempotent(controller):
    """Verify multiple resumes don't affect the behavior."""
    controller.pause()
    original_time.sleep(0.1)

    controller.resume()
    first_resume_time = controller.time()
    controller.resume()  # Second resume should have no effect
    second_resume_time = controller.time()

    assert abs(second_resume_time - first_resume_time) < 0.01


@skip_if_platform_is_darwin()
@skip_if_kernel_is_linuxkit()
def test_pause_resume_fixed_time_unaffected():
    """Verify fixed time functions are not affected by pause/resume."""
    start_fixed = pamiq_time.fixed_time()
    pamiq_time.pause()
    original_time.sleep(0.1)
    pause_fixed = pamiq_time.fixed_time()
    pamiq_time.resume()
    end_fixed = pamiq_time.fixed_time()

    # Fixed time should continue normally regardless of pause
    assert pause_fixed - start_fixed == pytest.approx(0.1, abs=0.01)
    assert end_fixed - start_fixed == pytest.approx(0.1, abs=0.01)


@skip_if_platform_is_darwin()
@skip_if_kernel_is_linuxkit()
def test_get_and_load_state_dict():
    """Verify state dict."""
    controller = TimeController()
    state = controller.state_dict()
    assert isinstance(state, dict)
    assert state["scaled_anchor_time"] == controller._scaled_anchor_time
    assert state["scaled_anchor_monotonic"] == controller._scaled_anchor_monotonic
    assert state["scaled_anchor_perf_counter"] == controller._scaled_anchor_perf_counter

    original_time.sleep(0.1)
    new_controller = TimeController()
    assert new_controller.state_dict() != state

    new_controller.load_state_dict(state)
    func_names = ["time", "perf_counter", "monotonic"]
    for name in func_names:
        func = getattr(controller, name)
        new_func = getattr(new_controller, name)
        assert func() == pytest.approx(
            new_func() + 0.1, abs=0.01
        ), f"'{name}' is not consistent."

    original_time.sleep(0.1)

    new_controller.load_state_dict(state)
    controller.load_state_dict(state)
    for name in func_names:
        func = getattr(controller, name)
        new_func = getattr(new_controller, name)
        assert func() == pytest.approx(
            new_func(), abs=0.001
        ), f"'{name}' is not consistent."


def test_is_paused(controller):
    assert not controller.is_paused()
    controller.pause()
    assert controller.is_paused()
    controller.resume()
    assert not controller.is_paused()


def test_save_and_load_state(controller: TimeController, tmp_path: Path):
    time_path = tmp_path / "time"

    controller.save_state(time_path)

    assert time_path.with_suffix(".pkl").is_file()

    controller.load_state(time_path)
