from typing import override
from unittest.mock import MagicMock

import pytest

from pamiq_core.utils.schedulers import (
    Scheduler,
    StepIntervalScheduler,
    TimeIntervalScheduler,
)


# Helper class for testing abstract Scheduler class
class MockScheduler(Scheduler):
    @override
    def __init__(self, callbacks=None):
        super().__init__(callbacks)
        self.available = False

    @override
    def is_available(self):
        return self.available

    @override
    def update(self):
        super().update()


class TestScheduler:
    def test_init_empty(self):
        scheduler = MockScheduler()
        assert scheduler._callbacks == []

    def test_init_with_callbacks(self):
        callbacks = [MagicMock(), MagicMock()]
        scheduler = MockScheduler(callbacks)
        assert scheduler._callbacks == callbacks

    def test_init_with_single_callback(self):
        callback = MagicMock()
        scheduler = MockScheduler(callback)
        assert len(scheduler._callbacks) == 1
        assert callback in scheduler._callbacks

    def test_register_callback(self):
        scheduler = MockScheduler()
        callback = MagicMock()
        scheduler.register_callback(callback)
        assert callback in scheduler._callbacks

    def test_remove_callback(self):
        callback = MagicMock()
        scheduler = MockScheduler([callback])
        scheduler.remove_callback(callback)
        assert callback not in scheduler._callbacks

    def test_remove_nonexistent_callback(self):
        scheduler = MockScheduler()
        callback = MagicMock()
        with pytest.raises(ValueError):
            scheduler.remove_callback(callback)

    def test_update_when_available(self):
        callbacks = [MagicMock(), MagicMock()]
        scheduler = MockScheduler(callbacks)
        scheduler.available = True
        scheduler.update()
        for callback in callbacks:
            callback.assert_called_once()

    def test_update_with_single_callback(self):
        callback = MagicMock()
        scheduler = MockScheduler(callback)
        scheduler.available = True
        scheduler.update()
        callback.assert_called_once()

    def test_update_when_not_available(self):
        callbacks = [MagicMock(), MagicMock()]
        scheduler = MockScheduler(callbacks)
        scheduler.available = False
        scheduler.update()
        for callback in callbacks:
            callback.assert_not_called()


class TestTimeIntervalScheduler:
    def test_init_valid_interval(self):
        scheduler = TimeIntervalScheduler(1.0)
        assert scheduler._interval == 1.0

    def test_init_negative_interval(self):
        with pytest.raises(ValueError):
            TimeIntervalScheduler(-1.0)

    def test_is_available(self, mocker):
        mock_time = mocker.patch("pamiq_core.time.time")
        mock_time.return_value = 0.0

        scheduler = TimeIntervalScheduler(5.0)
        assert scheduler.is_available() is False  # Initial state

        mock_time.return_value = 5.1
        assert scheduler.is_available() is True

    def test_update(self, mocker):
        mock_time = mocker.patch("pamiq_core.time.time")
        mock_time.return_value = 0.0

        callback = MagicMock()
        scheduler = TimeIntervalScheduler(5.0, callback)

        # First update should not execute callbacks
        scheduler.update()
        callback.assert_not_called()

        # Reset mock to check next call
        callback.reset_mock()
        mock_time.return_value = 5.1

        # Second update should not execute callbacks (not enough time elapsed)
        scheduler.update()
        callback.assert_called_once_with()

        # Advance time past interval
        callback.reset_mock()
        mock_time.return_value = 16.0
        scheduler.update()
        callback.assert_called_once()


class TestStepIntervalScheduler:
    def test_init_valid_interval(self):
        scheduler = StepIntervalScheduler(5)
        assert scheduler._interval == 5
        assert scheduler._steps_since_last_call == 0

    def test_init_non_positive_interval(self):
        with pytest.raises(ValueError):
            StepIntervalScheduler(0)
        with pytest.raises(ValueError):
            StepIntervalScheduler(-1)

    def test_is_available(self):
        scheduler = StepIntervalScheduler(3)
        assert scheduler.is_available() is False

        scheduler._steps_since_last_call = 2
        assert scheduler.is_available() is False

        scheduler._steps_since_last_call = 3
        assert scheduler.is_available() is True

        scheduler._steps_since_last_call = 4
        assert scheduler.is_available() is True

    def test_update(self):
        callback = MagicMock()
        scheduler = StepIntervalScheduler(3, [callback])

        # First update (step 1)
        scheduler.update()
        callback.assert_not_called()
        assert scheduler._steps_since_last_call == 1

        # Second update (step 2)
        scheduler.update()
        callback.assert_not_called()
        assert scheduler._steps_since_last_call == 2

        # Third update (step 3) - should execute callback and reset counter
        scheduler.update()
        callback.assert_called_once()
        assert scheduler._steps_since_last_call == 0

        # Fourth update (starts new cycle)
        callback.reset_mock()
        scheduler.update()
        callback.assert_not_called()
        assert scheduler._steps_since_last_call == 1
