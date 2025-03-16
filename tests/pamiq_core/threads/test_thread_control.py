import threading
import time

import pytest

from pamiq_core.threads import (
    ControllerCommandHandler,
    ReadOnlyController,
    ThreadController,
)


class TestThreadController:
    def test_resume_and_related_predicate_methods(self):
        thread_controller = ThreadController()
        thread_controller.resume()

        assert thread_controller.is_resume() is True
        assert thread_controller.is_pause() is False

    def test_resume_when_shutdown(self):
        thread_controller = ThreadController()
        thread_controller.shutdown()

        with pytest.raises(RuntimeError):
            thread_controller.resume()

    def test_pause_and_related_predicate_methods(self):
        thread_controller = ThreadController()
        thread_controller.pause()

        assert thread_controller.is_resume() is False
        assert thread_controller.is_pause() is True

    def test_pause_when_shutdown(self):
        thread_controller = ThreadController()
        thread_controller.shutdown()

        with pytest.raises(RuntimeError, match="ThreadController must be activated before pause()."):
            thread_controller.pause()

    def test_shutdown_and_related_predicate_methods(self):
        thread_controller = ThreadController()
        thread_controller.shutdown()

        assert thread_controller.is_shutdown() is True
        assert thread_controller.is_active() is False

    def test_activate_and_related_predicate_methods(self):
        thread_controller = ThreadController()
        thread_controller.activate()

        assert thread_controller.is_shutdown() is False
        assert thread_controller.is_active() is True

    def test_shutdown_when_already_shutdown(self):
        thread_controller = ThreadController()
        thread_controller.shutdown()

        # Test that `resume()` in `shutdown()` does not raise an error
        # when already shutdown.
        thread_controller.shutdown()

    def test_resume_before_shutdown(self, mocker):
        thread_controller = ThreadController()
        call_order = []
        mocker.patch.object(
            thread_controller, "resume", side_effect=lambda: call_order.append("resume")
        )
        mocker.patch.object(
            thread_controller._shutdown_event,
            "set",
            side_effect=lambda: call_order.append("shutdown_set"),
        )

        thread_controller.shutdown()

        assert call_order == ["resume", "shutdown_set"]
        thread_controller.resume.assert_called_once()
        thread_controller._shutdown_event.set.assert_called_once()

    def test_initial_state(self):
        thread_controller = ThreadController()

        assert thread_controller.is_resume() is True
        assert thread_controller.is_active() is True

    def test_wait_for_resume_when_already_resumed(self):
        thread_controller = ThreadController()

        # immediately return True if already resumed
        thread_controller.resume()
        start = time.perf_counter()
        assert thread_controller.wait_for_resume(timeout=0.1) is True
        assert time.perf_counter() - start < 1e-3

    def test_wait_for_resume_when_already_paused(self):
        thread_controller = ThreadController()

        # wait timeout and return False if paused
        thread_controller.pause()
        start = time.perf_counter()
        assert thread_controller.wait_for_resume(0.1) is False
        assert 0.1 <= time.perf_counter() - start < 0.2

    def test_wait_for_resume_when_resumed_after_waiting(self):
        thread_controller = ThreadController()

        # immediately return True if resumed after waiting
        thread_controller.pause()
        threading.Timer(0.1, thread_controller.resume).start()
        start = time.perf_counter()
        assert thread_controller.wait_for_resume(0.5) is True
        assert 0.1 <= time.perf_counter() - start < 0.2


class TestReadOnlyController:
    def test_exposed_methods(self):
        thread_controller = ThreadController()
        read_only_controller = ReadOnlyController(thread_controller)

        assert read_only_controller.is_resume == thread_controller.is_resume
        assert read_only_controller.is_pause == thread_controller.is_pause
        assert read_only_controller.is_shutdown == thread_controller.is_shutdown
        assert read_only_controller.is_active == thread_controller.is_active
        assert read_only_controller.wait_for_resume == thread_controller.wait_for_resume


class TestControllerCommandHandler:
    def test_stop_if_pause_when_already_resumed(self):
        thread_controller = ThreadController()
        read_only_controller = ReadOnlyController(thread_controller)
        handler = ControllerCommandHandler(read_only_controller)

        # immediately return if already resumed
        thread_controller.resume()
        start = time.perf_counter()
        handler.stop_if_pause()
        assert time.perf_counter() - start < 1e-3

    def test_stop_if_pause_pause_to_resume(self):
        thread_controller = ThreadController()
        read_only_controller = ReadOnlyController(thread_controller)
        handler = ControllerCommandHandler(read_only_controller)

        # immediately return if resumed after waiting
        thread_controller.pause()
        threading.Timer(0.1, thread_controller.resume).start()
        start = time.perf_counter()
        handler.stop_if_pause()
        assert 0.1 <= time.perf_counter() - start < 0.2

    def test_stop_if_pause_when_already_shutdown(self):
        thread_controller = ThreadController()
        read_only_controller = ReadOnlyController(thread_controller)
        handler = ControllerCommandHandler(read_only_controller)

        # immediately return if already shutdown
        thread_controller.shutdown()
        start = time.perf_counter()
        handler.stop_if_pause()
        assert time.perf_counter() - start < 1e-3

    def test_stop_if_pause_pause_to_shutdown(self):
        thread_controller = ThreadController()
        read_only_controller = ReadOnlyController(thread_controller)
        handler = ControllerCommandHandler(read_only_controller)

        # immediately return if shutdown after waiting
        thread_controller.pause()
        threading.Timer(0.1, thread_controller.shutdown).start()
        start = time.perf_counter()
        handler.stop_if_pause()
        assert 0.1 <= time.perf_counter() - start < 0.2

    def test_manage_loop_when_already_resumed(self):
        thread_controller = ThreadController()
        read_only_controller = ReadOnlyController(thread_controller)
        handler = ControllerCommandHandler(read_only_controller)

        # immediately return True if already resumed
        thread_controller.resume()
        start = time.perf_counter()
        assert handler.manage_loop() is True
        assert time.perf_counter() - start < 1e-3

    def test_manage_loop_pause_to_resume(self):
        thread_controller = ThreadController()
        read_only_controller = ReadOnlyController(thread_controller)
        handler = ControllerCommandHandler(read_only_controller)

        # immediately return True if resumed after waiting
        thread_controller.pause()
        threading.Timer(0.1, thread_controller.resume).start()
        start = time.perf_counter()
        assert handler.manage_loop() is True
        assert 0.1 <= time.perf_counter() - start < 0.2

    def test_manage_loop_when_already_shutdown(self):
        thread_controller = ThreadController()
        read_only_controller = ReadOnlyController(thread_controller)
        handler = ControllerCommandHandler(read_only_controller)

        # immediately return False if already shutdown
        thread_controller.shutdown()
        start = time.perf_counter()
        assert handler.manage_loop() is False
        assert time.perf_counter() - start < 1e-3

    def test_manage_loop_pause_to_shutdown(self):
        thread_controller = ThreadController()
        read_only_controller = ReadOnlyController(thread_controller)
        handler = ControllerCommandHandler(read_only_controller)

        # immediately return False if shutdown after waiting
        thread_controller.pause()
        threading.Timer(0.1, thread_controller.shutdown).start()
        start = time.perf_counter()
        assert handler.manage_loop() is False
        assert 0.1 <= time.perf_counter() - start < 0.2

    def test_manage_loop_with_pause_resume_shutdown(self):
        counter = 0

        def inifinity_count():
            nonlocal counter
            while handler.manage_loop():
                counter += 1
                time.sleep(0.001)

        thread_controller = ThreadController()
        read_only_controller = ReadOnlyController(thread_controller)
        handler = ControllerCommandHandler(read_only_controller)

        # increment occur if active & resume
        thread_controller.resume()
        thread = threading.Thread(target=inifinity_count)
        thread.start()
        time.sleep(0.01)
        assert counter > 0

        # increment does not occur if paused
        prev_count = counter
        thread_controller.pause()
        time.sleep(0.01)
        assert counter == prev_count

        # increment does not occur if shutdown (from when thread is paused)
        thread_controller.shutdown()
        time.sleep(0.01)
        thread.join()  # ensure the thread has finished
        assert counter == prev_count  # check that the loop has exited immediately

        # increment does not occur if shutdown (from when thread is resumed)
        thread_controller.activate()
        thread_controller.resume()
        thread = threading.Thread(target=inifinity_count)  # restart the thread
        thread.start()
        time.sleep(0.01)
        prev_count = counter
        thread_controller.shutdown()
        time.sleep(0.01)
        thread.join()  # ensure the thread has finished
        assert counter == prev_count  # check that the loop has exited immediately
