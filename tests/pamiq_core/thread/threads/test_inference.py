import time

import pytest
from pytest_mock import MockerFixture

from pamiq_core.interaction import Interaction
from pamiq_core.thread import (
    ReadOnlyController,
    ThreadController,
    ThreadTypes,
)
from pamiq_core.thread.threads.inference import InferenceThread
from tests.helpers import check_log_message


class TestInferenceThread:
    """Test suite for InferenceThread class."""

    @pytest.fixture
    def mock_interaction(self, mocker: MockerFixture) -> Interaction:
        """Fixture providing a mock Interaction object."""
        return mocker.Mock(spec=Interaction)

    @pytest.fixture
    def thread_controller(self) -> ThreadController:
        """Fixture providing a ThreadController instance."""
        return ThreadController()

    @pytest.fixture
    def read_only_controller(self, thread_controller) -> ReadOnlyController:
        """Fixture providing a ReadOnlyController instance."""
        return thread_controller.read_only

    @pytest.fixture
    def inference_thread(self, mock_interaction) -> InferenceThread:
        """Fixture providing an InferenceThread instance with a mock
        interaction."""
        return InferenceThread(mock_interaction, log_tick_time_statistics_interval=0.05)

    @pytest.fixture
    def inference_thread_with_controller(
        self, mock_interaction, read_only_controller: ReadOnlyController
    ) -> InferenceThread:
        """Fixture providing an InferenceThread with controller attached."""
        thread = InferenceThread(mock_interaction)
        thread.attach_controller(read_only_controller)
        return thread

    def test_thread_type(self, inference_thread: InferenceThread) -> None:
        """Test that thread type is correctly set to INFERENCE."""
        assert inference_thread.THREAD_TYPE is ThreadTypes.INFERENCE

    def test_on_start(
        self, inference_thread: InferenceThread, mock_interaction
    ) -> None:
        """Test that on_start calls interaction.setup()."""
        inference_thread.on_start()
        mock_interaction.setup.assert_called_once()

    def test_on_tick(self, inference_thread: InferenceThread, mock_interaction) -> None:
        """Test that on_tick calls interaction.step()."""
        inference_thread.on_tick()
        mock_interaction.step.assert_called_once()

    def test_on_finally(
        self, inference_thread: InferenceThread, mock_interaction
    ) -> None:
        """Test that on_finally calls interaction.teardown()."""
        inference_thread.on_finally()
        mock_interaction.teardown.assert_called_once()

    def test_on_paused(
        self, inference_thread: InferenceThread, mock_interaction
    ) -> None:
        """Test that on_paused calls interaction.on_paused()."""
        inference_thread.on_paused()
        mock_interaction.on_paused.assert_called_once()

    def test_on_resumed(
        self, inference_thread: InferenceThread, mock_interaction
    ) -> None:
        """Test that on_resumed calls interaction.on_resumed()."""
        inference_thread.on_resumed()
        mock_interaction.on_resumed.assert_called_once()

    def test_log_tick_time_statistics(
        self,
        thread_controller: ThreadController,
        inference_thread: InferenceThread,
        caplog,
    ) -> None:
        """Test that log tick time statistics."""

        inference_thread.attach_controller(thread_controller.read_only)

        inference_thread.start()
        time.sleep(0.1)
        thread_controller.shutdown()
        inference_thread.join()

        check_log_message(
            r"Step time: (\d+\.\d+e[+-]\d+) ± (\d+\.\d+e[+-]\d+) \[s\] in (\d+) steps\.",
            "INFO",
            caplog,
        )
