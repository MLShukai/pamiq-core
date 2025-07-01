from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
from pytest_mock import MockerFixture

from pamiq_core.data import DataBuffer
from pamiq_core.interaction import Agent, Environment, Interaction
from pamiq_core.launcher import LaunchConfig, launch
from pamiq_core.model import InferenceModel, TrainingModel
from pamiq_core.trainer import Trainer
from tests.helpers import check_log_message


class TestLaunchConfig:
    """Tests for the LaunchConfig dataclass."""

    def test_frozen(self):
        """Test that LaunchConfig instances are immutable."""
        cfg = LaunchConfig()
        with pytest.raises(FrozenInstanceError):
            cfg.max_uptime = 0.0  # pyright: ignore[reportAttributeAccessIssue]


class TestLaunch:
    """Tests for the launch function."""

    @pytest.fixture
    def mock_agent(self, mocker: MockerFixture):
        """Create a mock Agent instance."""
        return mocker.Mock(Agent)

    @pytest.fixture
    def mock_env(self, mocker: MockerFixture):
        """Create a mock Environment instance."""
        return mocker.Mock(Environment)

    @pytest.fixture
    def interaction(self, mock_agent, mock_env) -> Interaction:
        """Create an Interaction with mock agent and environment."""
        return Interaction(mock_agent, mock_env)

    @pytest.fixture
    def mock_trainer(self, mocker: MockerFixture):
        """Create a mock Trainer that does not run training."""
        trainer = mocker.Mock(Trainer)
        trainer.run.return_value = False  # Do not log training time.
        return trainer

    @pytest.fixture
    def mock_buffer(self, mocker: MockerFixture):
        """Create a mock DataBuffer with zero queue size."""
        buf = mocker.MagicMock(DataBuffer)
        buf.max_queue_size = 0
        return buf

    @pytest.fixture
    def mock_model(self, mocker: MockerFixture):
        """Create a mock TrainingModel with inference model."""
        model = mocker.Mock(TrainingModel)
        model.has_inference_model = True
        model.inference_thread_only = False
        model.inference_model = mocker.Mock(InferenceModel)
        return model

    def test_launch(
        self,
        caplog: pytest.LogCaptureFixture,
        tmp_path: Path,
        mock_agent,
        mock_env,
        interaction,
        mock_trainer,
        mock_buffer,
        mock_model,
    ):
        """Test the full launch cycle with state persistence."""
        state_dir = tmp_path / "states"
        cfg = LaunchConfig(
            states_dir=state_dir,
            web_api_address=("localhost", 12345),
            max_uptime=1.0,
            time_scale=2.0,
        )

        launch(
            interaction=interaction,
            models={"model": mock_model},
            buffers={"buffer": mock_buffer},
            trainers={"trainer": mock_trainer},
            config=cfg,
        )

        # Verify state directory was created
        assert state_dir.is_dir()
        assert len(list(state_dir.glob("*"))) > 0
        for path in state_dir.glob("*"):
            assert (path / "interaction").is_dir()
            assert (path / "models").is_dir()
            assert (path / "data").is_dir()
            assert (path / "trainers").is_dir()
            assert (path / "time.pkl").is_file()

        # Check for expected log messages
        check_log_message("Setting time scale to 2.0", "INFO", caplog)
        check_log_message(
            f"Setting up state persistence in directory: {state_dir}", "INFO", caplog
        )
        check_log_message("Launching AMI system...", "INFO", caplog)
        check_log_message("Saving final system state", "INFO", caplog)
        check_log_message("Final state saved to", "INFO", caplog)

        # Verify core components were used
        mock_agent.step.assert_called()
        mock_env.observe.assert_called()
        mock_env.affect.assert_called()
        mock_trainer.run.assert_called()

    def test_launch_loading_state(
        self,
        caplog: pytest.LogCaptureFixture,
        tmp_path: Path,
        interaction,
        mock_trainer,
    ):
        """Test loading from a non-existent state raises an error."""
        state_dir = tmp_path / "states"
        state_path = state_dir / "not_exist"
        cfg = LaunchConfig(
            states_dir=state_dir,
            web_api_address=("localhost", 12345),
            max_uptime=1.0,
            time_scale=2.0,
            saved_state_path=state_path,
        )

        with pytest.raises(FileNotFoundError):
            launch(
                interaction=interaction,
                models={},
                buffers={},
                trainers={"trainer": mock_trainer},
                config=cfg,
            )
        check_log_message(f"Loading state from '{state_path}'", "INFO", caplog)
