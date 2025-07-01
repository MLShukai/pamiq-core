import pytest
from pytest_mock import MockerFixture

from pamiq_core.data import DataBuffer, DataCollectorsDict, DataUsersDict
from pamiq_core.interaction import Agent
from pamiq_core.model import (
    InferenceModel,
    InferenceModelsDict,
    TrainingModel,
    TrainingModelsDict,
)
from pamiq_core.testing import (
    ConnectedComponents,
    connect_components,
    create_mock_buffer,
    create_mock_models,
)
from pamiq_core.trainer import Trainer


class TestConnectComponents:
    """Test suite for connect_components function."""

    @pytest.fixture
    def mock_agent(self, mocker: MockerFixture) -> Agent:
        """Fixture for a mock agent."""
        agent = mocker.Mock(spec=Agent)
        return agent

    @pytest.fixture
    def mock_trainer(self, mocker: MockerFixture) -> Trainer:
        """Fixture for a mock trainer."""
        trainer = mocker.Mock(spec=Trainer)
        return trainer

    @pytest.fixture
    def mock_buffer(self, mocker: MockerFixture) -> DataBuffer:
        """Fixture for a mock data buffer."""
        buffer = mocker.MagicMock(spec=DataBuffer)
        buffer.max_queue_size = 0
        return buffer

    @pytest.fixture
    def mock_training_model(self, mocker: MockerFixture) -> TrainingModel:
        """Fixture for a mock training model."""
        model = mocker.Mock(spec=TrainingModel)
        model.has_inference_model = True
        model.inference_thread_only = False
        model.inference_model = mocker.Mock(spec=InferenceModel)
        return model

    def test_connect_components_empty(self) -> None:
        """Test connect_components with no parameters."""
        result = connect_components()

        assert isinstance(result, ConnectedComponents)
        assert isinstance(result.data_users, DataUsersDict)
        assert isinstance(result.data_collectors, DataCollectorsDict)
        assert isinstance(result.training_models, TrainingModelsDict)
        assert isinstance(result.inference_models, InferenceModelsDict)

        assert len(result.data_users) == 0
        assert len(result.data_collectors) == 0
        assert len(result.training_models) == 0
        assert len(result.inference_models) == 0

    def test_connect_components_with_agent(self, mock_agent) -> None:
        """Test connect_components with only an agent."""
        result = connect_components(agent=mock_agent)

        # Verify agent's methods were called with correct arguments
        mock_agent.attach_data_collectors.assert_called_once()
        mock_agent.attach_inference_models.assert_called_once()

        # Verify the collectors and models passed to agent are the ones returned
        assert (
            mock_agent.attach_data_collectors.call_args[0][0] is result.data_collectors
        )
        assert (
            mock_agent.attach_inference_models.call_args[0][0]
            is result.inference_models
        )

    def test_connect_components_with_single_trainer(self, mock_trainer) -> None:
        """Test connect_components with a single trainer."""
        result = connect_components(trainers=mock_trainer)

        # Verify trainer's methods were called with correct arguments
        mock_trainer.attach_data_users.assert_called_once()
        mock_trainer.attach_training_models.assert_called_once()

        # Verify the users and models passed to trainer are the ones returned
        assert mock_trainer.attach_data_users.call_args[0][0] is result.data_users
        assert (
            mock_trainer.attach_training_models.call_args[0][0]
            is result.training_models
        )

    def test_connect_components_with_multiple_trainers(
        self, mock_trainer, mocker: MockerFixture
    ) -> None:
        """Test connect_components with multiple trainers."""
        second_trainer = mocker.Mock(spec=Trainer)
        trainers = {"trainer1": mock_trainer, "trainer2": second_trainer}

        result = connect_components(trainers=trainers)

        # Verify both trainers' methods were called with correct arguments
        mock_trainer.attach_data_users.assert_called_once()
        mock_trainer.attach_training_models.assert_called_once()
        second_trainer.attach_data_users.assert_called_once()
        second_trainer.attach_training_models.assert_called_once()

        # Verify the users and models passed to trainers are the ones returned
        assert mock_trainer.attach_data_users.call_args[0][0] is result.data_users
        assert (
            mock_trainer.attach_training_models.call_args[0][0]
            is result.training_models
        )

    def test_connect_components_with_buffers(self, mock_buffer: DataBuffer) -> None:
        """Test connect_components with data buffers."""
        buffers = {"buffer1": mock_buffer}

        result = connect_components(buffers=buffers)

        # Check that buffers were included in the result
        assert "buffer1" in result.data_users
        assert "buffer1" in result.data_collectors

    def test_connect_components_with_models(
        self, mock_training_model: TrainingModel
    ) -> None:
        """Test connect_components with training models."""
        models = {"model1": mock_training_model}

        result = connect_components(models=models)

        # Verify models were added to the training_models dictionary
        assert "model1" in result.training_models
        assert result.training_models["model1"] is mock_training_model

        # Verify inference model is in the inference_models dictionary
        assert "model1" in result.inference_models
        assert result.inference_models["model1"] is mock_training_model.inference_model

    def test_connect_components_all_parameters(
        self, mock_agent, mock_trainer, mock_buffer, mock_training_model
    ) -> None:
        """Test connect_components with all parameters provided."""
        result = connect_components(
            agent=mock_agent,
            trainers=mock_trainer,
            buffers={"buffer1": mock_buffer},
            models={"model1": mock_training_model},
        )

        # Verify agent connections
        mock_agent.attach_data_collectors.assert_called_once()
        mock_agent.attach_inference_models.assert_called_once()

        # Verify trainer connections
        mock_trainer.attach_data_users.assert_called_once()
        mock_trainer.attach_training_models.assert_called_once()

        # Verify return values contain all components
        assert "buffer1" in result.data_users
        assert "buffer1" in result.data_collectors
        assert "model1" in result.training_models
        assert "model1" in result.inference_models


from unittest.mock import MagicMock, Mock


class TestCreateMockModels:
    """Test suite for create_mock_models helper function."""

    def test_default_configuration(self) -> None:
        """Test create_mock_models with default parameters."""
        # Default values: has_inference_model=True, inference_thread_only=False
        training_model, inference_model = create_mock_models()

        # Verify the models are correctly configured
        assert training_model.has_inference_model is True
        assert training_model.inference_thread_only is False
        assert training_model.inference_model is inference_model
        assert isinstance(training_model, Mock)
        assert isinstance(inference_model, Mock)
        assert isinstance(training_model, TrainingModel)
        assert isinstance(inference_model, InferenceModel)

    def test_without_inference_model(self) -> None:
        """Test create_mock_models with has_inference_model=False."""
        training_model, inference_model = create_mock_models(has_inference_model=False)

        # Verify the training model doesn't have an inference model
        assert training_model.has_inference_model is False
        assert training_model.inference_thread_only is False
        assert training_model.inference_model != inference_model
        assert isinstance(inference_model, Mock)

    def test_inference_thread_only(self) -> None:
        """Test create_mock_models with inference_thread_only=True."""
        training_model, inference_model = create_mock_models(
            has_inference_model=True, inference_thread_only=True
        )

        # Verify inference_thread_only is correctly set
        assert training_model.has_inference_model is True
        assert training_model.inference_thread_only is True
        assert training_model.inference_model is inference_model


class TestCreateMockBuffer:
    """Test suite for create_mock_buffer helper function."""

    def test_default_configuration(self) -> None:
        """Test create_mock_buffer with default parameters."""
        buffer = create_mock_buffer()

        # Verify buffer is correctly configured
        assert buffer.max_queue_size == 1
        assert isinstance(buffer, MagicMock)
        assert isinstance(buffer, DataBuffer)

    def test_custom_max_queue_size(self) -> None:
        """Test create_mock_buffer with custom max_queue_size."""
        custom_size = 100
        buffer = create_mock_buffer(max_queue_size=custom_size)

        # Verify buffer has the specified max_queue_size
        assert buffer.max_queue_size == custom_size
