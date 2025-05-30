"""Tests for the Web API server module."""

import json
from queue import Empty

import httpx
import pytest
from pytest_mock import MockerFixture
from starlette.testclient import TestClient

from pamiq_core.console.system_status import SystemStatus, SystemStatusProvider
from pamiq_core.console.web_api import (
    ERROR_INTERNAL_SERVER,
    ERROR_INVALID_ENDPOINT,
    ERROR_INVALID_METHOD,
    RESULT_OK,
    ControlCommands,
    WebApiClient,
    WebApiServer,
)


class TestWebApiServer:
    """Test class for the WebApiServer."""

    @pytest.fixture
    def mock_system_status(self, mocker: MockerFixture) -> SystemStatusProvider:
        """Fixture for a mock SystemStatusProvider."""
        mock = mocker.Mock(spec=SystemStatusProvider)
        mock.get_current_status.return_value = SystemStatus.ACTIVE
        return mock

    @pytest.fixture
    def web_api_server(self, mock_system_status, mocker: MockerFixture) -> WebApiServer:
        """Fixture for a WebApiServer instance."""
        # Mock uvicorn.run to prevent the server from actually starting
        mocker.patch("uvicorn.run")
        return WebApiServer(mock_system_status)

    @pytest.fixture
    def test_client(self, web_api_server: WebApiServer) -> TestClient:
        """Fixture for a TestClient to test the API endpoints."""
        return TestClient(web_api_server._app)

    def test_init_default_values(self, mock_system_status) -> None:
        """Test initialization with default values."""
        # Act
        server = WebApiServer(mock_system_status)

        # Assert
        assert not server.has_commands()

    def test_has_commands_and_receive_command(
        self, web_api_server: WebApiServer, test_client: TestClient
    ) -> None:
        """Test has_commands and receive_command methods using public API."""
        # Arrange - initially there are no commands
        assert not web_api_server.has_commands()

        # Act - send a command through the API
        response = test_client.post("/api/pause")

        # Assert - check response and command existence
        assert response.status_code == 200
        assert web_api_server.has_commands()

        # Verify command content
        command = web_api_server.receive_command()
        assert command is ControlCommands.PAUSE

        # Verify queue is empty after retrieving the command
        assert not web_api_server.has_commands()

    def test_receive_command_when_queue_is_empty(
        self, web_api_server: WebApiServer
    ) -> None:
        """Test receive_command raises Empty when queue is empty."""
        with pytest.raises(Empty):
            web_api_server.receive_command()

    def test_get_status_endpoint_returns_current_status(
        self, test_client: TestClient, mock_system_status
    ) -> None:
        """Test GET /api/status endpoint returns the current system status."""
        # Arrange - set the return value for system status
        mock_system_status.get_current_status.return_value = SystemStatus.ACTIVE

        # Act - send request to status API
        response = test_client.get("/api/status")

        # Assert - verify response
        assert response.status_code == 200
        assert response.json() == {"status": SystemStatus.ACTIVE.value}
        mock_system_status.get_current_status.assert_called_once()

    def test_get_status_endpoint_handles_errors(
        self, test_client: TestClient, mock_system_status
    ) -> None:
        """Test GET /api/status endpoint returns an error response when status
        retrieval fails."""
        # Arrange - setup error condition
        mock_system_status.get_current_status.side_effect = Exception("Test error")

        # Act - send request to status API
        response = test_client.get("/api/status")

        # Assert - verify error response
        assert response.status_code == 500
        assert response.json() == ERROR_INTERNAL_SERVER

    def test_post_pause_endpoint_queues_pause_command(
        self, test_client: TestClient, web_api_server: WebApiServer
    ) -> None:
        """Test POST /api/pause endpoint successfully queues a PAUSE
        command."""
        # Act - send request to API endpoint
        response = test_client.post("/api/pause")

        # Assert - verify response and command queue state
        assert response.status_code == 200
        assert response.json() == RESULT_OK
        assert web_api_server.has_commands()
        assert web_api_server.receive_command() is ControlCommands.PAUSE

    def test_post_resume_endpoint_queues_resume_command(
        self, test_client: TestClient, web_api_server: WebApiServer
    ) -> None:
        """Test POST /api/resume endpoint successfully queues a RESUME
        command."""
        # Act - send request to API endpoint
        response = test_client.post("/api/resume")

        # Assert - verify response and command queue state
        assert response.status_code == 200
        assert response.json() == RESULT_OK
        assert web_api_server.has_commands()
        assert web_api_server.receive_command() is ControlCommands.RESUME

    def test_post_shutdown_endpoint_queues_shutdown_command(
        self, test_client: TestClient, web_api_server: WebApiServer
    ) -> None:
        """Test POST /api/shutdown endpoint successfully queues a SHUTDOWN
        command."""
        # Act - send request to API endpoint
        response = test_client.post("/api/shutdown")

        # Assert - verify response and command queue state
        assert response.status_code == 200
        assert response.json() == RESULT_OK
        assert web_api_server.has_commands()
        assert web_api_server.receive_command() is ControlCommands.SHUTDOWN

    def test_post_save_state_endpoint_queues_save_state_command(
        self, test_client: TestClient, web_api_server: WebApiServer
    ) -> None:
        """Test POST /api/save-state endpoint successfully queues a SAVE_state
        command."""
        # Act - send request to API endpoint
        response = test_client.post("/api/save-state")

        # Assert - verify response and command queue state
        assert response.status_code == 200
        assert response.json() == RESULT_OK
        assert web_api_server.has_commands()
        assert web_api_server.receive_command() is ControlCommands.SAVE_STATE

    def test_command_queue_overflow_returns_error(
        self, test_client: TestClient, web_api_server: WebApiServer
    ) -> None:
        """Test that attempting to add commands beyond queue capacity returns
        an error."""
        # Arrange - send first command (should succeed)
        response1 = test_client.post("/api/pause")
        assert response1.status_code == 200

        # Act - send second command (should fail due to queue being full)
        response2 = test_client.post("/api/resume")

        # Assert - verify error response
        assert response2.status_code == 503
        assert "full" in response2.json().get("error", "")

        # Verify first command is still in queue
        assert web_api_server.has_commands()
        command = web_api_server.receive_command()
        assert command is ControlCommands.PAUSE

        # Verify queue is now empty
        assert not web_api_server.has_commands()

    def test_command_sequence_processes_in_order(self, mocker: MockerFixture) -> None:
        """Test that commands are processed in the order they were queued."""
        # Arrange - create server with larger queue size
        status_provider = mocker.Mock(spec=SystemStatusProvider)
        server = WebApiServer(status_provider, max_queue_size=3)
        client = TestClient(server._app)

        # Act - send multiple commands in sequence
        client.post("/api/pause")
        client.post("/api/resume")
        client.post("/api/shutdown")

        # Assert - verify commands are retrieved in the same order
        assert server.has_commands()
        assert server.receive_command() is ControlCommands.PAUSE

        assert server.has_commands()
        assert server.receive_command() is ControlCommands.RESUME

        assert server.has_commands()
        assert server.receive_command() is ControlCommands.SHUTDOWN

        # Verify all commands have been processed
        assert not server.has_commands()

    def test_invalid_endpoint_returns_404(self, test_client: TestClient) -> None:
        """Test that requesting an invalid endpoint returns a 404 error."""
        # Act
        response = test_client.get("/api/invalid-endpoint")

        # Assert
        assert response.status_code == 404
        assert response.json() == ERROR_INVALID_ENDPOINT

    def test_invalid_method_returns_405(self, test_client: TestClient) -> None:
        """Test that using an invalid HTTP method returns a 405 error."""
        # Act - send GET request to a POST endpoint
        response = test_client.get("/api/pause")

        # Assert
        assert response.status_code == 405
        assert response.json() == ERROR_INVALID_METHOD


class TestWebApiClient:
    """Test suite for WebApiClient class."""

    @pytest.fixture
    def mock_client(self, mocker: MockerFixture):
        """Fixture providing a mock httpx.Client."""
        mock_client = mocker.Mock()
        mocker.patch("httpx.Client", return_value=mock_client)
        return mock_client

    @pytest.fixture
    def client(self, mock_client) -> WebApiClient:
        """Fixture providing a WebApiClient instance with mocked
        httpx.Client."""
        return WebApiClient("localhost", 8391)

    def test_init(self, client: WebApiClient):
        """Test WebApiClient initialization."""
        assert client.host == "localhost"
        assert client.port == 8391

    def test_get_status_success(self, client: WebApiClient, mock_client):
        """Test get_status method with successful response."""
        mock_response = mock_client.get.return_value
        mock_response.text = json.dumps({"status": SystemStatus.ACTIVE.value})

        result = client.get_status()

        assert result == SystemStatus.ACTIVE
        mock_client.get.assert_called_once_with("http://localhost:8391/api/status")

    def test_get_status_request_error(self, client: WebApiClient, mock_client):
        """Test get_status method with RequestError."""
        mock_client.get.side_effect = httpx.RequestError("Connection failed")

        result = client.get_status()

        assert result == SystemStatus.OFFLINE

    def test_get_status_http_error(self, client: WebApiClient, mock_client, mocker):
        """Test get_status method with HTTPStatusError."""
        mock_response = mock_client.get.return_value
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Server error", request=mocker.Mock(), response=mock_response
        )

        result = client.get_status()

        assert result == SystemStatus.OFFLINE

    def test_pause_success(self, client: WebApiClient, mock_client):
        """Test pause method with successful response."""
        mock_response = mock_client.post.return_value
        mock_response.text = json.dumps({"result": "ok"})

        result = client.pause()

        assert result == "ok"
        mock_client.post.assert_called_once_with("http://localhost:8391/api/pause")

    def test_pause_request_error(self, client: WebApiClient, mock_client):
        """Test pause method with RequestError."""
        mock_client.post.side_effect = httpx.RequestError("Connection failed")

        result = client.pause()

        assert result is None

    def test_pause_http_error(
        self, client: WebApiClient, mock_client, mocker: MockerFixture
    ):
        """Test pause method with HTTPStatusError."""
        mock_response = mock_client.post.return_value
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Server error", request=mocker.Mock(), response=mock_response
        )

        result = client.pause()

        assert result is None

    def test_resume_success(self, client: WebApiClient, mock_client):
        """Test resume method with successful response."""
        mock_response = mock_client.post.return_value
        mock_response.text = json.dumps({"result": "ok"})

        result = client.resume()

        assert result == "ok"
        mock_client.post.assert_called_once_with("http://localhost:8391/api/resume")

    def test_resume_request_error(self, client: WebApiClient, mock_client):
        """Test resume method with RequestError."""
        mock_client.post.side_effect = httpx.RequestError("Connection failed")

        result = client.resume()

        assert result is None

    def test_resume_http_error(
        self, client: WebApiClient, mock_client, mocker: MockerFixture
    ):
        """Test resume method with HTTPStatusError."""
        mock_response = mock_client.post.return_value
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Server error", request=mocker.Mock(), response=mock_response
        )

        result = client.resume()

        assert result is None

    def test_save_state_success(self, client: WebApiClient, mock_client):
        """Test save_state method with successful response."""
        mock_response = mock_client.post.return_value
        mock_response.text = json.dumps({"result": "ok"})

        result = client.save_state()

        assert result == "ok"
        mock_client.post.assert_called_once_with("http://localhost:8391/api/save-state")

    def test_save_state_request_error(self, client: WebApiClient, mock_client):
        """Test save_state method with RequestError."""
        mock_client.post.side_effect = httpx.RequestError("Connection failed")

        result = client.save_state()

        assert result is None

    def test_save_state_http_error(
        self, client: WebApiClient, mock_client, mocker: MockerFixture
    ):
        """Test save_state method with HTTPStatusError."""
        mock_response = mock_client.post.return_value
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Server error", request=mocker.Mock(), response=mock_response
        )

        result = client.save_state()

        assert result is None

    def test_shutdown_success(self, client: WebApiClient, mock_client):
        """Test shutdown method with successful response."""
        mock_response = mock_client.post.return_value
        mock_response.text = json.dumps({"result": "ok"})

        result = client.shutdown()

        assert result == "ok"
        mock_client.post.assert_called_once_with("http://localhost:8391/api/shutdown")

    def test_shutdown_request_error(self, client: WebApiClient, mock_client):
        """Test shutdown method with RequestError."""
        mock_client.post.side_effect = httpx.RequestError("Connection failed")

        result = client.shutdown()

        assert result is None

    def test_shutdown_http_error(
        self, client: WebApiClient, mock_client, mocker: MockerFixture
    ):
        """Test shutdown method with HTTPStatusError."""
        mock_response = mock_client.post.return_value
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Server error", request=mocker.Mock(), response=mock_response
        )

        result = client.shutdown()

        assert result is None

    def test_close(self, client: WebApiClient, mock_client):
        """Test close method calls client.close()."""
        client.close()

        mock_client.close.assert_called_once()
