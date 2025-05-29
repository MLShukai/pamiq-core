"""Web API interface for system control.

This module provides a simple web API for controlling the system,
allowing external applications to pause, resume, shutdown the system and
save states.
"""

import json
import logging
import threading
from enum import Enum, auto
from queue import Full, Queue
from typing import Any

import httpx
import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from pamiq_core.utils.reflection import get_class_module_path

from .system_status import SystemStatus, SystemStatusProvider


class ControlCommands(Enum):
    """Enumerates the commands for the system control."""

    SHUTDOWN = auto()
    PAUSE = auto()
    RESUME = auto()
    SAVE_STATE = auto()


# Constants for commonly used responses
RESULT_OK = {"result": "ok"}
ERROR_QUEUE_FULL = {"error": "Command queue is full, try again later"}
ERROR_INVALID_ENDPOINT = {"error": "Invalid API endpoint"}
ERROR_INVALID_METHOD = {"error": "Invalid API method"}
ERROR_INTERNAL_SERVER = {"error": "Internal server error"}


class WebApiServer:
    """Web API Server for controlling the system.

    This class provides a simple Web API for controlling the thread
    controller, allowing external applications to pause, resume, and
    shutdown the system.

    API Endpoints:
        - GET /api/status
            - Returns the current system status.
            - Response: {"status": *status_name*}
            - Status codes: 200 OK, 500 Internal Server Error

        - POST /api/pause
            - Pauses the system.
            - Response: {"result": "ok"} or {"error": "Command queue is full, try again later"}
            - Status codes: 200 OK, 503 Service Unavailable

        - POST /api/resume
            - Resumes the system.
            - Response: {"result": "ok"} or {"error": "Command queue is full, try again later"}
            - Status codes: 200 OK, 503 Service Unavailable

        - POST /api/shutdown
            - Shuts down the system.
            - Response: {"result": "ok"} or {"error": "Command queue is full, try again later"}
            - Status codes: 200 OK, 503 Service Unavailable

        - POST /api/save-state:
            - Saves a state of the current system state.
            - Response: {"result": "ok"} or {"error": "Command queue is full, try again later"}
            - Status codes: 200 OK, 503 Service Unavailable

    Error Responses:
        - 404 Not Found: {"error": "Invalid API endpoint"}
        - 405 Method Not Allowed: {"error": "Invalid API method"}
        - 500 Internal Server Error: {"error": "Internal server error"}
    """

    DEFAULT_QUEUE_SIZE = 1  # Default maximum size of command queue

    def __init__(
        self,
        system_status: SystemStatusProvider,
        host: str = "localhost",
        port: int = 8391,
        max_queue_size: int = DEFAULT_QUEUE_SIZE,
    ) -> None:
        """Initialize the WebApiHandler.

        Args:
            system_status: Provider for system status information.
            host: Hostname to run the API server on.
            port: Port to run the API server on.
            max_queue_size: Maximum size of the command queue.
        """
        self._logger = logging.getLogger(get_class_module_path(self.__class__))
        self._system_status = system_status
        self._host = host
        self._port = port

        self._received_commands_queue: Queue[ControlCommands] = Queue(
            maxsize=max_queue_size
        )

        # Define routes for the Starlette app
        routes = [
            Route("/api/status", endpoint=self._get_status, methods=["GET"]),
            Route("/api/pause", endpoint=self._post_pause, methods=["POST"]),
            Route("/api/resume", endpoint=self._post_resume, methods=["POST"]),
            Route("/api/shutdown", endpoint=self._post_shutdown, methods=["POST"]),
            Route(
                "/api/save-state",
                endpoint=self._post_save_state,
                methods=["POST"],
            ),
        ]

        # Create Starlette app with routes and exception handlers
        self._app = Starlette(
            routes=routes,
            exception_handlers={
                404: self._error_404,
                405: self._error_405,
                500: self._error_500,
            },
        )

        # Setup background thread
        self._handler_thread = threading.Thread(
            target=self._run_server, daemon=True, name="webapi"
        )

    def run_in_background(self) -> None:
        """Run the API server in a background thread."""
        self._handler_thread.start()

    def has_commands(self) -> bool:
        """Check if there are commands in the queue.

        Returns:
            True if there are commands, False otherwise.
        """
        return not self._received_commands_queue.empty()

    def receive_command(self) -> ControlCommands:
        """Receive a command from the queue without blocking.

        Returns:
            The command from the queue.

        Raises:
            Empty: If there are no commands in the queue.
        """
        return self._received_commands_queue.get_nowait()

    def _run_server(self) -> None:
        """Run the API server."""
        try:
            self._logger.info(f"Starting API server on {self._host}:{self._port}")
            uvicorn.run(self._app, host=self._host, port=self._port, log_level="error")
        except Exception:
            self._logger.exception("Failed to start API server")
            raise

    async def _add_command(self, command: ControlCommands) -> JSONResponse:
        """Add a command to the queue with overflow handling.

        Args:
            command: The command to add to the queue.

        Returns:
            JSONResponse: Success or error response.
        """
        try:
            self._received_commands_queue.put_nowait(command)
            return JSONResponse(RESULT_OK)
        except Full:
            self._logger.error(f"Command queue is full, rejecting {command}")
            return JSONResponse(ERROR_QUEUE_FULL, status_code=503)

    async def _get_status(self, request: Request) -> JSONResponse:
        """Handle GET /api/status request.

        Args:
            request: The HTTP request.

        Returns:
            JSONResponse with the current system status.
        """
        try:
            status = self._system_status.get_current_status()
            self._logger.info(f"Status request: returning {status.status_name}")
            return JSONResponse({"status": status.value})
        except Exception as e:
            self._logger.exception("Error getting system status")
            return await self._error_500(request, e)

    async def _post_pause(self, request: Request) -> JSONResponse:
        """Handle POST /api/pause request.

        Args:
            request: The HTTP request.

        Returns:
            JSONResponse: Success or error response.
        """
        self._logger.info("Pause command received")
        return await self._add_command(ControlCommands.PAUSE)

    async def _post_resume(self, request: Request) -> JSONResponse:
        """Handle POST /api/resume request.

        Args:
            request: The HTTP request.

        Returns:
            JSONResponse: Success or error response.
        """
        self._logger.info("Resume command received")
        return await self._add_command(ControlCommands.RESUME)

    async def _post_shutdown(self, request: Request) -> JSONResponse:
        """Handle POST /api/shutdown request.

        Args:
            request: The HTTP request.

        Returns:
            JSONResponse: Success or error response.
        """
        self._logger.info("Shutdown command received")
        return await self._add_command(ControlCommands.SHUTDOWN)

    async def _post_save_state(self, request: Request) -> JSONResponse:
        """Handle POST /api/save-state request.

        Args:
            request: The HTTP request.

        Returns:
            JSONResponse: Success or error response.
        """
        self._logger.info("Save state command received")
        return await self._add_command(ControlCommands.SAVE_STATE)

    async def _error_404(self, request: Request, exc: Any) -> JSONResponse:
        """Handle 404 errors.

        Args:
            request: The request that caused the error.
            exc: The exception information.

        Returns:
            JSON error response.
        """
        path = request.url.path
        method = request.method
        self._logger.error(f"404: {method} {path} is invalid API endpoint")
        return JSONResponse(ERROR_INVALID_ENDPOINT, status_code=404)

    async def _error_405(self, request: Request, exc: Any) -> JSONResponse:
        """Handle 405 errors.

        Args:
            request: The request that caused the error.
            exc: The exception information.

        Returns:
            JSON error response.
        """
        path = request.url.path
        method = request.method
        self._logger.error(f"405: {method} {path} is invalid API method")
        return JSONResponse(ERROR_INVALID_METHOD, status_code=405)

    async def _error_500(self, request: Request, exc: Any) -> JSONResponse:
        """Handle 500 errors.

        Args:
            request: The request that caused the error.
            exc: The exception information.

        Returns:
            JSON error response.
        """
        path = request.url.path
        method = request.method
        error_msg = str(exc) if exc else "Unknown error"
        self._logger.error(f"500: {method} {path} caused an error: {error_msg}")
        return JSONResponse(ERROR_INTERNAL_SERVER, status_code=500)


class WebApiClient:
    """Client for PAMIQ Web API communication.

    Provides methods to interact with PAMIQ system via HTTP API.
    """

    def __init__(self, host: str, port: int) -> None:
        """Initialize Web API client.

        Args:
            host: API server host
            port: API server port
        """
        self.host = host
        self.port = port
        self._client = httpx.Client()

    @property
    def _base_url(self) -> str:
        """Get base URL for API requests."""
        return f"http://{self.host}:{self.port}/api"

    def get_status(self) -> SystemStatus:
        """Get system status.

        Returns:
            Status enum. If error is occurred, return offline status
        """
        try:
            response = self._client.get(f"{self._base_url}/status")
            response.raise_for_status()
            return SystemStatus(json.loads(response.text)["status"])
        except (httpx.RequestError, httpx.HTTPStatusError):
            return SystemStatus.OFFLINE

    def pause(self) -> str | None:
        """Pause the system.

        Returns:
            Result message or None if request failed
        """
        try:
            response = self._client.post(f"{self._base_url}/pause")
            response.raise_for_status()
            return json.loads(response.text)["result"]
        except (httpx.RequestError, httpx.HTTPStatusError):
            return None

    def resume(self) -> str | None:
        """Resume the system.

        Returns:
            Result message or None if request failed
        """
        try:
            response = self._client.post(f"{self._base_url}/resume")
            response.raise_for_status()
            return json.loads(response.text)["result"]
        except (httpx.RequestError, httpx.HTTPStatusError):
            return None

    def save_state(self) -> str | None:
        """Save system state.

        Returns:
            Result message or None if request failed
        """
        try:
            response = self._client.post(f"{self._base_url}/save-state")
            response.raise_for_status()
            return json.loads(response.text)["result"]
        except (httpx.RequestError, httpx.HTTPStatusError):
            return None

    def shutdown(self) -> str | None:
        """Shutdown the system.

        Returns:
            Result message or None if request failed
        """
        try:
            response = self._client.post(f"{self._base_url}/shutdown")
            response.raise_for_status()
            return json.loads(response.text)["result"]
        except (httpx.RequestError, httpx.HTTPStatusError):
            return None

    def close(self) -> None:
        """Close the HTTP client."""
        self._client.close()
