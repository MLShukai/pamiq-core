import json
import sys
from unittest.mock import MagicMock, patch

import pytest
import requests

from pamiq_core.console.cui import Console, main


class TestConsole:
    @pytest.fixture
    def mock_requests(self):
        # Mock request module imported in pamiq_core/console/cui.py
        with patch("pamiq_core.console.cui.requests") as mock_req:
            # Mock GET response
            mock_response = MagicMock()
            mock_response.text = json.dumps({"status": "running"})
            mock_req.get.return_value = mock_response
            # Mock POST response
            api_response = MagicMock()
            api_response.text = json.dumps({"result": "success"})
            mock_req.post.return_value = api_response
            yield mock_req

    @pytest.fixture
    def console(self, mock_requests):
        return Console(host="localhost", port=8391)

    def test_initial_prompt(self, mock_requests):
        mock_requests.get.return_value.text = json.dumps({"status": "running"})
        console = Console("localhost", 8391)
        assert console.prompt == "ami (running) > "

    def test_initial_prompt_when_offline(self, mock_requests):
        mock_requests.exceptions = requests.exceptions
        mock_requests.get.side_effect = requests.exceptions.ConnectionError()
        console = Console("localhost", 8391)
        assert console.prompt == "ami (offline) > "

    def test_help_command(self, console, capsys):
        console.do_help("")
        captured = capsys.readouterr()
        assert "h/help" in captured.out
        assert "p/pause" in captured.out
        assert "r/resume" in captured.out
        assert "c/ckpt" in captured.out
        assert "s/shutdown" in captured.out
        assert "q/quit" in captured.out

    def test_do_pause(self, console, mock_requests, capsys):
        mock_requests.post.return_value.text = json.dumps({"result": "test do_pause"})
        console.do_pause("")
        mock_requests.post.assert_called_once_with("http://localhost:8391/api/pause")
        captured = capsys.readouterr()
        assert "test do_pause" in captured.out

    def test_do_p_as_alias(self, console, mock_requests):
        with patch.object(console, "do_pause") as mock_pause:
            console.do_p("")
            mock_pause.assert_called_once_with("")

    def test_do_resume(self, console, mock_requests, capsys):
        mock_requests.post.return_value.text = json.dumps({"result": "test do_resume"})
        console.do_resume("")
        mock_requests.post.assert_called_once_with("http://localhost:8391/api/resume")
        captured = capsys.readouterr()
        assert "test do_resume" in captured.out

    def test_do_r_as_alias(self, console, mock_requests):
        with patch.object(console, "do_resume") as mock_resume:
            console.do_r("")
            mock_resume.assert_called_once_with("")

    def test_do_shutdown(self, console, mock_requests, capsys):
        mock_requests.post.return_value.text = json.dumps(
            {"result": "test do_shutdown"}
        )
        result = console.do_shutdown("")
        mock_requests.post.assert_called_once_with("http://localhost:8391/api/shutdown")
        captured = capsys.readouterr()
        assert "test do_shutdown" in captured.out
        assert result is True  # Return True if cmd loop.

    def test_do_s_as_alias(self, console, mock_requests):
        with patch.object(console, "do_shutdown") as mock_shutdown:
            console.do_s("")
            mock_shutdown.assert_called_once_with("")

    def test_do_quit(self, console):
        result = console.do_quit("")
        assert result is True  # Return True if cmd loop.

    def test_do_q_as_alias(self, console):
        with patch.object(console, "do_quit") as mock_quit:
            console.do_q("")
            mock_quit.assert_called_once_with("")

    def test_do_ckpt(self, console, mock_requests, capsys):
        mock_requests.post.return_value.text = json.dumps({"result": "test do_ckpt"})
        console.do_ckpt("")
        mock_requests.post.assert_called_once_with(
            "http://localhost:8391/api/save-state"
        )
        captured = capsys.readouterr()
        assert "test do_ckpt" in captured.out

    def test_do_c_as_alias(self, console, mock_requests):
        with patch.object(console, "do_ckpt") as mock_ckpt:
            console.do_c("")
            mock_ckpt.assert_called_once_with("")

    @pytest.mark.parametrize("exit_console", [True, False])
    def test_postcmd_updates_status(self, console, mock_requests, exit_console: bool):
        mock_requests.get.return_value.text = json.dumps({"status": "test postcmd"})
        result = console.postcmd(stop=exit_console, line="")
        assert console.prompt == "ami (test postcmd) > "
        assert result is exit_console  # Return True if cmd loop.


def test_main():
    with (
        patch("sys.argv", ["pamiq-core", "--host", "test_host.com", "--port", "1938"]),
        patch("pamiq_core.console.cui.Console") as mock_console_class,
    ):
        mock_console = MagicMock()
        mock_console_class.return_value = mock_console
        main()
        mock_console_class.assert_called_once_with("test_host.com", 1938)
        mock_console.cmdloop.assert_called_once_with()
