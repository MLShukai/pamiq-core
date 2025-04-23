import json
import re
import sys
from collections.abc import Generator

import httpx
import pytest
from pytest_mock import MockerFixture

from pamiq_core.console.cui import Console, main


class TestConsole:
    @pytest.fixture
    def mock_httpx(self, mocker: MockerFixture):
        # Mock httpx module imported in pamiq_core/console/cui.py
        mock_httpx = mocker.patch("pamiq_core.console.cui.httpx")
        # Mock GET response
        mock_response = mocker.Mock()
        mock_response.text = json.dumps({"status": "running"})
        mock_httpx.get.return_value = mock_response
        # Mock POST response
        api_response = mocker.Mock()
        api_response.text = json.dumps({"result": "success"})
        mock_httpx.post.return_value = api_response
        return mock_httpx

    @pytest.fixture
    def console(self, mock_httpx) -> Console:
        return Console(host="localhost", port=8391)

    def test_fetch_status_when_online(self, console: Console, mock_httpx) -> None:
        console.fetch_status()
        assert console.status == "running"

    def test_fetch_status_when_offline(self, console: Console, mock_httpx) -> None:
        mock_httpx.RequestError = httpx.RequestError
        mock_httpx.get.side_effect = httpx.RequestError("Test RequestError")
        console.fetch_status()
        assert console.status == "offline"

    def test_all_commands(self, console: Console) -> None:
        assert set(console.all_commands) == {
            "h",
            "help",
            "p",
            "pause",
            "r",
            "resume",
            "save",
            "shutdown",
            "q",
            "quit",
        }

    def test_run_command_when_online(
        self, mocker: MockerFixture, console: Console, mock_httpx
    ) -> None:
        mock_help = mocker.spy(console, "command_help")
        # test if connect WebAPI
        console.run_command("help")
        assert not console.status == "offline"
        mock_help.assert_called_once_with()

    def test_run_command_when_offline(
        self,
        console: Console,
        mock_httpx,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        # set to rise connection fails
        mock_httpx.RequestError = httpx.RequestError
        mock_httpx.get.side_effect = httpx.RequestError("Test RequestError")
        # test each command
        for command in console.all_commands:
            console.run_command(command)
            assert console.status == "offline"
            captured = capsys.readouterr()
            if command in [
                "pause",
                "p",
                "resume",
                "r",
                "save",
                "shutdown",
            ]:
                assert f'Command "{command}" not executed.' in captured.out
            else:
                assert f'Command "{command}" not executed.' not in captured.out

    def test_main_loop_with_quit(
        self,
        mocker: MockerFixture,
        console: Console,
        mock_httpx,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        mocker.patch(
            "pamiq_core.console.cui.prompt", side_effect=["quit", "other_strings"]
        )
        mock_run_command = mocker.spy(console, "run_command")
        console.main_loop()
        # Check if "quit" finishes CUI and "other_strings" as an invalid command.
        assert mock_run_command.call_count == 1
        captured = capsys.readouterr()
        assert "*** Unknown syntax: other_strings" not in captured.out

    def test_main_loop_with_available_command(
        self,
        mocker: MockerFixture,
        console: Console,
        mock_httpx,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        mocker.patch("pamiq_core.console.cui.prompt", side_effect=["help", "quit"])
        mock_run_command = mocker.spy(console, "run_command")
        console.main_loop()
        # Check if "help" runs as an available command and "quit" finishes CUI.
        assert mock_run_command.call_count == 2
        captured = capsys.readouterr()
        assert "*** Unknown syntax: other_strings" not in captured.out

    def test_main_loop_with_unknown_command(
        self,
        mocker: MockerFixture,
        console: Console,
        mock_httpx,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        mocker.patch(
            "pamiq_core.console.cui.prompt", side_effect=["other_strings", "quit"]
        )
        mock_run_command = mocker.spy(console, "run_command")
        # Check if "other_strings" as an invalid command and "quit" finishes CUI.
        console.main_loop()
        assert mock_run_command.call_count == 1
        captured = capsys.readouterr()
        assert "*** Unknown syntax: other_strings" in captured.out

    def test_command_help(
        self, console: Console, capsys: pytest.CaptureFixture[str]
    ) -> None:
        console.command_help()
        captured = capsys.readouterr()
        # Check if "help" explains all commands.
        captured_commands: list[str] = []
        for line in captured.out.split("\n"):
            # catch the format "cmd1/cmd2/... Explain"
            match = re.compile(r"^([\w/]+)").match(line)
            if match:
                cmds = match.group(1).split("/")
                captured_commands += cmds
        assert set(console.all_commands) == set(captured_commands)

    def test_command_h_as_alias(
        self, mocker: MockerFixture, console: Console, mock_httpx
    ) -> None:
        mock_help = mocker.spy(console, "command_help")
        console.command_h()
        mock_help.assert_called_once_with()

    def test_command_pause(
        self,
        console: Console,
        mock_httpx,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        mock_httpx.post.return_value.text = json.dumps({"result": "test command_pause"})
        console.command_pause()
        mock_httpx.post.assert_called_once_with("http://localhost:8391/api/pause")
        captured = capsys.readouterr()
        assert "test command_pause" in captured.out

    def test_command_p_as_alias(
        self, mocker: MockerFixture, console: Console, mock_httpx
    ) -> None:
        mock_pause = mocker.spy(console, "command_pause")
        console.command_p()
        mock_pause.assert_called_once_with()

    def test_command_resume(
        self,
        console: Console,
        mock_httpx,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        mock_httpx.post.return_value.text = json.dumps(
            {"result": "test command_resume"}
        )
        console.command_resume()
        mock_httpx.post.assert_called_once_with("http://localhost:8391/api/resume")
        captured = capsys.readouterr()
        assert "test command_resume" in captured.out

    def test_command_r_as_alias(
        self, mocker: MockerFixture, console: Console, mock_httpx
    ) -> None:
        mock_resume = mocker.spy(console, "command_resume")
        console.command_r()
        mock_resume.assert_called_once_with()

    @pytest.mark.parametrize("users_answer", ["y", "yes"])
    def test_command_shutdown_yes(
        self,
        console: Console,
        mock_httpx,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
        users_answer: str,
    ) -> None:
        monkeypatch.setattr(
            "builtins.input", lambda prompt: users_answer
        )  # Mock input() to return <users_answer>.
        mock_httpx.post.return_value.text = json.dumps(
            {"result": "test command_shutdown"}
        )
        result = console.command_shutdown()
        mock_httpx.post.assert_called_once_with("http://localhost:8391/api/shutdown")
        captured = capsys.readouterr()
        assert "test command_shutdown" in captured.out
        assert result is True

    @pytest.mark.parametrize("users_answer", ["n", "N", "other_strings"])
    def test_command_shutdown_no(
        self,
        console: Console,
        mock_httpx,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
        users_answer: str,
    ) -> None:
        monkeypatch.setattr(
            "builtins.input", lambda prompt: users_answer
        )  # Mock input() to return <users_answer>.
        mock_httpx.post.return_value.text = json.dumps(
            {"result": "test command_shutdown"}
        )
        result = console.command_shutdown()
        captured = capsys.readouterr()
        assert "Shutdown cancelled" in captured.out
        assert result is False

    def test_command_quit(self, console: Console) -> None:
        result = console.command_quit()
        assert result is True

    def test_command_q_as_alias(self, mocker: MockerFixture, console: Console) -> None:
        mock_quit = mocker.spy(console, "command_quit")
        console.command_q()
        mock_quit.assert_called_once_with()

    def test_command_save(
        self,
        console: Console,
        mock_httpx,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        mock_httpx.post.return_value.text = json.dumps({"result": "test command_save"})
        console.command_save()
        mock_httpx.post.assert_called_once_with("http://localhost:8391/api/save-state")
        captured = capsys.readouterr()
        assert "test command_save" in captured.out


def test_main(mocker: MockerFixture) -> None:
    mock_console_class = mocker.patch(
        "sys.argv", ["consoletest", "--host", "test-host.com", "--port", "1938"]
    )
    mock_console_class = mocker.patch("pamiq_core.console.cui.Console")
    mock_console = mocker.Mock(Console)
    mock_console_class.return_value = mock_console
    main()
    mock_console_class.assert_called_once_with("test-host.com", 1938)
    mock_console.main_loop.assert_called_once_with()
