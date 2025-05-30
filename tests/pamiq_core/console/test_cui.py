import re

import pytest
from pytest_mock import MockerFixture

from pamiq_core.console.cui import Console, main
from pamiq_core.console.system_status import SystemStatus
from pamiq_core.console.web_api import WebApiClient


class TestConsole:
    @pytest.fixture
    def mock_web_api_client(self, mocker: MockerFixture) -> WebApiClient:
        """Mock WebApiClient for testing."""
        mock_client = mocker.Mock(spec=WebApiClient)
        mock_client.get_status.return_value = SystemStatus.ACTIVE
        mock_client.pause.return_value = "ok"
        mock_client.resume.return_value = "ok"
        mock_client.save_state.return_value = "ok"
        mock_client.shutdown.return_value = "ok"
        return mock_client

    @pytest.fixture
    def console(self, mocker: MockerFixture, mock_web_api_client) -> Console:
        """Create Console instance with mocked WebApiClient."""
        mocker.patch(
            "pamiq_core.console.cui.WebApiClient", return_value=mock_web_api_client
        )
        return Console(host="localhost", port=8391)

    def test_fetch_status_when_online(
        self, console: Console, mock_web_api_client
    ) -> None:
        """Test fetch_status when system is online."""
        mock_web_api_client.get_status.return_value = SystemStatus.ACTIVE
        console.fetch_status()
        assert console.status is SystemStatus.ACTIVE
        mock_web_api_client.get_status.assert_called_once()

    def test_fetch_status_when_offline(
        self, console: Console, mock_web_api_client
    ) -> None:
        """Test fetch_status when system is offline."""
        mock_web_api_client.get_status.return_value = SystemStatus.OFFLINE
        console.fetch_status()
        assert console.status is SystemStatus.OFFLINE

    def test_all_commands(self, console: Console) -> None:
        """Test that all expected commands are available."""
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
        self, mocker: MockerFixture, console: Console, mock_web_api_client
    ) -> None:
        """Test run_command when system is online."""
        mock_help = mocker.spy(console, "command_help")
        mock_web_api_client.get_status.return_value = SystemStatus.ACTIVE
        console.run_command("help")
        assert console.status is SystemStatus.ACTIVE
        mock_help.assert_called_once_with()

    def test_run_command_when_offline(
        self,
        console: Console,
        mock_web_api_client,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Test run_command when system is offline."""
        mock_web_api_client.get_status.return_value = SystemStatus.OFFLINE

        # Test each command
        for command in console.all_commands:
            console.run_command(command)
            assert console.status is SystemStatus.OFFLINE
            captured = capsys.readouterr()
            if command in ["pause", "p", "resume", "r", "save", "shutdown"]:
                assert f'Command "{command}" not executed.' in captured.out
            else:
                assert f'Command "{command}" not executed.' not in captured.out

    def test_main_loop_with_quit(
        self,
        mocker: MockerFixture,
        console: Console,
        mock_web_api_client,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Test main_loop exits with quit command."""
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
        mock_web_api_client,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Test main_loop with available command."""
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
        mock_web_api_client,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Test main_loop with unknown command."""
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
        """Test command_help displays all commands."""
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
        self, mocker: MockerFixture, console: Console, mock_web_api_client
    ) -> None:
        """Test command_h is alias for command_help."""
        mock_help = mocker.spy(console, "command_help")
        console.command_h()
        mock_help.assert_called_once_with()

    def test_command_pause_success(
        self,
        console: Console,
        mock_web_api_client,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Test command_pause when successful."""
        mock_web_api_client.pause.return_value = "test command_pause"
        console.command_pause()
        mock_web_api_client.pause.assert_called_once()
        captured = capsys.readouterr()
        assert "test command_pause" in captured.out

    def test_command_pause_failure(
        self,
        console: Console,
        mock_web_api_client,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Test command_pause when failed."""
        mock_web_api_client.pause.return_value = None
        console.command_pause()
        captured = capsys.readouterr()
        assert "Failed to pause..." in captured.out

    def test_command_p_as_alias(
        self, mocker: MockerFixture, console: Console, mock_web_api_client
    ) -> None:
        """Test command_p is alias for command_pause."""
        mock_pause = mocker.spy(console, "command_pause")
        console.command_p()
        mock_pause.assert_called_once_with()

    def test_command_resume_success(
        self,
        console: Console,
        mock_web_api_client,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Test command_resume when successful."""
        mock_web_api_client.resume.return_value = "test command_resume"
        console.command_resume()
        mock_web_api_client.resume.assert_called_once()
        captured = capsys.readouterr()
        assert "test command_resume" in captured.out

    def test_command_resume_failure(
        self,
        console: Console,
        mock_web_api_client,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Test command_resume when failed."""
        mock_web_api_client.resume.return_value = None
        console.command_resume()
        captured = capsys.readouterr()
        assert "Failed to resume..." in captured.out

    def test_command_r_as_alias(
        self, mocker: MockerFixture, console: Console, mock_web_api_client
    ) -> None:
        """Test command_r is alias for command_resume."""
        mock_resume = mocker.spy(console, "command_resume")
        console.command_r()
        mock_resume.assert_called_once_with()

    @pytest.mark.parametrize("users_answer", ["y", "yes"])
    def test_command_shutdown_yes_success(
        self,
        console: Console,
        mock_web_api_client,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
        users_answer: str,
    ) -> None:
        """Test command_shutdown with yes answer and successful response."""
        monkeypatch.setattr("builtins.input", lambda prompt: users_answer)
        mock_web_api_client.shutdown.return_value = "test command_shutdown"
        result = console.command_shutdown()
        mock_web_api_client.shutdown.assert_called_once()
        assert result is True

    @pytest.mark.parametrize("users_answer", ["y", "yes"])
    def test_command_shutdown_yes_failure(
        self,
        console: Console,
        mock_web_api_client,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
        users_answer: str,
    ) -> None:
        """Test command_shutdown with yes answer but failed response."""
        monkeypatch.setattr("builtins.input", lambda prompt: users_answer)
        mock_web_api_client.shutdown.return_value = None
        result = console.command_shutdown()
        captured = capsys.readouterr()
        assert "Failed to shutdown..." in captured.out
        assert "Shutdown cancelled." in captured.out
        assert result is False

    @pytest.mark.parametrize("users_answer", ["n", "N", "other_strings"])
    def test_command_shutdown_no(
        self,
        console: Console,
        mock_web_api_client,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
        users_answer: str,
    ) -> None:
        """Test command_shutdown with no answer."""
        monkeypatch.setattr("builtins.input", lambda prompt: users_answer)
        result = console.command_shutdown()
        mock_web_api_client.shutdown.assert_not_called()
        captured = capsys.readouterr()
        assert "Shutdown cancelled." in captured.out
        assert result is False

    def test_command_quit(self, console: Console) -> None:
        """Test command_quit returns True."""
        result = console.command_quit()
        assert result is True

    def test_command_q_as_alias(self, mocker: MockerFixture, console: Console) -> None:
        """Test command_q is alias for command_quit."""
        mock_quit = mocker.spy(console, "command_quit")
        console.command_q()
        mock_quit.assert_called_once_with()

    def test_command_save_success(
        self,
        console: Console,
        mock_web_api_client,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Test command_save when successful."""
        mock_web_api_client.save_state.return_value = "test command_save"
        console.command_save()
        mock_web_api_client.save_state.assert_called_once()
        captured = capsys.readouterr()
        assert "test command_save" in captured.out

    def test_command_save_failure(
        self,
        console: Console,
        mock_web_api_client,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Test command_save when failed."""
        mock_web_api_client.save_state.return_value = None
        console.command_save()
        captured = capsys.readouterr()
        assert "Failed to save state..." in captured.out


def test_main(mocker: MockerFixture) -> None:
    """Test main function with command line arguments."""
    mocker.patch(
        "sys.argv", ["consoletest", "--host", "test-host.com", "--port", "1938"]
    )
    mock_console_class = mocker.patch("pamiq_core.console.cui.Console")
    mock_console = mocker.Mock(Console)
    mock_console_class.return_value = mock_console
    main()
    mock_console_class.assert_called_once_with("test-host.com", 1938)
    mock_console.main_loop.assert_called_once_with()
