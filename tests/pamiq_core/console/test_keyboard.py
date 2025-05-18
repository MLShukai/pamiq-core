import json

import httpx
import pytest
from pytest_mock import MockerFixture

try:
    from pynput import keyboard
except ImportError:
    pytest.skip("Can not import 'pynput' module.", allow_module_level=True)

from pamiq_core.console.keyboard import KeyboardController


class TestKeyboardController:
    @pytest.fixture
    def controller(self) -> KeyboardController:
        return KeyboardController("localhost", 8391, "alt+shift+p", "alt+shift+r", None)

    def test_init(self):
        controller = KeyboardController("test.com", 1234, "ctrl+p", "ctrl+r", None)
        assert controller.host == "test.com"
        assert controller.port == 1234

    def test_send_command_success(
        self, controller: KeyboardController, mocker: MockerFixture, capsys
    ):
        mock_response = mocker.Mock()
        mock_response.text = json.dumps({"result": "ok"})
        mocker.patch("httpx.post", return_value=mock_response)

        controller.send_command("pause")

        captured = capsys.readouterr()
        assert "pause: ok" in captured.out

    def test_send_command_connect_error(
        self, controller: KeyboardController, mocker: MockerFixture, capsys
    ):
        mocker.patch("httpx.post", side_effect=httpx.ConnectError("test"))

        controller.send_command("pause")

        captured = capsys.readouterr()
        assert "pause: Connection failed, continuing..." in captured.out

    def test_send_command_503_error(
        self, controller: KeyboardController, mocker: MockerFixture, capsys
    ):
        mock_response = mocker.Mock()
        mock_response.status_code = 503
        mocker.patch(
            "httpx.post",
            side_effect=httpx.HTTPStatusError(
                "test", request=mocker.Mock(), response=mock_response
            ),
        )

        controller.send_command("pause")

        captured = capsys.readouterr()
        assert "pause: Service unavailable, continuing..." in captured.out

    def test_get_key_name_special_key(self):
        alt_key = keyboard.Key.alt
        assert KeyboardController.get_key_name(alt_key) == "alt"

    def test_get_key_name_char_key(self):
        p_key = keyboard.KeyCode.from_char("P")
        assert KeyboardController.get_key_name(p_key) == "p"

    def test_get_key_name_no_char(self, mocker: MockerFixture):
        mock_keycode = mocker.Mock(spec=keyboard.KeyCode)
        mock_keycode.char = None
        assert KeyboardController.get_key_name(mock_keycode) is None

    def test_on_press_pause_combination_and_release(
        self, controller: KeyboardController, mocker: MockerFixture
    ):
        mock_send = mocker.patch.object(controller, "send_command")

        controller.on_press(keyboard.Key.alt)
        controller.on_press(keyboard.Key.shift)
        controller.on_press(keyboard.KeyCode.from_char("p"))

        mock_send.assert_called_once_with("pause")

        mock_send.reset_mock()
        controller.on_release(keyboard.KeyCode.from_char("p"))
        mock_send.assert_not_called()
        controller.on_press(keyboard.KeyCode.from_char("r"))
        mock_send.assert_called_once_with("resume")
