import time

import pytest
from pytest_mock import MockerFixture

try:
    from pynput import keyboard
except ImportError:
    pytest.skip("Can not import 'pynput' module.", allow_module_level=True)

from pamiq_core.console.keyboard import KeyboardController
from pamiq_core.console.web_api import WebApiClient


class TestKeyboardController:
    @pytest.fixture
    def mock_web_api_client(self, mocker: MockerFixture) -> WebApiClient:
        """Mock WebApiClient for testing."""
        mock_client = mocker.Mock(spec=WebApiClient)
        mock_client.pause.return_value = "ok"
        mock_client.resume.return_value = "ok"
        return mock_client

    @pytest.fixture
    def controller(
        self, mocker: MockerFixture, mock_web_api_client
    ) -> KeyboardController:
        """Create KeyboardController with mocked WebApiClient."""
        mocker.patch(
            "pamiq_core.console.keyboard.WebApiClient", return_value=mock_web_api_client
        )
        return KeyboardController("localhost", 8391, "alt+shift+p", "alt+shift+r", None)

    def test_init(self, mocker: MockerFixture):
        """Test KeyboardController initialization."""
        mock_client = mocker.Mock(spec=WebApiClient)
        mock_web_api_client_class = mocker.patch(
            "pamiq_core.console.keyboard.WebApiClient", return_value=mock_client
        )

        KeyboardController("test.com", 1234, "ctrl+p", "ctrl+r", None)

        mock_web_api_client_class.assert_called_once_with("test.com", 1234)

    def test_get_key_name_special_key(self):
        """Test get_key_name with special keys."""
        alt_key = keyboard.Key.alt
        assert KeyboardController.get_key_name(alt_key) == "alt"

    def test_get_key_name_char_key(self):
        """Test get_key_name with character keys."""
        p_key = keyboard.KeyCode.from_char("P")
        assert KeyboardController.get_key_name(p_key) == "p"

    def test_get_key_name_no_char(self, mocker: MockerFixture):
        """Test get_key_name with key that has no char."""
        mock_keycode = mocker.Mock(spec=keyboard.KeyCode)
        mock_keycode.char = None
        assert KeyboardController.get_key_name(mock_keycode) is None

    def test_on_press_pause_combination(
        self, controller: KeyboardController, mock_web_api_client, capsys
    ):
        """Test pause key combination triggers pause command."""
        controller.on_press(keyboard.Key.alt)
        controller.on_press(keyboard.Key.shift)
        controller.on_press(keyboard.KeyCode.from_char("p"))

        time.sleep(0.001)
        mock_web_api_client.pause.assert_called_once()

    def test_on_press_resume_combination(
        self, controller: KeyboardController, mock_web_api_client, capsys
    ):
        """Test resume key combination triggers resume command."""
        controller.on_press(keyboard.Key.alt)
        controller.on_press(keyboard.Key.shift)
        controller.on_press(keyboard.KeyCode.from_char("r"))

        time.sleep(0.001)
        mock_web_api_client.resume.assert_called_once()

    def test_on_press_quit_combination(
        self, controller: KeyboardController, mocker: MockerFixture
    ):
        """Test quit key combination stops listener."""
        # Create controller with quit keys
        mock_client = mocker.Mock(spec=WebApiClient)
        mocker.patch(
            "pamiq_core.console.keyboard.WebApiClient", return_value=mock_client
        )
        controller = KeyboardController(
            "localhost", 8391, "alt+shift+p", "alt+shift+r", "alt+shift+q"
        )

        # Mock the listener
        mock_listener = mocker.Mock()
        controller._listener = mock_listener

        controller.on_press(keyboard.Key.alt)
        controller.on_press(keyboard.Key.shift)
        controller.on_press(keyboard.KeyCode.from_char("q"))

        mock_listener.stop.assert_called_once()

    def test_on_press_partial_combination(
        self, controller: KeyboardController, mock_web_api_client
    ):
        """Test partial key combination doesn't trigger commands."""
        controller.on_press(keyboard.Key.alt)
        controller.on_press(keyboard.KeyCode.from_char("p"))  # Missing shift

        mock_web_api_client.pause.assert_not_called()
        mock_web_api_client.resume.assert_not_called()

    def test_on_press_release_and_press_again(
        self, controller: KeyboardController, mock_web_api_client
    ):
        """Test key release and press different combination."""
        # Press pause combination
        controller.on_press(keyboard.Key.alt)
        controller.on_press(keyboard.Key.shift)
        controller.on_press(keyboard.KeyCode.from_char("p"))
        time.sleep(0.001)
        mock_web_api_client.pause.assert_called_once()

        # Release one key and press resume
        mock_web_api_client.reset_mock()
        controller.on_release(keyboard.KeyCode.from_char("p"))
        controller.on_press(keyboard.KeyCode.from_char("r"))
        time.sleep(0.001)
        mock_web_api_client.resume.assert_called_once()

    def test_on_press_none_key(
        self, controller: KeyboardController, mock_web_api_client
    ):
        """Test on_press with None key."""
        controller.on_press(None)
        mock_web_api_client.pause.assert_not_called()
        mock_web_api_client.resume.assert_not_called()

    def test_on_release_none_key(self, controller: KeyboardController):
        """Test on_release with None key."""
        # Should not raise exception
        controller.on_release(None)
