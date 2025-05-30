import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import Literal

from pynput import keyboard

from .web_api import WebApiClient


class KeyboardController:
    """Keyboard controller for PAMIQ system control."""

    def __init__(
        self,
        host: str,
        port: int,
        pause_keys: str,
        resume_keys: str,
        quit_keys: str | None,
    ) -> None:
        """Initialize keyboard controller.

        Args:
            host: API server host
            port: API server port
            pause_keys: Key combination for pause command (e.g., "alt+shift+p")
            resume_keys: Key combination for resume command (e.g., "alt+shift+r")
            quit_keys: Key combination for quit command (e.g., "alt+shift+q")
        """
        self._client = WebApiClient(host, port)
        self._pause_keys = self._parse_key_combination(pause_keys)
        self._resume_keys = self._parse_key_combination(resume_keys)
        self._quit_keys = quit_keys
        if quit_keys:
            self._quit_keys = self._parse_key_combination(quit_keys)
        self._current_keys: set[str] = set()

        self._executor = ThreadPoolExecutor()

    def _parse_key_combination(self, keys_str: str) -> set[str]:
        """Parse key combination string to key name set.

        Args:
            keys_str: Key combination string (e.g., "alt+shift+p")

        Returns:
            Set of key names in lowercase
        """
        return set(keys_str.lower().split("+"))

    @staticmethod
    def get_key_name(key: keyboard.Key | keyboard.KeyCode) -> str | None:
        """Convert key object to lowercase string name, and erase LR
        difference.

        Args:
            key: Key object from pynput

        Returns:
            Lowercase key name or None if not determinable
        """
        if isinstance(key, keyboard.Key):
            return key.name.lower().split("_", 1)[0]  # lower, and erase LR difference.
        else:
            if key.char:
                return key.char.lower()

    def _send_command(self, command: Literal["pause", "resume"]) -> None:
        """Send command with client and print result message."""
        match command:
            case "pause":
                success = self._client.pause()
            case "resume":
                success = self._client.resume()
        if success:
            print(f"Success to send {command} command.")
        else:
            print(f"Failed to send {command} command.")

    def on_press(self, key: keyboard.Key | keyboard.KeyCode | None) -> None:
        """Handle key press event.

        Args:
            key: Pressed key
        """
        if key is None:
            return
        name = self.get_key_name(key)
        if not name:
            return
        print("key press:", name)
        self._current_keys.add(name)

        if self._current_keys == self._pause_keys:
            self._executor.submit(self._send_command, "pause")
        elif self._current_keys == self._resume_keys:
            self._executor.submit(self._send_command, "resume")
        elif self._current_keys == self._quit_keys:
            self._listener.stop()

    def on_release(self, key: keyboard.Key | keyboard.KeyCode | None) -> None:
        """Handle key release event.

        Args:
            key: Released key
        """
        if key is None:
            return
        name = self.get_key_name(key)
        if not name:
            return
        self._current_keys.discard(name)

    def run(self) -> None:
        """Start keyboard listener."""
        print("Keyboard controller started.")
        print(f"Pause: {'+'.join(sorted(self._pause_keys))}")
        print(f"Resume: {'+'.join(sorted(self._resume_keys))}")
        if self._quit_keys:
            print(f"Quit: {'+'.join(sorted(self._quit_keys))}")
        print("Press Ctrl+C to exit.")

        self._listener = keyboard.Listener(
            on_press=self.on_press, on_release=self.on_release
        )

        with self._listener as listener:
            listener.join()

        self._executor.shutdown()


def main() -> None:
    """Entry point of pamiq-kbctl."""
    default_pause_key = "alt+shift+p"
    default_resume_key = "alt+shift+r"
    default_quit_key = None
    if sys.platform == "darwin":  # macOS setting
        default_pause_key = "alt+shift+∏"  # "∏" is opt + p
        default_resume_key = "alt+shift+‰"  # "‰" is opt + r
    elif sys.platform == "win32":
        default_quit_key = "alt+shift+q"

    parser = argparse.ArgumentParser(description="PAMIQ keyboard controller")
    parser.add_argument("--host", default="localhost", help="API server host")
    parser.add_argument("--port", default=8391, type=int, help="API server port")
    parser.add_argument(
        "--pause-key",
        default=default_pause_key,
        help="Key combination for pause. Default is 'alt+shift+p' ('alt' is 'option' on macOS)",
    )
    parser.add_argument(
        "--resume-key",
        default=default_resume_key,
        help="Key combination for resume. Default is 'alt+shift+r' ('alt' is 'option' on macOS)",
    )

    parser.add_argument(
        "--quit-key",
        default=default_quit_key,
        help="Key combination for quit keyboard controller. Default is None (but 'alt+shift+q'on Windows)",
    )

    args = parser.parse_args()

    controller = KeyboardController(
        host=args.host,
        port=args.port,
        pause_keys=args.pause_key,
        resume_keys=args.resume_key,
        quit_keys=args.quit_key,
    )

    try:
        controller.run()
    except KeyboardInterrupt:
        print("\nKeyboard controller stopped.")


if __name__ == "__main__":
    main()
