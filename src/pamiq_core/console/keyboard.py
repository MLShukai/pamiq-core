import argparse
import json
import sys

import httpx
from pynput import keyboard


class KeyboardController:
    """Keyboard controller for PAMIQ system control."""

    def __init__(self, host: str, port: int, pause_keys: str, resume_keys: str) -> None:
        """Initialize keyboard controller.

        Args:
            host: API server host
            port: API server port
            pause_keys: Key combination for pause command (e.g., "alt+shift+p")
            resume_keys: Key combination for resume command (e.g., "alt+shift+r")
        """
        self.host = host
        self.port = port
        self._pause_keys = self._parse_key_combination(pause_keys)
        self._resume_keys = self._parse_key_combination(resume_keys)
        self._current_keys: set[str] = set()

    def _parse_key_combination(self, keys_str: str) -> set[str]:
        """Parse key combination string to key name set.

        Args:
            keys_str: Key combination string (e.g., "alt+shift+p")

        Returns:
            Set of key names in lowercase
        """
        return set(keys_str.lower().split("+"))

    def send_command(self, endpoint: str) -> None:
        """Send command to PAMIQ API.

        Args:
            endpoint: API endpoint name
        """
        try:
            response = httpx.post(f"http://{self.host}:{self.port}/api/{endpoint}")
            result = json.loads(response.text)
            print(f"{endpoint}: {result.get('result', 'error')}")
        except httpx.ConnectError:
            print(f"{endpoint}: Connection failed, continuing...")
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 503:
                print(f"{endpoint}: Service unavailable, continuing...")
            else:
                print(f"{endpoint}: HTTP error {e.response.status_code}")
        except httpx.RequestError as e:
            print(f"{endpoint}: Request error: {e}")

    @staticmethod
    def get_key_name(key: keyboard.Key | keyboard.KeyCode) -> str | None:
        """Convert key object to lowercase string name.

        Args:
            key: Key object from pynput

        Returns:
            Lowercase key name or None if not determinable
        """
        if isinstance(key, keyboard.Key):
            return key.name.lower()
        else:
            if key.char:
                return key.char.lower()

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
        self._current_keys.add(name)

        if self._current_keys == self._pause_keys:
            self.send_command("pause")
        elif self._current_keys == self._resume_keys:
            self.send_command("resume")

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
        print(f"Pause: {'+'.join(self._pause_keys)}")
        print(f"Resume: {'+'.join(self._resume_keys)}")
        print("Press Ctrl+C to exit.")

        with keyboard.Listener(
            on_press=self.on_press, on_release=self.on_release
        ) as listener:
            listener.join()


def main() -> None:
    """Entry point of pamiq-kbctl."""
    parser = argparse.ArgumentParser(description="PAMIQ keyboard controller")
    parser.add_argument("--host", default="localhost", help="API server host")
    parser.add_argument("--port", default=8391, type=int, help="API server port")
    parser.add_argument(
        "--pause-key", default="alt+shift+p", help="Key combination for pause"
    )
    parser.add_argument(
        "--resume-key", default="alt+shift+r", help="Key combination for resume"
    )

    args = parser.parse_args()

    controller = KeyboardController(
        host=args.host,
        port=args.port,
        pause_keys=args.pause_key,
        resume_keys=args.resume_key,
    )

    try:
        controller.run()
    except KeyboardInterrupt:
        print("\nKeyboard controller stopped.")
        sys.exit(0)


if __name__ == "__main__":
    main()
