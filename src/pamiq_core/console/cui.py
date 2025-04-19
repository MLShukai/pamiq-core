import argparse
import cmd
import json
import sys
from typing import override

import requests


class Console(cmd.Cmd):
    intro = 'Welcome to the AMI system console. "help" lists commands.\n'
    prompt: str

    def __init__(self, host: str, port: int) -> None:
        super().__init__()
        self._host = host
        self._port = port

        try:
            requests.get(f"http://{self._host}:{self._port}/api/status")
        except requests.exceptions.ConnectionError:
            print(
                "Error: Could not connect to the AMI system. Please make sure the AMI system is running and try again."
            )
            sys.exit(1)

        self._fetch_status()

    def do_pause(self, line: str) -> None:
        """Pause the AMI system."""
        response = requests.post(f"http://{self._host}:{self._port}/api/pause")
        print(json.loads(response.text)["result"])

    def do_p(self, line: str) -> None:
        """Pause the AMI system."""
        return self.do_pause(line)

    def do_resume(self, line: str) -> None:
        """Resume the AMI system."""
        response = requests.post(f"http://{self._host}:{self._port}/api/resume")
        print(json.loads(response.text)["result"])

    def do_r(self, line: str) -> None:
        """Resume the AMI system."""
        return self.do_resume(line)

    def do_shutdown(self, line: str) -> bool:
        """Shutdown the AMI system."""
        response = requests.post(f"http://{self._host}:{self._port}/api/shutdown")
        print(json.loads(response.text)["result"])
        return True

    def do_quit(self, line: str) -> bool:
        """Exit the console."""
        return True

    def do_q(self, line: str) -> bool:
        """Exit the console."""
        return self.do_quit(line)

    def do_exit(self, line: str) -> bool:
        """Exit the console."""
        return self.do_quit(line)

    def do_save_checkpoint(self, line: str) -> None:
        """Saves a checkpoint."""
        response = requests.post(f"http://{self._host}:{self._port}/api/save-state")
        print(json.loads(response.text)["result"])

    def do_ckpt(self, line: str) -> None:
        """Saves a checkpoint."""
        return self.do_save_checkpoint(line)

    @override
    def postcmd(self, stop: bool, line: str) -> bool:
        self._fetch_status()
        return stop

    def _fetch_status(self) -> None:
        try:
            response = requests.get(f"http://{self._host}:{self._port}/api/status")
        except requests.exceptions.ConnectionError:
            self.prompt = "ami (offline) > "
            return
        status = json.loads(response.text)["status"]
        self.prompt = f"ami ({status}) > "


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", default=8391, type=int)
    args = parser.parse_args()

    console = Console(args.host, args.port)
    console.cmdloop()


if __name__ == "__main__":
    main()
