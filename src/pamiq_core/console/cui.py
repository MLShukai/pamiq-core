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

    @override
    def do_help(self, arg: str) -> None:
        """Show all commands and details."""
        # Store methods with the same docstring to the same key
        # groups = {"Doscstring": ["command", "c"]}
        groups: dict[str, list[str]] = {}
        for attr in dir(self):
            if attr.startswith("do_"):
                cmd_name = attr[3:]
                method = getattr(self, attr)
                doc = method.__doc__ or ""
                doc = doc
                if doc not in groups:
                    groups[doc] = []
                groups[doc].append(cmd_name)
        # Show details in the format f"{c}/{command} {Docstring}"
        for doc, cmds in groups.items():
            cmd_list = "/".join(sorted(cmds, key=len))
            print(f"{cmd_list:<11} {doc}")

    def do_h(self, arg: str) -> None:
        """Show all commands and details."""
        self.do_help(arg)

    def do_pause(self, arg: str) -> None:
        """Pause the AMI system."""
        response = requests.post(f"http://{self._host}:{self._port}/api/pause")
        print(json.loads(response.text)["result"])

    def do_p(self, arg: str) -> None:
        """Pause the AMI system."""
        return self.do_pause(arg)

    def do_resume(self, arg: str) -> None:
        """Resume the AMI system."""
        response = requests.post(f"http://{self._host}:{self._port}/api/resume")
        print(json.loads(response.text)["result"])

    def do_r(self, arg: str) -> None:
        """Resume the AMI system."""
        return self.do_resume(arg)

    def do_shutdown(self, arg: str) -> bool:
        """Shutdown the AMI system."""
        response = requests.post(f"http://{self._host}:{self._port}/api/shutdown")
        print(json.loads(response.text)["result"])
        return True

    def do_s(self, arg: str) -> bool:
        """Shutdown the AMI system."""
        return self.do_shutdown(arg)

    def do_quit(self, arg: str) -> bool:
        """Exit the console."""
        return True

    def do_q(self, arg: str) -> bool:
        """Exit the console."""
        return self.do_quit(arg)

    def do_ckpt(self, arg: str) -> None:
        """Save a checkpoint."""
        response = requests.post(f"http://{self._host}:{self._port}/api/save-state")
        print(json.loads(response.text)["result"])

    def do_c(self, arg: str) -> None:
        """Save a checkpoint."""
        self.do_ckpt(arg)

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
