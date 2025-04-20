import argparse
import cmd
import json
from typing import override

import requests


class Console(cmd.Cmd):
    """pamiq-console.

    Users can Control pamiq with CUI interface interactively.
    """

    intro = 'Welcome to the PAMIQ console. "help" lists commands.\n'
    prompt: str

    def __init__(self, host: str, port: int) -> None:
        """Initialize CUI interface."""
        super().__init__()
        self._host = host
        self._port = port
        self._fetch_status()

    @override
    def onecmd(self, line: str) -> bool:
        """Check connection status before command execution."""
        # Check command depend on WebAPI
        cmd_name, _, _ = self.parseline(line)
        if cmd_name in ["pause", "p", "resume", "r", "ckpt", "c", "shutdown", "s"]:
            # Check if WebAPI available.
            if self._fetch_status() == "offline":
                print(f'Command "{cmd_name}" not executed. Can\'t connect AMI system.')
                return False
        # Execute command
        return super().onecmd(line)

    @override
    def do_help(self, arg: str) -> None:
        """Show all commands and details."""
        # Store methods with the same docstring
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
        # Define showing order
        _doc_order = [
            "Show all commands and details.",
            "Pause the AMI system.",
            "Resume the AMI system.",
            "Save a checkpoint.",
            "Shutdown the AMI system.",
            "Exit the console.",
        ]
        assert set(_doc_order) == set(groups.keys())
        # Show details in the format f"{c}/{command} {Docstring}"
        for doc in _doc_order:
            cmds = "/".join(sorted(groups[doc], key=len))
            print(f"{cmds:<11} {doc}")

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
        confirm = input("Confirm AMI system shutdown? (y/[N]): ")
        if confirm.lower() in ["y", "yes"]:
            response = requests.post(f"http://{self._host}:{self._port}/api/shutdown")
            print(json.loads(response.text)["result"])
            return True
        print("Shutdown cancelled.")
        return False

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

    def _fetch_status(self) -> str:
        try:
            response = requests.get(f"http://{self._host}:{self._port}/api/status")
        except requests.exceptions.ConnectionError:
            self.prompt = "PAMIQ-console (offline) > "
            return "offline"
        status = json.loads(response.text)["status"]
        self.prompt = f"PAMIQ-console ({status}) > "
        return status


def main() -> None:
    "Entry point of pamiq-console."
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", default=8391, type=int)
    args = parser.parse_args()

    console = Console(args.host, args.port)
    console.cmdloop()


if __name__ == "__main__":
    main()
