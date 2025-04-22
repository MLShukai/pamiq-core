import argparse
import cmd
import json
from typing import override

import httpx


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
        # Update status when every command execution.
        status = self._fetch_status()
        # Check command depend on WebAPI
        cmd_name, _, _ = self.parseline(line)
        if cmd_name in ["pause", "p", "resume", "r", "save", "shutdown"]:
            # Check if WebAPI available.
            if status == "offline":
                print(f'Command "{cmd_name}" not executed. Can\'t connect AMI system.')
                return False
        # Execute command
        return super().onecmd(line)

    def get_all_commands(self) -> list[str]:
        return [attr[3:] for attr in dir(self) if attr.startswith("do_")]

    @override
    def do_help(self, arg: str) -> None:
        """Show all commands and details."""
        print(
            "\n".join(
                [
                    "h/help    Show all commands and details.",
                    "p/pause   Pause the AMI system.",
                    "r/resume  Resume the AMI system.",
                    "save    Save a checkpoint.",
                    "shutdown  Shutdown the AMI system.",
                    "q/quit    Exit the console.",
                ]
            )
        )

    def do_h(self, arg: str) -> None:
        """Show all commands and details."""
        self.do_help(arg)

    def do_pause(self, arg: str) -> None:
        """Pause the AMI system."""
        response = httpx.post(f"http://{self._host}:{self._port}/api/pause")
        print(json.loads(response.text)["result"])

    def do_p(self, arg: str) -> None:
        """Pause the AMI system."""
        return self.do_pause(arg)

    def do_resume(self, arg: str) -> None:
        """Resume the AMI system."""
        response = httpx.post(f"http://{self._host}:{self._port}/api/resume")
        print(json.loads(response.text)["result"])

    def do_r(self, arg: str) -> None:
        """Resume the AMI system."""
        return self.do_resume(arg)

    def do_shutdown(self, arg: str) -> bool:
        """Shutdown the AMI system."""
        confirm = input("Confirm AMI system shutdown? (y/[N]): ")
        if confirm.lower() in ["y", "yes"]:
            response = httpx.post(f"http://{self._host}:{self._port}/api/shutdown")
            print(json.loads(response.text)["result"])
            return True
        print("Shutdown cancelled.")
        return False

    def do_quit(self, arg: str) -> bool:
        """Exit the console."""
        return True

    def do_q(self, arg: str) -> bool:
        """Exit the console."""
        return self.do_quit(arg)

    def do_save(self, arg: str) -> None:
        """Save a checkpoint."""
        response = httpx.post(f"http://{self._host}:{self._port}/api/save-state")
        print(json.loads(response.text)["result"])

    @override
    def postcmd(self, stop: bool, line: str) -> bool:
        self._fetch_status()
        return stop

    def _fetch_status(self) -> str:
        try:
            response = httpx.get(f"http://{self._host}:{self._port}/api/status")
        except httpx.RequestError:
            self.prompt = "pamiq-console (offline) > "
            return "offline"
        status = json.loads(response.text)["status"]
        self.prompt = f"pamiq-console ({status}) > "
        return status


def main() -> None:
    """Entry point of pamiq-console."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", default=8391, type=int)
    args = parser.parse_args()

    console = Console(args.host, args.port)
    console.cmdloop()


if __name__ == "__main__":
    main()
