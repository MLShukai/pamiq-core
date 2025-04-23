import argparse
import json

import httpx
from prompt_toolkit import prompt
from prompt_toolkit.completion import WordCompleter


class Console:
    """pamiq-console.

    Users can Control pamiq with CUI interface interactively.
    """

    status: str

    def __init__(self, host: str, port: int) -> None:
        """Initialize CUI interface."""
        super().__init__()
        self._host = host
        self._port = port
        self.all_commands: list[str] = [
            attr[len("command_") :] for attr in dir(self) if attr.startswith("command_")
        ]
        self._completer = WordCompleter(self.all_commands)
        self.fetch_status()

    def fetch_status(self) -> None:
        """Check WebAPI status."""
        try:
            response = httpx.get(f"http://{self._host}:{self._port}/api/status")
        except httpx.RequestError:
            self.status = "offline"
            return
        self.status = json.loads(response.text)["status"]

    def run_command(self, command: str) -> bool | None:
        """Check connection status before command execution."""
        # Update self.status before command execution.
        self.fetch_status()
        # Check command depend on WebAPI
        if command in ["pause", "p", "resume", "r", "save", "shutdown"]:
            # Check if WebAPI available.
            if self.status == "offline":
                print(f'Command "{command}" not executed. Can\'t connect AMI system.')
                return False
        # Execute command
        loop_end = getattr(self, f"command_{command}")()
        # Update self.status after command execution.
        self.fetch_status()
        # If True, main_loop ends.
        return loop_end

    def main_loop(self) -> None:
        """Running CUI interface."""
        print('Welcome to the PAMIQ console. "help" lists commands.\n')
        while True:
            self.fetch_status()
            command = prompt(
                f"pamiq-console ({self.status}) > ", completer=self._completer
            )
            if command == "":
                continue
            if command in self.all_commands:
                if self.run_command(command):
                    break
            else:
                print(f"*** Unknown syntax: {command}")

    def command_help(self) -> None:
        """Show all commands and details."""
        print(
            "\n".join(
                [
                    "h/help    Show all commands and details.",
                    "p/pause   Pause the AMI system.",
                    "r/resume  Resume the AMI system.",
                    "save      Save a checkpoint.",
                    "shutdown  Shutdown the AMI system.",
                    "q/quit    Exit the console.",
                ]
            )
        )

    def command_h(self) -> None:
        """Show all commands and details."""
        self.command_help()

    def command_pause(self) -> None:
        """Pause the AMI system."""
        response = httpx.post(f"http://{self._host}:{self._port}/api/pause")
        print(json.loads(response.text)["result"])

    def command_p(self) -> None:
        """Pause the AMI system."""
        self.command_pause()

    def command_resume(self) -> None:
        """Resume the AMI system."""
        response = httpx.post(f"http://{self._host}:{self._port}/api/resume")
        print(json.loads(response.text)["result"])

    def command_r(self) -> None:
        """Resume the AMI system."""
        self.command_resume()

    def command_shutdown(self) -> bool:
        """Shutdown the AMI system."""
        confirm = input("Confirm AMI system shutdown? (y/[N]): ")
        if confirm.lower() in ["y", "yes"]:
            response = httpx.post(f"http://{self._host}:{self._port}/api/shutdown")
            print(json.loads(response.text)["result"])
            return True
        print("Shutdown cancelled.")
        return False

    def command_quit(self) -> bool:
        """Exit the console."""
        return True

    def command_q(self) -> bool:
        """Exit the console."""
        return self.command_quit()

    def command_save(self) -> None:
        """Save a checkpoint."""
        response = httpx.post(f"http://{self._host}:{self._port}/api/save-state")
        print(json.loads(response.text)["result"])


def main() -> None:
    """Entry point of pamiq-console."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", default=8391, type=int)
    args = parser.parse_args()

    console = Console(args.host, args.port)
    console.main_loop()


if __name__ == "__main__":
    main()
