import argparse

from prompt_toolkit import prompt
from prompt_toolkit.completion import WordCompleter

from .system_status import SystemStatus
from .web_api import WebApiClient


class Console:
    """pamiq-console.

    Users can Control pamiq with CUI interface interactively.
    """

    status: SystemStatus

    def __init__(self, host: str, port: int) -> None:
        """Initialize CUI interface."""
        super().__init__()
        self._client = WebApiClient(host, port)
        self.all_commands: list[str] = [
            attr[len("command_") :] for attr in dir(self) if attr.startswith("command_")
        ]
        self._completer = WordCompleter(self.all_commands)
        self.status = SystemStatus.OFFLINE

    def fetch_status(self) -> None:
        """Check WebAPI status."""
        self.status = self._client.get_status()

    def run_command(self, command: str) -> bool | None:
        """Check connection status before command execution."""
        # Update self.status before command execution.
        self.fetch_status()
        # Check command depend on WebAPI
        if command in ["pause", "p", "resume", "r", "save", "shutdown"]:
            # Check if WebAPI available.
            if self.status is SystemStatus.OFFLINE:
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
                f"pamiq-console ({self.status.status_name}) > ",
                completer=self._completer,
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
        response = self._client.pause()
        if response:
            print(response)
        else:
            print("Failed to pause...")

    def command_p(self) -> None:
        """Pause the AMI system."""
        self.command_pause()

    def command_resume(self) -> None:
        """Resume the AMI system."""
        response = self._client.resume()
        if response:
            print(response)
        else:
            print("Failed to resume...")

    def command_r(self) -> None:
        """Resume the AMI system."""
        self.command_resume()

    def command_shutdown(self) -> bool:
        """Shutdown the AMI system."""
        confirm = input("Confirm AMI system shutdown? (y/[N]): ")
        if confirm.lower() in ["y", "yes"]:
            response = self._client.shutdown()
            if response:
                return True
            else:
                print("Failed to shutdown...")
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
        response = self._client.save_state()
        if response:
            print(response)
        else:
            print("Failed to save state...")


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
