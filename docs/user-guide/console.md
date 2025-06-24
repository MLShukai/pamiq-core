# Console

The `console` module provides the interactive command-line interface. It can control PAMIQ-Core system externally.

## PAMIQ Console

After launching a PAMIQ-Core system, you can connect to it using the `pamiq-console` command-line tool:

```sh
$ pamiq-console
Welcome to the PAMIQ console. "help" lists commands.

pamiq-console (active) >
```

The console prompt shows the current system state (e.g., `active`, `paused`, `shutting down`) in parentheses.

### Available Commands

- `h` or `help` - Show all available commands and their descriptions
- `p` or `pause` - Pause the system
- `r` or `resume` - Resume the system
- `save` - Save a checkpoint of the current system state
- `shutdown` - Shutdown the system (requires confirmation)
- `q` or `quit` - Exit the console (does not affect the running system)

### Launch Options

The `pamiq-console` command accepts the following options:

- `--host`: Hostname or IP address of the PAMIQ system (default: localhost)
- `--port`: Port number for the web API connection (default: 8391)

Example with custom connection settings:

```sh
$ pamiq-console --host 192.168.1.100 --port 9000
```

NOTE: You can modify the default address used by the system by changing the `web_api_address` parameter in [LaunchConfig](../api/launch.md). To disable the web API entirely, set `web_api_address=None`.

## Web API

PAMIQ-Core exposes a RESTful API that allows controlling the system over a network connection. This enables integration with external applications, monitoring tools, or custom interfaces.

### API Endpoints

- `GET /api/status` - Retrieve the current system status
- `POST /api/pause` - Pause the system
- `POST /api/resume` - Resume the system
- `POST /api/shutdown` - Shutdown the system
- `POST /api/save-state` - Save the current system state

### Example API Usage

Using `curl` to interact with the API:

```sh
# Get current status
curl http://localhost:8391/api/status

# Pause the system
curl -X POST http://localhost:8391/api/pause

# Save current state
curl -X POST http://localhost:8391/api/save-state
```

Response format is JSON, for example:

```json
{"status": "active"}  // For status endpoint
{"result": "ok"}      // For action endpoints
```

### System Status

The status endpoint returns one of the following values:

- `active` - System is running normally
- `pausing` - System is in the process of pausing
- `paused` - System is fully paused
- `resuming` - System is in the process of resuming
- `shutting down` - System is shutting down

## Keyboard Shortcut Controller

PAMIQ-Core provides a keyboard shortcut controller. Users can pause/resume a agent by keyboard shortcuts.

### Default Shortcuts

- Windows/Linux: `Alt+Shift+P` (Pause) and `Alt+Shift+R` (Resume)
- macOS: `Option+Shift+P` (Pause) and `Option+Shift+R` (Resume)
- Windows also supports `Alt+Shift+Q` to quit the controller

### Installation

```sh
pip install pamiq-core[kbctl]
```

**Note for Linux users**: The following dependencies may be required:

```sh
sudo apt-get install libevdev-dev build-essential
```

### Usage

```sh
pamiq-kbctl
```

### macOS Accessibility Permissions

On macOS, you must grant Accessibility permissions to your terminal application before using `pamiq-kbctl`:

1. Go to System Settings → Privacy & Security → Accessibility
2. Click the "+" button
3. Add your terminal application (Terminal, iTerm2, VS Code Terminal, etc.)
4. Restart your terminal application after granting permissions

Without these permissions, macOS will prevent keyboard monitoring and `pamiq-kbctl` will not function properly.

### Command-line Options

- `--host`: Hostname of the PAMIQ system (default: localhost)
- `--port`: Port number for the web API (default: 8391)
- `--pause-key`: Custom key combination for pause
- `--resume-key`: Custom key combination for resume
- `--quit-key`: Custom key combination to exit the controller

Example with custom shortcuts:

```sh
pamiq-kbctl --pause-key "ctrl+p" --resume-key "ctrl+r"
```

## API Reference

For detailed information about the classes and methods in the console module, check out the [API Reference](../api/console.md).
