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

NOTE: You can modify the default address used by the system by changing the `web_api_address` parameter in [LaunchConfig](../api/launch.md).

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

## API Reference

For detailed information about the classes and methods in the console module, check out the [API Reference](../api/console.md).
