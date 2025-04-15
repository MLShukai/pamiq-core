import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import time
from .data import DataBuffer, DataUsersDict
from .interaction import Interaction
from .model import TrainingModel, TrainingModelsDict
from .state_persistence import LatestStatesKeeper, StateStore
from .thread.threads import ControlThread, InferenceThread, TrainingThread
from .trainer import Trainer, TrainersDict


@dataclass(frozen=True)
class LaunchConfig:
    """Configuration parameters for system launch.

    This class encapsulates all configuration parameters needed to launch
    the AMI system. It controls thread behavior, state persistence, API settings,
    timing, and other critical system parameters.

    Attributes:
        states_dir: Directory path where states will be saved.
        state_name_format: Format string for state directory names.
        saved_state_path: Optional path to a previously saved state to load at startup.
        save_state_interval: Interval in seconds between automatic state saves.
            Use infinity for no automatic saves.
        max_keep_states: Maximum number of state directories to keep in the states directory.
            Older state directories beyond this number will be automatically removed.
            Use -1 to disable this feature (no automatic removal).
        state_name_pattern: Glob pattern to identify state directories for management.
        states_cleanup_interval: Interval in seconds between automatic state cleanup.
        timeout_for_all_threads_pause: Maximum time in seconds to wait for all
            threads to pause before timing out.
        max_attempts_to_pause_all_threads: Maximum number of retry attempts
            when pausing threads fails.
        max_uptime: Maximum time in seconds the system is allowed to run.
            Use infinity for no time limit.
        web_api_address: Tuple of (host, port) for the web API server.
        web_api_command_queue_size: Maximum size of the command queue for the web API.
        log_tick_time_statistics_interval: Interval in seconds for logging
            step time statistics in inference thread.
        time_scale: Scale factor for system time, affecting the speed of time passage.
    """

    states_dir: str | Path = Path("./states")
    state_name_format: str = "%Y-%m-%d_%H-%M-%S,%f.state"
    saved_state_path: str | Path | None = None
    save_state_interval: float = float("inf")
    max_keep_states: int = -1
    state_name_pattern: str = "*.state"
    states_cleanup_interval: float = 60.0
    timeout_for_all_threads_pause: float = 60.0
    max_attempts_to_pause_all_threads: int = 3
    max_uptime: float = float("inf")
    web_api_address: tuple[str, int] = ("localhost", 8319)
    web_api_command_queue_size: int = 1
    log_tick_time_statistics_interval: float = 60.0
    time_scale: float = 1.0


def launch(
    interaction: Interaction[Any, Any],
    models: Mapping[str, TrainingModel[Any]],
    data: Mapping[str, DataBuffer[Any]],
    trainers: Mapping[str, Trainer],
    config: Mapping[str, Any] | LaunchConfig | None = None,
) -> None:
    """Launch the AMI system with specified components and configuration.

    This function is the main entry point for starting the AMI system. It initializes
    and connects all system components, sets up the multi-threading architecture,
    and starts the control, inference, and training threads.

    The function will run until the system is shut down or interrupted. Once complete,
    it will save the final system state before exiting.

    Args:
        interaction: Agent-environment interaction procedure.
        models: Dictionary mapping names to training models.
        data: Dictionary mapping names to data buffers.
        trainers: Dictionary mapping names to trainers.
        config: Configuration parameters, either as a LaunchConfig instance
            or a dictionary of parameter values. If None, default configuration
            is used.
    """
    logger = logging.getLogger(__name__)

    # Initialize configuration
    if config is None:
        config = {}
    if not isinstance(config, LaunchConfig):
        config = LaunchConfig(**config)

    # Initialize system components with proper containers
    training_models = TrainingModelsDict(models)
    data_users = DataUsersDict.from_data_buffers(data)
    trainers_dict = TrainersDict(trainers)

    interaction.agent.attach_inference_models(training_models.inference_models_dict)
    interaction.agent.attach_data_collectors(data_users.data_collectors_dict)

    trainers_dict.attach_training_models(training_models)
    trainers_dict.attach_data_users(data_users)

    # Set up state persistence
    logger.info(f"Setting up state persistence in directory: {config.states_dir}")
    state_store = StateStore(config.states_dir, config.state_name_format)
    state_store.register("interaction", interaction)
    state_store.register("models", training_models)
    state_store.register("data", data_users)
    state_store.register("trainers", trainers_dict)
    state_store.register("time", time.get_global_time_controller())

    states_keeper = LatestStatesKeeper(
        states_dir=config.states_dir,
        state_name_pattern=config.state_name_pattern,
        max_keep=config.max_keep_states,
        cleanup_interval=config.states_cleanup_interval,
    )

    # Load state if specified
    if config.saved_state_path is not None:
        logger.info(f"Loading state from '{config.saved_state_path}'")
        state_store.load_state(config.saved_state_path)

    # Initialize threads
    control_thread = ControlThread(
        state_store,
        save_state_interval=config.save_state_interval,
        timeout_for_all_threads_pause=config.timeout_for_all_threads_pause,
        max_attempts_to_pause_all_threads=config.max_attempts_to_pause_all_threads,
        max_uptime=config.max_uptime,
        web_api_address=config.web_api_address,
        web_api_command_queue_size=config.web_api_command_queue_size,
    )
    inference_thread = InferenceThread(
        interaction,
        log_tick_time_statistics_interval=config.log_tick_time_statistics_interval,
    )
    training_thread = TrainingThread(trainers_dict)

    # Connect threads
    for thread in [inference_thread, training_thread]:
        thread.attach_controller(control_thread.controller)

    control_thread.attach_thread_statuses(
        {t.THREAD_TYPE: t.thread_status for t in [inference_thread, training_thread]}
    )

    # Launch the system
    logger.info("Launching AMI system...")

    try:
        logger.info(f"Setting time scale to {config.time_scale}")
        time.set_time_scale(config.time_scale)

        states_keeper.start()

        inference_thread.start()
        training_thread.start()
        control_thread.run()  # Blocking until shutdown or KeyboardInterrupt
    except KeyboardInterrupt:
        logger.info("Keyboard Interrupt detected, shutting down system")
    except Exception as e:
        logger.error(f"Error during system execution: {e}", exc_info=True)
        raise
    finally:
        inference_thread.join()
        training_thread.join()
        time.set_time_scale(1.0)  # Fix time scale.

        logger.info("Saving final system state")
        state_path = state_store.save_state()
        logger.info(f"Final state saved to '{state_path}'")

        states_keeper.stop()
        states_keeper.cleanup()
