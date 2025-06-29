from typing import Any

from pamiq_core.data import DataCollector, DataUser

# For Agent.get_data_collector
type DataUserType[R] = DataUser[Any, R]

# For Trainer.get_data_user
type DataCollectorType[T] = DataCollector[T, Any]
