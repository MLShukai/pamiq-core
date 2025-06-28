from typing import Any

from pamiq_core.data import DataCollector, DataUser

type DataUserType[R] = DataUser[Any, R]
type DataCollectorType[T] = DataCollector[T, Any]
