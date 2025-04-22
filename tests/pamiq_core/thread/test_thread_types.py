from pamiq_core.thread import ThreadTypes


def test_thread_types() -> None:
    assert ThreadTypes.CONTROL.thread_name == "control"
    assert ThreadTypes.INFERENCE.thread_name == "inference"
    assert ThreadTypes.TRAINING.thread_name == "training"
