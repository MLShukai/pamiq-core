from pamiq_core.threads import ThreadTypes


def test_thread_types() -> None:
    assert ThreadTypes.CONTROL.name == "control"
    assert ThreadTypes.INFERENCE.name == "inference"
    assert ThreadTypes.TRAINING.name == "training"
