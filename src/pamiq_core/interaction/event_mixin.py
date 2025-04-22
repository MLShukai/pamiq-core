class InteractionEventMixin:
    """Define event functions for interactions."""

    def setup(self) -> None:
        """It is called first when an Interaction is performed."""
        pass

    def teardown(self) -> None:
        """It is called last when performing an Interaction."""
        pass
