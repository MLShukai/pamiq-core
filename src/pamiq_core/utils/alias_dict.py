from collections import UserDict
from collections.abc import KeysView
from typing import Any, override


class AliasDict[K, V](UserDict[K, V]):
    """Dictionary that supports aliasing between keys.

    This dictionary allows creating aliases between keys, where multiple
    keys can reference the same value through key aliasing. When a key
    is aliased to another key, accessing the alias key will return the
    value of the original key.
    """

    @override
    def __init__(self, *args: Any, **kwds: Any) -> None:
        """Initialize an AliasDict.

        Initializes a new dictionary that can handle aliased keys.
        Accepts same arguments as dict constructor.
        """
        self._alias_map: dict[K, K] = {}
        super().__init__(*args, **kwds)

    def set_alias(self, src: K, dst: K) -> None:
        """Create an alias from source key to destination key.

        Creates a reference where dst key will return the same value as src key.

        Args:
            src: Source key to create alias from
            dst: Destination key to create alias to

        Raises:
            KeyError: If source key doesn't exist in dictionary
        """
        if src not in self:
            raise KeyError(f"Source key '{src}' does not exist in dictionary")
        self._alias_map[dst] = src

    def original_keys(self) -> KeysView[K]:
        """Get keys that are not aliases."""
        return self.keys()

    def alias_keys(self) -> KeysView[K]:
        """Get keys that are aliases."""
        return self._alias_map.keys()

    def all_keys(self) -> set[K]:
        """Get all keys including both original and alias keys."""
        return {*self.original_keys(), *self.alias_keys()}

    @override
    def __getitem__(self, key: K) -> V:
        """Get item from dictionary, resolving aliases if necessary.

        Args:
            key: Key to get value for

        Returns:
            Value associated with key or its alias
        """
        if key in self._alias_map:
            key = self._alias_map[key]
        return super().__getitem__(key)

    @override
    def __setitem__(self, key: K, item: V) -> None:
        """Set item in dictionary, removing any alias if it exists.

        Args:
            key: Key to set value for
            item: Value to set
        """
        super().__setitem__(key, item)
        if key in self._alias_map:
            del self._alias_map[key]

    @override
    def __delitem__(self, key: K) -> None:
        """Delete operation is prohibited.

        Raises:
            RuntimeError: Always raised as deletion is not allowed
        """
        raise RuntimeError(
            "Delete operation is prohibited to maintain alias integrity."
        )

    @override
    def __contains__(self, key: object) -> bool:
        """Check if key exists in dictionary or as an alias.

        Args:
            key: Key to check

        Returns:
            True if key exists either as a regular key or an alias
        """
        if key in self._alias_map:
            return True
        return super().__contains__(key)
