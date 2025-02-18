import pytest

from pamiq_core.utils.alias_dict import AliasDict


class TestAliasDict:
    """Test cases for AliasDict class."""

    def test_init_empty(self):
        """Test initialization of empty dictionary."""
        d = AliasDict()
        assert len(d) == 0
        assert len(list(d.original_keys())) == 0
        assert len(list(d.alias_keys())) == 0
        assert len(d.all_keys()) == 0

    def test_init_with_dict(self):
        """Test initialization with dictionary."""
        d = AliasDict({"a": 1, "b": 2})
        assert d["a"] == 1
        assert d["b"] == 2
        assert set(d.original_keys()) == {"a", "b"}
        assert len(list(d.alias_keys())) == 0
        assert d.all_keys() == {"a", "b"}

    def test_set_alias(self):
        """Test setting aliases."""
        d = AliasDict({"a": 1})
        d.set_alias("a", "x")

        assert d["a"] == d["x"] == 1
        assert set(d.original_keys()) == {"a"}
        assert set(d.alias_keys()) == {"x"}
        assert d.all_keys() == {"a", "x"}

    def test_set_multiple_aliases(self):
        """Test setting multiple aliases for the same key."""
        d = AliasDict({"a": 1})
        d.set_alias("a", "x")
        d.set_alias("a", "y")

        assert d["a"] == d["x"] == d["y"] == 1
        assert set(d.alias_keys()) == {"x", "y"}
        assert d.all_keys() == {"a", "x", "y"}

    def test_set_alias_to_nonexistent_key(self):
        """Test setting alias to non-existent key raises KeyError."""
        d = AliasDict()
        with pytest.raises(
            KeyError, match="Source key 'a' does not exist in dictionary"
        ):
            d.set_alias("a", "x")

    def test_value_updates(self):
        """Test value updates with aliases."""
        d = AliasDict({"a": 1})
        d.set_alias("a", "x")

        # Update through original key
        d["a"] = 2
        assert d["a"] == d["x"] == 2

        # Update through alias removes the alias
        d["x"] = 3
        assert "x" not in d.alias_keys()
        assert d["x"] == 3
        assert d["a"] == 2

    def test_delete_operation(self):
        """Test delete operation is prohibited."""
        d = AliasDict({"a": 1})
        d.set_alias("a", "x")

        with pytest.raises(
            RuntimeError,
            match="Delete operation is prohibited to maintain alias integrity.",
        ):
            del d["a"]

        with pytest.raises(
            RuntimeError,
            match="Delete operation is prohibited to maintain alias integrity.",
        ):
            del d["x"]

    def test_contains_operation(self):
        """Test contains operation with aliases."""
        d = AliasDict({"a": 1})
        d.set_alias("a", "x")

        assert "a" in d
        assert "x" in d
        assert "b" not in d
        assert None not in d

    def test_keys_views(self):
        """Test key view operations."""
        d = AliasDict({"a": 1, "b": 2})
        d.set_alias("a", "x")
        d.set_alias("b", "y")

        assert set(d.original_keys()) == {"a", "b"}
        assert set(d.alias_keys()) == {"x", "y"}
        assert d.all_keys() == {"a", "b", "x", "y"}

        # Test that views are dynamic
        d["c"] = 3
        assert set(d.original_keys()) == {"a", "b", "c"}
        assert d.all_keys() == {"a", "b", "c", "x", "y"}
