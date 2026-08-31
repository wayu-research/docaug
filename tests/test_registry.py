import pytest

from docaug.registry import Registry, UnknownComponent


def test_register_and_create():
    registry: Registry[str] = Registry("thing")

    @registry.register("greeting")
    def build(name: str = "world") -> str:
        return f"hello {name}"

    assert registry.create("greeting") == "hello world"
    assert registry.create("greeting", name="docaug") == "hello docaug"
    assert "greeting" in registry


def test_unknown_name_lists_what_is_available():
    registry: Registry[str] = Registry("thing")
    registry.register("a")(lambda: "a")
    with pytest.raises(UnknownComponent) as excinfo:
        registry.create("b")
    assert "available: a" in str(excinfo.value)


def test_factories_are_not_called_until_the_name_is_used():
    registry: Registry[str] = Registry("thing")
    calls = []
    registry.register("lazy")(lambda: calls.append(1) or "x")
    assert registry.names() == ["lazy"]
    assert not calls
    registry.create("lazy")
    assert calls == [1]


def test_builtin_registries_are_populated():
    from docaug import ERASERS, RENDERERS, SOURCES, TEXT_GENERATORS, WRITERS

    assert "jsonl" in SOURCES
    assert {"keep", "translate", "synth"} <= set(TEXT_GENERATORS)
    assert "adaptive" in ERASERS
    assert {"font", "glyph", "chain"} <= set(RENDERERS)
    assert {"dataset", "preview"} <= set(WRITERS)
