"""A tiny named-component registry.

Every swappable stage of the pipeline -- sources, text generators, erasers,
renderers, writers -- is looked up by name through one of these. That is what
lets the CLI take `--renderer glyph` and lets you add your own without touching
the package:

    from docaug.render import RENDERERS

    @RENDERERS.register("my-renderer")
    def build(settings, **kw):
        return MyRenderer(**kw)

Third-party packages register the same way through entry points, so an installed
plugin shows up in `docaug list` without any import in this codebase::

    [project.entry-points."docaug.renderers"]
    my-renderer = "my_pkg:build"

A registry stores *factories*, not instances, so nothing expensive is
constructed until the name is actually used.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from importlib.metadata import entry_points
from typing import Generic, TypeVar

T = TypeVar("T")


class UnknownComponent(KeyError):
    """Raised for a name the registry does not have, listing the ones it does."""

    def __init__(self, kind: str, name: str, known: list[str]) -> None:
        super().__init__(name)
        self.kind, self.name, self.known = kind, name, known

    def __str__(self) -> str:
        return f"unknown {self.kind} {self.name!r}; available: {', '.join(self.known) or '(none)'}"


@dataclass
class Registry(Generic[T]):
    """Maps a name to a factory that builds a component of type `T`."""

    kind: str
    """What is being registered ("renderer", "writer", ...). Used in messages."""
    entry_point_group: str | None = None
    """Setuptools entry-point group scanned for third-party components."""

    _factories: dict[str, Callable[..., T]] = field(default_factory=dict, repr=False)
    _loaded_plugins: bool = field(default=False, repr=False)

    def register(self, name: str) -> Callable[[Callable[..., T]], Callable[..., T]]:
        """Decorator: bind `name` to the decorated factory."""

        def decorate(factory: Callable[..., T]) -> Callable[..., T]:
            self._factories[name] = factory
            return factory

        return decorate

    def create(self, name: str, /, **kwargs) -> T:
        """Build the component registered as `name`."""
        self._load_plugins()
        try:
            factory = self._factories[name]
        except KeyError:
            raise UnknownComponent(self.kind, name, self.names()) from None
        return factory(**kwargs)

    def names(self) -> list[str]:
        self._load_plugins()
        return sorted(self._factories)

    def _load_plugins(self) -> None:
        if self._loaded_plugins or not self.entry_point_group:
            return
        self._loaded_plugins = True  # set first: a broken plugin must not retry forever
        for ep in entry_points(group=self.entry_point_group):
            self._factories.setdefault(ep.name, ep.load())

    def __contains__(self, name: str) -> bool:
        return name in self.names()

    def __iter__(self) -> Iterator[str]:
        return iter(self.names())
