"""Import every module in a consumer package so its agents register. Agents
self-register when their class body runs (``Agent.__init_subclass__``);
importing the module is what triggers that. Drop a new ``*.py`` into the
package and it goes live on the next start, with no list to maintain.
Modules whose name starts with ``_`` are skipped."""

from __future__ import annotations

import importlib
import pkgutil

from aixon.logging import Logger

_log = Logger("aixon.discovery")


class NotAPackageError(ValueError):
    """O nome importa, mas é um módulo e não um pacote (sem ``__path__``).
    Exceção própria para que o CLI silencie só este caso, e não um
    ValueError levantado dentro do módulo do usuário."""


def autodiscover(package: str) -> None:
    pkg = importlib.import_module(package)
    if not hasattr(pkg, "__path__"):
        raise NotAPackageError(f"{package!r} is not a package (has no __path__).")
    _log.info(f"autodiscover: scanning package '{package}'")
    count = 0
    for module in pkgutil.iter_modules(pkg.__path__):
        if not module.name.startswith("_"):
            importlib.import_module(f"{package}.{module.name}")
            count += 1
    _log.info(f"autodiscover: imported {count} module(s) from '{package}'")
