"""Thin wrapper over Python's stdlib logging. Level is configurable via the
LOG_LEVEL env var (default INFO). Use it for framework/consumer diagnostics —
NOT for streaming an agent's generated content (that is the reasoning channel
and the CLI)."""

import logging
import os


class Logger:
    def __init__(self, name: str):
        level_name = os.getenv("LOG_LEVEL", "INFO").upper()
        level = logging.getLevelName(level_name)
        if not isinstance(level, int):
            level = logging.INFO

        self._logger = logging.getLogger(name)
        self._logger.setLevel(level)
        # Own handler + no propagation: one line per record, even when the
        # consumer configures the root logger.
        self._logger.propagate = False
        if not self._logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "[%(asctime)s] %(levelname)s %(name)s — %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            handler.setFormatter(formatter)
            self._logger.addHandler(handler)
        # Named `h`, not `handler`: reusing `handler` here would make mypy
        # widen its inferred type from the concrete `StreamHandler` built
        # above to the base `logging.Handler` yielded by `.handlers`.
        for h in self._logger.handlers:
            h.setLevel(level)

    def info(self, msg: str, *args, **kwargs):
        self._logger.info(msg, *args, **kwargs)

    def warning(self, msg: str, *args, **kwargs):
        self._logger.warning(msg, *args, **kwargs)

    def error(self, msg: str, *args, **kwargs):
        self._logger.error(msg, *args, **kwargs)

    def debug(self, msg: str, *args, **kwargs):
        self._logger.debug(msg, *args, **kwargs)
