"""Scraper registry with pkgutil auto-discovery.

discover() imports every module in the ``scrapers`` package, collects BaseScraper
subclasses (skipping base/registry/template_scraper) and registers them by
SCRAPER_KEY. Module import is defensive: a scraper that fails to import (e.g. an
optional dependency like playwright is missing) is logged and skipped rather than
breaking discovery of the others.
"""
from __future__ import annotations

import importlib
import logging
import pkgutil
from typing import Optional, Type

from scrapers.base import BaseScraper

logger = logging.getLogger(__name__)

# Modules in the package that are not concrete, auto-routable scrapers.
_SKIP_MODULES = {"base", "registry", "__init__", "template_scraper"}


class ScraperRegistry:
    def __init__(self) -> None:
        self._scrapers: dict[str, Type[BaseScraper]] = {}
        self._discovered = False

    def register(self, cls: Type[BaseScraper]) -> None:
        key = getattr(cls, "SCRAPER_KEY", "")
        if not key:
            logger.warning("Skipping %s: empty SCRAPER_KEY", cls.__name__)
            return
        existing = self._scrapers.get(key)
        if existing is not None and existing is not cls:
            logger.warning(
                "Duplicate SCRAPER_KEY %r (%s overrides %s)",
                key, cls.__name__, existing.__name__,
            )
        self._scrapers[key] = cls

    def discover(self, force: bool = False) -> dict[str, Type[BaseScraper]]:
        if self._discovered and not force:
            return self._scrapers
        import scrapers as pkg

        for mod in pkgutil.iter_modules(pkg.__path__):
            if mod.name in _SKIP_MODULES:
                continue
            try:
                module = importlib.import_module(f"scrapers.{mod.name}")
            except Exception as exc:  # missing optional dep, syntax error, etc.
                logger.warning("Could not import scraper module %r: %s", mod.name, exc)
                continue
            for attr in vars(module).values():
                if (
                    isinstance(attr, type)
                    and issubclass(attr, BaseScraper)
                    and attr is not BaseScraper
                    and getattr(attr, "SCRAPER_KEY", "")
                ):
                    self.register(attr)
        self._discovered = True
        logger.info(
            "Discovered %d scrapers: %s",
            len(self._scrapers), ", ".join(sorted(self._scrapers)) or "(none)",
        )
        return self._scrapers

    def all_keys(self) -> list[str]:
        self.discover()
        return sorted(self._scrapers)

    def all_classes(self) -> list[Type[BaseScraper]]:
        self.discover()
        return list(self._scrapers.values())

    def get_class(self, key: str) -> Optional[Type[BaseScraper]]:
        self.discover()
        return self._scrapers.get(key)

    def get(self, key: str, **kwargs) -> Optional[BaseScraper]:
        cls = self.get_class(key)
        if cls is None:
            logger.error("No scraper registered for key %r", key)
            return None
        try:
            return cls(**kwargs)
        except Exception as exc:
            logger.error("Failed to instantiate scraper %r: %s", key, exc)
            return None

    def match(
        self, url: str, page_source: Optional[str] = None
    ) -> Optional[Type[BaseScraper]]:
        """Return the best-matching scraper class (lowest PRIORITY), or None.

        Scrapers with empty SITE_HINTS (e.g. static_html) never auto-match.
        """
        self.discover()
        candidates = [
            c for c in self._scrapers.values() if c.matches(url, page_source)
        ]
        if not candidates:
            return None
        candidates.sort(key=lambda c: (c.PRIORITY, c.SCRAPER_KEY))
        return candidates[0]


# Module-level singleton shared across the codebase.
registry = ScraperRegistry()


def discover(force: bool = False) -> dict[str, Type[BaseScraper]]:
    return registry.discover(force=force)
