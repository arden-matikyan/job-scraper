"""scrapers package — auto-discovers concrete scrapers via pkgutil on import.

Importing this package runs registry.discover() so that ``from scrapers import
registry`` yields a populated registry. Discovery is defensive: a scraper whose
optional dependency is missing is skipped, not fatal (see scrapers.registry).
"""
from scrapers.registry import discover, registry  # noqa: F401

# Populate the registry as a side effect of importing the package.
discover()
