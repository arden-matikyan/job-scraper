"""Shared config/path helpers used across the pipeline.

Locates the project root and the ``config/`` directory, and loads YAML config
files. These used to live in ``agent/recon_agent.py``; they are generic and have
no dependency on any agent code.
"""
from __future__ import annotations

import logging
import os

import yaml

logger = logging.getLogger(__name__)


def project_root() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def config_path(name: str) -> str:
    return os.path.join(project_root(), "config", name)


def load_yaml(path: str) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as exc:
        logger.error("Could not load YAML %s: %s", path, exc)
        return {}
