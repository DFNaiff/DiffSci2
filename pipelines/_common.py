"""Shared helpers for all pipelines/*/ scripts.

Keeps the individual stage scripts thin and uniform in how they resolve
paths, load YAML configs, and locate the three root directories.

The three environment variables the whole pipeline uses:

- DIFFSCI2_DATA_ROOT   — where raw micro-CT volumes live.
                         default: <repo>/saveddata/raw/imperial_college
- DIFFSCI2_MODEL_ROOT  — where checkpoints live.
                         default: <repo>/savedmodels
- DIFFSCI2_FIELDS_ROOT — where GP fits / porosity fields live.
                         default: <repo>/notebooks/exploratory/dfn/data

None of these are mandatory; each has a sensible in-repo default. The
point is that a user with a different disk layout can set one env var
and be done.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent


def data_root() -> Path:
    return Path(
        os.environ.get(
            "DIFFSCI2_DATA_ROOT",
            str(REPO_ROOT / "saveddata" / "raw" / "imperial_college"),
        )
    )


def model_root() -> Path:
    return Path(
        os.environ.get(
            "DIFFSCI2_MODEL_ROOT",
            str(REPO_ROOT / "savedmodels"),
        )
    )


def fields_root() -> Path:
    return Path(
        os.environ.get(
            "DIFFSCI2_FIELDS_ROOT",
            str(REPO_ROOT / "notebooks" / "exploratory" / "dfn" / "data"),
        )
    )


def load_config(path: str | Path) -> dict[str, Any]:
    """Read a pipeline YAML config and return it as a plain dict."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


@dataclass(frozen=True)
class PublishedCheckpoints:
    """Canonical on-disk location of the published artifacts."""

    vae: Path
    base: Path           # <rock>_pcond.ckpt
    fc257: Path          # field_controlled/fc257/<rock>.ckpt (symlink)
    fc129: Path          # field_controlled/fc129/<rock>.ckpt (symlink)


def published_checkpoints(rock: str) -> PublishedCheckpoints:
    """Canonical paths for the *published* view of the checkpoints.

    The physical checkpoint for fc257/fc129 is a symlink under
    savedmodels/pore/field_controlled/ pointing into the corresponding
    dated run under savedmodels/experimental/. The symlink layer is
    created by the reorg once and never touched again.
    """
    mr = model_root()
    pore = mr / "pore"
    prod = pore / "production"
    fc = pore / "field_controlled"
    rl = rock.lower()
    return PublishedCheckpoints(
        vae=prod / "converted_vaenet.ckpt",
        base=prod / f"{rl}_pcond.ckpt",
        fc257=fc / "fc257" / f"{rl}.ckpt",
        fc129=fc / "fc129" / f"{rl}.ckpt",
    )


def raw_volume_path(volume_filename: str) -> Path:
    """Absolute path to a raw micro-CT volume file."""
    return data_root() / volume_filename


def field_dir(radius: int, rock: str) -> Path:
    """Directory where the GP fit + field live for a given radius/rock.

    Matches the layout of the paper:
        <fields_root>/gpdata4-<kernel_size>/<rock_lower>/
    with kernel_size = 2 * radius + 1.
    """
    kernel_size = 2 * radius + 1
    return fields_root() / f"gpdata4-{kernel_size}" / rock.lower()


def parse_devices(spec: str) -> list[int]:
    """Turn '0,1,2,3' into [0, 1, 2, 3]."""
    return [int(d.strip()) for d in spec.split(",") if d.strip()]
