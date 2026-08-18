"""Metadata and cached LightGlue packet I/O."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


REQUIRED_PACKET_KEYS = ("points0", "points1", "scores", "image_size0", "image_size1")
REQUIRED_PAIR_KEYS = ("scene_id", "K0", "K1", "T_0to1")


def load_metadata(path: str | Path) -> list[dict]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError("Metadata must be a JSON list of image-pair objects.")
    for index, pair in enumerate(value):
        missing = [key for key in REQUIRED_PAIR_KEYS if key not in pair]
        if missing:
            raise ValueError(f"Metadata item {index} is missing keys: {missing}")
    return value


def load_packet(packet_dir: str | Path, index: int) -> dict[str, np.ndarray]:
    path = Path(packet_dir) / f"{int(index):04d}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing cached match packet: {path}")
    with np.load(path, allow_pickle=False) as source:
        packet = {key: source[key] for key in source.files}
    missing = [key for key in REQUIRED_PACKET_KEYS if key not in packet]
    if missing:
        raise ValueError(f"Packet {path} is missing keys: {missing}")
    count = len(packet["scores"])
    if len(packet["points0"]) != count or len(packet["points1"]) != count:
        raise ValueError(f"Packet {path} has inconsistent match-array lengths.")
    return packet
