from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit.components.v1 as components


_COMPONENT_PATH = Path(__file__).resolve().parent / "components" / "tactical_board"
_tactical_board = components.declare_component("tactical_board_v3", path=str(_COMPONENT_PATH))


def tactical_board(
    *,
    field_image_url: str,
    objects: list[dict[str, Any]] | None = None,
    active_color: str = "#ef4444",
    active_size: int = 18,
    active_label: str = "",
    exercise_id: str = "",
    objects_revision: int = 0,
    palette_mode: str = "objects",
    height: int = 620,
    key: str | None = None,
):
    return _tactical_board(
        fieldImageUrl=field_image_url,
        objects=objects or [],
        activeColor=active_color,
        activeSize=active_size,
        activeLabel=active_label,
        exerciseId=exercise_id,
        objectsRevision=objects_revision,
        paletteMode=palette_mode,
        height=height,
        key=key,
        default={"objects": objects or [], "selected_id": None},
    )
