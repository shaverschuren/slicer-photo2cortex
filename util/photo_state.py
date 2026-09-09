"""Slicer-side bookkeeping for the per-photograph projection state.

Replaces the flat `Nodes` dictionary with one small container
per photograph, so `slicer_script.py` can look up/update/save a photograph's
plane, projected envelope, and `Projection` instance without re-deriving them
from node names each time.
"""

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class PhotoProjectionState:
    """Everything materialised in the Slicer scene for one photograph."""

    photo_id: str
    role: str
    photo_type: Optional[str]
    image_path: str
    volume_node: Any
    plane_node: Any
    envelope_node: Any
    projection: Any

    @property
    def plane_node_name(self) -> str:
        return self.plane_node.GetName()

    @property
    def envelope_node_name(self) -> str:
        return self.envelope_node.GetName()
