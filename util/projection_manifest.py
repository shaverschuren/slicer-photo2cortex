"""Explicit, machine-readable record of the photo-projection workflow endpoint.

This enables appropriate skipping logic etc with multiple projections.
This module has no dependency on `slicer`/`vtk`/`qt` so it can be imported both
from ordinary Python (`main_slicer_loop.py`) and from inside 3D Slicer
(`slicer_script.py`). Completion here means "every selected photograph has a
projected brain-envelope representation".
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

PROJECTION_MANIFEST_FILENAME = "projection_manifest.json"


def build_projection_manifest(
    reference_photo_id: str,
    scene_path: Optional[str],
    geometry: Dict[str, Any],
    projections: List[Dict[str, Any]],
    unresolved_photo_ids: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Assemble the projection-manifest payload.

    `status` is only ever "complete" when there are no unresolved photos and a
    scene path was actually saved; otherwise it is "incomplete" so the outer
    loop never mistakes a partial result for a finished patient.
    """

    unresolved_photo_ids = list(unresolved_photo_ids or [])
    status = "complete" if (scene_path and not unresolved_photo_ids and projections) else "incomplete"
    return {
        "status": status,
        "reference_photo_id": reference_photo_id,
        "scene_path": scene_path,
        "geometry": geometry,
        "projections": projections,
        "unresolved_photo_ids": unresolved_photo_ids,
    }


def save_projection_manifest(manifest: Dict[str, Any], path: str) -> str:
    """Persist a projection manifest to JSON."""

    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    return path


def load_projection_manifest(path: str) -> Optional[Dict[str, Any]]:
    """Load a projection manifest, returning None if it does not exist or is unreadable."""

    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        return None


def _photo_ids_from_photo_set_manifest(photo_set_manifest: Dict[str, Any]) -> List[str]:
    photos = photo_set_manifest.get("photos") or []
    return sorted(str(photo.get("id")) for photo in photos if photo.get("id") is not None)


def projection_set_is_complete(projection_manifest_path: str, photo_set_manifest_path: Optional[str] = None) -> bool:
    """Return True only if a saved projection result exists and still matches the current photo set.

    If `photo_set_manifest_path` is provided and readable, the reference photo id
    and the full set of photo ids recorded in the projection manifest must match
    the current photo-set manifest. This prevents a stale "complete" result from
    hiding a patient whose photo selection has since changed.
    """

    manifest = load_projection_manifest(projection_manifest_path)
    if manifest is None or manifest.get("status") != "complete":
        return False
    if not manifest.get("scene_path") or not os.path.exists(manifest["scene_path"]):
        return False

    if photo_set_manifest_path and os.path.exists(photo_set_manifest_path):
        try:
            with open(photo_set_manifest_path, "r", encoding="utf-8") as fh:
                photo_set_manifest = json.load(fh)
        except (json.JSONDecodeError, OSError):
            return False

        if str(manifest.get("reference_photo_id")) != str(photo_set_manifest.get("reference_photo_id") or photo_set_manifest.get("reference")):
            return False

        current_ids = set(_photo_ids_from_photo_set_manifest(photo_set_manifest))
        recorded_ids = {str(entry.get("photo_id")) for entry in manifest.get("projections", []) if entry.get("photo_id") is not None}
        if current_ids != recorded_ids:
            return False

    return True
