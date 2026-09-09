"""Ordinary-Python tests for the projection-manifest completion record. Claude-generated."""

import json

from util import projection_manifest as pm


_GEOMETRY = {
    "plane_dims": [100.0, 80.0],
    "camera_distance_mm": 150.0,
    "projection_slab_thickness_mm": 15.0,
    "reference_transform_matrix": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
}


def _projection_entry(photo_id, role="secondary"):
    return {
        "photo_id": photo_id,
        "role": role,
        "photo_type": None,
        "image_path": f"{photo_id}.jpg",
        "plane_node_name": f"PhotoPlane__{photo_id}",
        "envelope_node_name": f"ProjectedEnvelope__{photo_id}",
    }


def test_build_projection_manifest_is_complete_with_scene_and_no_unresolved(tmp_path):
    scene_dir = tmp_path / "scene"
    scene_dir.mkdir()
    projections = [_projection_entry("ref", role="reference"), _projection_entry("aux")]

    manifest = pm.build_projection_manifest("ref", str(scene_dir), _GEOMETRY, projections)

    assert manifest["status"] == "complete"
    assert manifest["reference_photo_id"] == "ref"


def test_build_projection_manifest_incomplete_when_unresolved():
    manifest = pm.build_projection_manifest(
        "ref", None, _GEOMETRY, [_projection_entry("ref", role="reference")], unresolved_photo_ids=["aux"]
    )
    assert manifest["status"] == "incomplete"


def test_build_projection_manifest_incomplete_when_scene_not_saved():
    manifest = pm.build_projection_manifest(
        "ref", None, _GEOMETRY, [_projection_entry("ref", role="reference")]
    )
    assert manifest["status"] == "incomplete"


def test_projection_set_is_complete_matches_current_photo_set(tmp_path):
    scene_dir = tmp_path / "scene"
    scene_dir.mkdir()
    projections = [_projection_entry("ref", role="reference"), _projection_entry("aux")]
    manifest = pm.build_projection_manifest("ref", str(scene_dir), _GEOMETRY, projections)
    manifest_path = tmp_path / "projection_manifest.json"
    pm.save_projection_manifest(manifest, str(manifest_path))

    photo_set_manifest_path = tmp_path / "photo_set_manifest.json"
    photo_set_manifest_path.write_text(
        json.dumps({"reference_photo_id": "ref", "photos": [{"id": "ref"}, {"id": "aux"}]})
    )

    assert pm.projection_set_is_complete(str(manifest_path), str(photo_set_manifest_path))


def test_projection_set_is_complete_false_when_photo_set_changed(tmp_path):
    scene_dir = tmp_path / "scene"
    scene_dir.mkdir()
    projections = [_projection_entry("ref", role="reference")]
    manifest = pm.build_projection_manifest("ref", str(scene_dir), _GEOMETRY, projections)
    manifest_path = tmp_path / "projection_manifest.json"
    pm.save_projection_manifest(manifest, str(manifest_path))

    # A new auxiliary photo was added after the projection set was saved.
    photo_set_manifest_path = tmp_path / "photo_set_manifest.json"
    photo_set_manifest_path.write_text(
        json.dumps({"reference_photo_id": "ref", "photos": [{"id": "ref"}, {"id": "new_aux"}]})
    )

    assert not pm.projection_set_is_complete(str(manifest_path), str(photo_set_manifest_path))


def test_projection_set_is_complete_false_when_scene_missing(tmp_path):
    projections = [_projection_entry("ref", role="reference")]
    manifest = pm.build_projection_manifest("ref", str(tmp_path / "does_not_exist"), _GEOMETRY, projections)
    manifest_path = tmp_path / "projection_manifest.json"
    pm.save_projection_manifest(manifest, str(manifest_path))

    assert not pm.projection_set_is_complete(str(manifest_path))


def test_projection_set_is_complete_false_when_manifest_missing(tmp_path):
    assert not pm.projection_set_is_complete(str(tmp_path / "projection_manifest.json"))
