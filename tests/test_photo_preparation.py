"""Ordinary-Python tests for the photo-set data model and manifest round trip. Claude-generated."""

import json

import photo_preparation as pp


def test_all_photos_iterates_reference_post_resection_secondary_exactly_once():
    reference = pp.PhotoRecord(photo_id="ref", source_path="ref.jpg", role="reference")
    post_resection = pp.PhotoRecord(photo_id="post", source_path="post.jpg", role="secondary", photo_type="post_resection")
    aux = pp.PhotoRecord(photo_id="aux", source_path="aux.jpg", role="secondary")
    photo_set = pp.PatientPhotoSet(
        patient_id="P1", reference_photo=reference, post_resection_photo=post_resection, secondary_photos=[aux]
    )

    photos = photo_set.all_photos()
    assert [p.photo_id for p in photos] == ["ref", "post", "aux"]
    # Every photo appears exactly once (no aliasing between reference/post-resection/secondary)
    assert len(photos) == len({id(p) for p in photos})


def test_all_photos_without_post_resection_or_secondary():
    reference = pp.PhotoRecord(photo_id="ref", source_path="ref.jpg", role="reference")
    photo_set = pp.PatientPhotoSet(patient_id="P1", reference_photo=reference)
    assert [p.photo_id for p in photo_set.all_photos()] == ["ref"]


def test_reference_and_secondary_role_flags():
    reference = pp.PhotoRecord(photo_id="ref", source_path="ref.jpg", role="reference")
    aux = pp.PhotoRecord(photo_id="aux", source_path="aux.jpg", role="secondary")
    assert reference.is_reference and not reference.is_secondary
    assert aux.is_secondary and not aux.is_reference


def test_manifest_round_trip(tmp_path):
    (tmp_path / "ref.jpg").write_bytes(b"x")
    (tmp_path / "aux.jpg").write_bytes(b"x")

    reference = pp.PhotoRecord(photo_id="ref", source_path=str(tmp_path / "ref.jpg"), role="reference")
    aux = pp.PhotoRecord(
        photo_id="aux",
        source_path=str(tmp_path / "aux.jpg"),
        role="secondary",
        registration_method="projective_8dof",
        registration_status="registered",
        registered_image_path=str(tmp_path / "aux_registered.png"),
    )
    aux.masks["path"] = str(tmp_path / "aux_masks.npz")
    aux.masks["registered_path"] = str(tmp_path / "aux_registered_masks.npz")
    photo_set = pp.PatientPhotoSet(patient_id="P1", reference_photo=reference, secondary_photos=[aux])

    manifest_path = tmp_path / "photo_set_manifest.json"
    pp.save_photo_set_manifest(photo_set, str(manifest_path))

    with open(manifest_path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    assert raw["reference_photo_id"] == "ref"
    assert len(raw["photos"]) == 2

    reloaded = pp.discover_patient_photo_set(
        patient_id="P1", patient_photo_dir=str(tmp_path), manifest_path=str(manifest_path)
    )

    assert reloaded.reference_photo.photo_id == "ref"
    assert len(reloaded.secondary_photos) == 1
    reloaded_aux = reloaded.secondary_photos[0]
    assert reloaded_aux.registration_status == "registered"
    assert reloaded_aux.registered_image_path == aux.registered_image_path
    assert reloaded_aux.masks.get("path") == aux.masks["path"]
    assert reloaded_aux.masks.get("registered_path") == aux.masks["registered_path"]


def test_manifest_round_trip_keeps_registration_qc_fields(tmp_path):
    (tmp_path / "ref.jpg").write_bytes(b"x")
    (tmp_path / "aux.jpg").write_bytes(b"x")

    reference = pp.PhotoRecord(photo_id="ref", source_path=str(tmp_path / "ref.jpg"), role="reference")
    aux = pp.PhotoRecord(
        photo_id="aux",
        source_path=str(tmp_path / "aux.jpg"),
        role="secondary",
        registration_status="registered",
        registration_qc_status="approved",
        registration_qc_reviewed_at="2026-09-09T14:12:34Z",
    )
    photo_set = pp.PatientPhotoSet(patient_id="P1", reference_photo=reference, secondary_photos=[aux])
    manifest_path = tmp_path / "photo_set_manifest.json"
    pp.save_photo_set_manifest(photo_set, str(manifest_path))

    reloaded = pp.discover_patient_photo_set(patient_id="P1", patient_photo_dir=str(tmp_path), manifest_path=str(manifest_path))
    assert reloaded.reference_photo.registration_qc_status == "not_required"
    assert reloaded.secondary_photos[0].registration_qc_status == "approved"
    assert reloaded.secondary_photos[0].registration_qc_reviewed_at == "2026-09-09T14:12:34Z"


def test_manifest_with_multiple_reference_photos_raises(tmp_path):
    (tmp_path / "a.jpg").write_bytes(b"x")
    (tmp_path / "b.jpg").write_bytes(b"x")
    manifest_path = tmp_path / "photo_set_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "photos": [
                    {"id": "a", "path": "a.jpg", "role": "reference"},
                    {"id": "b", "path": "b.jpg", "role": "reference"},
                ]
            }
        )
    )
    try:
        pp.discover_patient_photo_set(patient_id="P1", patient_photo_dir=str(tmp_path), manifest_path=str(manifest_path))
        assert False, "expected ValueError for multiple reference photos"
    except ValueError:
        pass
