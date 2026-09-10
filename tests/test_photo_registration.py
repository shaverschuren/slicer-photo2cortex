"""Ordinary-Python tests for photo registration routing and mask warping. Claude-generated."""

import os
import numpy as np
import pytest

from photo_preparation import PhotoRecord, PatientPhotoSet
from util import photo_registration as pr


def _fake_register_factory(reference_dimensions=None, transform=None):
    def fake_register(moving_photo, reference_photo, dof=6, **kwargs):
        return pr.PhotoRegistrationResult(
            reference_photo_id=reference_photo.photo_id,
            moving_photo_id=moving_photo.photo_id,
            method=pr._method_from_dof(dof),
            status="registered",
            reference_dimensions=reference_dimensions,
            transform=transform,
        )
    return fake_register


def test_register_photo_set_uses_default_dof(monkeypatch, tmp_path):
    calls = {}

    def fake_register(moving_photo, reference_photo, dof=6, **kwargs):
        calls[moving_photo.photo_id] = dof
        return pr.PhotoRegistrationResult(
            reference_photo_id=reference_photo.photo_id,
            moving_photo_id=moving_photo.photo_id,
            method=pr._method_from_dof(dof),
            status="registered",
        )

    monkeypatch.setattr(pr, "register_photo_to_reference", fake_register)

    reference = PhotoRecord(photo_id="ref", source_path="ref.jpg", role="reference")
    aux_default = PhotoRecord(photo_id="aux_default", source_path="a.jpg", role="secondary")
    aux_explicit = PhotoRecord(photo_id="aux_explicit", source_path="b.jpg", role="secondary", registration_dof=3)
    photo_set = PatientPhotoSet(
        patient_id="P1", reference_photo=reference, secondary_photos=[aux_default, aux_explicit]
    )

    pr.register_photo_set(photo_set, str(tmp_path))

    # A photo without its own registration_method falls back to the caller's default...
    assert calls["aux_default"] == 6
    assert calls["aux_explicit"] == 3


def test_warp_mask_to_reference_identity_transform():
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:5, 2:5] = True
    result = pr.PhotoRegistrationResult(
        reference_photo_id="ref", moving_photo_id="m",
        method=pr.PhotoRegistrationMethod.PROJECTIVE_8DOF, status="registered",
        reference_dimensions=(10, 10), transform=np.eye(3).tolist(),
    )
    warped = pr.warp_mask_to_reference(mask, result)
    assert warped.dtype == bool
    assert np.array_equal(warped, mask)


def test_warp_mask_to_reference_accepts_similarity_transform():
    result = pr.PhotoRegistrationResult(
        reference_photo_id="ref", moving_photo_id="m",
        method=pr.PhotoRegistrationMethod.SIMILARITY_4DOF, status="registered",
        reference_dimensions=(4, 4), transform=np.eye(3).tolist(),
    )
    assert pr.warp_mask_to_reference(np.zeros((4, 4), dtype=bool), result).shape == (4, 4)


def test_warp_mask_to_reference_accepts_affine_ecc_transform():
    result = pr.PhotoRegistrationResult(
        reference_photo_id="ref", moving_photo_id="m",
        method=pr.PhotoRegistrationMethod.AFFINE_6DOF, status="registered",
        reference_dimensions=(4, 4), transform=np.eye(3).tolist(),
    )
    assert pr.warp_mask_to_reference(np.zeros((4, 4), dtype=bool), result).shape == (4, 4)


def test_register_photo_set_warps_masks_with_nearest_neighbour(monkeypatch, tmp_path):
    mask_path = tmp_path / "aux_masks.npz"
    outside_mask = np.zeros((8, 8), dtype=bool)
    outside_mask[0:2, :] = True
    resection_mask = np.zeros((8, 8), dtype=bool)
    resection_mask[5, 5] = True
    np.savez_compressed(mask_path, resection_mask=resection_mask, outside_mask=outside_mask)

    monkeypatch.setattr(
        pr, "register_photo_to_reference",
        _fake_register_factory(reference_dimensions=(8, 8), transform=np.eye(3).tolist()),
    )

    reference = PhotoRecord(photo_id="ref", source_path="ref.jpg", role="reference")
    aux = PhotoRecord(photo_id="aux", source_path="a.jpg", role="secondary")
    aux.masks["path"] = str(mask_path)
    photo_set = PatientPhotoSet(patient_id="P1", reference_photo=reference, secondary_photos=[aux])

    pr.register_photo_set(photo_set, str(tmp_path))

    assert "registered_path" in aux.masks
    with np.load(aux.masks["registered_path"]) as warped:
        assert np.array_equal(warped["outside_mask"], outside_mask)
        assert np.array_equal(warped["resection_mask"], resection_mask)


def test_register_photo_set_does_not_warp_masks_when_registration_pending(monkeypatch, tmp_path):
    mask_path = tmp_path / "aux_masks.npz"
    np.savez_compressed(mask_path, resection_mask=np.zeros((4, 4), bool), outside_mask=np.zeros((4, 4), bool))

    def fake_register(moving_photo, reference_photo, method=None, **kwargs):
        return pr.PhotoRegistrationResult(
            reference_photo_id=reference_photo.photo_id, moving_photo_id=moving_photo.photo_id,
            method=pr._normalise_registration_method(method), status="registration_pending",
        )
    monkeypatch.setattr(pr, "register_photo_to_reference", fake_register)

    reference = PhotoRecord(photo_id="ref", source_path="ref.jpg", role="reference")
    aux = PhotoRecord(photo_id="aux", source_path="a.jpg", role="secondary")
    aux.masks["path"] = str(mask_path)
    photo_set = PatientPhotoSet(patient_id="P1", reference_photo=reference, secondary_photos=[aux])

    pr.register_photo_set(photo_set, str(tmp_path))

    assert aux.registration_status == "registration_pending"
    assert "registered_path" not in aux.masks


def test_register_photo_set_reuses_valid_cached_registration(monkeypatch, tmp_path):
    from PIL import Image

    reference_path = tmp_path / "ref.jpg"
    moving_path = tmp_path / "aux.jpg"
    Image.new("RGB", (8, 8), color="white").save(reference_path)
    Image.new("RGB", (8, 8), color="white").save(moving_path)
    reference_mask_path = tmp_path / "ref_masks.npz"
    moving_mask_path = tmp_path / "aux_masks.npz"
    np.savez_compressed(reference_mask_path, outside_mask=np.zeros((8, 8), bool))
    np.savez_compressed(moving_mask_path, outside_mask=np.zeros((8, 8), bool))
    reference = PhotoRecord(photo_id="ref", source_path=str(reference_path), role="reference")
    aux = PhotoRecord(
        photo_id="aux", source_path=str(moving_path), role="secondary",
        registration_dof=8,
        registration_status="registered",
        registration_qc_status="approved",
        registration_qc_reviewed_at="2026-09-09T14:12:34Z",
    )
    reference.masks["path"] = str(reference_mask_path)
    aux.masks["path"] = str(moving_mask_path)
    registered_image_path = tmp_path / "aux_registered.png"
    registered_image_path.write_bytes(b"registered")
    result_path = tmp_path / "aux_registration.json"
    cached_result = pr.PhotoRegistrationResult(
        reference_photo_id="ref",
        moving_photo_id="aux",
        method=pr.PhotoRegistrationMethod.PROJECTIVE_8DOF,
        status="registered",
        source_image_path="aux.jpg",
        reference_image_path="ref.jpg",
        registered_image_path=str(registered_image_path),
        reference_dimensions=(8, 8),
        transform=np.eye(3).tolist(),
    )
    cached_result.metadata.update({"backend": "opencv_ecc", "dof": 8, "algorithm_version": pr.REGISTRATION_ALGORITHM_VERSION, "cache_signature": pr._registration_cache_signature(aux, reference, 8)})
    pr.save_registration_result(str(result_path), cached_result)

    def fail_if_called(*args, **kwargs):
        raise AssertionError("a valid cached registration should be reused")

    monkeypatch.setattr(pr, "register_photo_to_reference", fail_if_called)

    aux.registration_result_path = str(result_path)
    aux.registered_image_path = str(registered_image_path)
    photo_set = PatientPhotoSet(patient_id="P1", reference_photo=reference, secondary_photos=[aux])

    results = pr.register_photo_set(photo_set, str(tmp_path))

    assert len(results) == 1
    assert results[0].status == "registered"
    assert results[0].moving_photo_id == "aux"
    assert aux.registration_qc_status == "approved"
    assert aux.registration_qc_reviewed_at == "2026-09-09T14:12:34Z"


def test_register_photo_set_resets_qc_state_when_registration_is_recomputed(monkeypatch, tmp_path):
    reference = PhotoRecord(photo_id="ref", source_path="ref.jpg", role="reference")
    aux = PhotoRecord(
        photo_id="aux",
        source_path="aux.jpg",
        role="secondary",
        registration_qc_status="approved",
        registration_qc_reviewed_at="2026-09-09T14:12:34Z",
    )
    photo_set = PatientPhotoSet(patient_id="P1", reference_photo=reference, secondary_photos=[aux])

    def fake_register(moving_photo, reference_photo, method=None, **kwargs):
        return pr.PhotoRegistrationResult(
            reference_photo_id=reference_photo.photo_id,
            moving_photo_id=moving_photo.photo_id,
            method=pr._normalise_registration_method(method),
            status="registered",
            registered_image_path=str(tmp_path / "aux_registered.png"),
        )

    monkeypatch.setattr(pr, "register_photo_to_reference", fake_register)
    pr.register_photo_set(photo_set, str(tmp_path))

    assert aux.registration_qc_status == "pending"
    assert aux.registration_qc_reviewed_at is None


def test_build_registration_roi_excludes_outside_and_resection():
    outside = np.zeros((5, 6), dtype=bool)
    outside[0, :] = True
    resection = np.zeros_like(outside)
    resection[2, 3] = True
    roi = pr.build_registration_roi(outside, resection)
    assert not roi[0, 0]
    assert not roi[2, 3]
    assert roi[2, 2]


def test_load_registration_roi_rejects_mask_shape_mismatch(tmp_path):
    from PIL import Image

    image_path = tmp_path / "photo.jpg"
    mask_path = tmp_path / "photo_masks.npz"
    Image.new("RGB", (8, 6), color="white").save(image_path)
    np.savez_compressed(mask_path, outside_mask=np.zeros((5, 8), dtype=bool))
    photo = PhotoRecord(photo_id="photo", source_path=str(image_path), masks={"path": str(mask_path)})
    with pytest.raises(ValueError, match="does not match source image shape"):
        pr.load_registration_roi(photo, (6, 8, 3))


def test_ecc_dof_routing_matches_opencv_constants():
    import cv2

    assert pr.DEFAULT_REGISTRATION_DOF == 6
    assert pr.ecc_motion_type(2) == cv2.MOTION_TRANSLATION
    assert pr.ecc_motion_type(3) == cv2.MOTION_EUCLIDEAN
    assert pr.ecc_motion_type(6) == cv2.MOTION_AFFINE
    assert pr.ecc_motion_type(8) == cv2.MOTION_HOMOGRAPHY


def test_ecc_initialization_allows_large_rotation_and_resolution_scale():
    moving_shape = (100, 100, 3)
    reference_shape = (200, 200, 3)
    moving_roi = np.zeros(moving_shape[:2], dtype=bool)
    reference_roi = np.zeros(reference_shape[:2], dtype=bool)
    moving_roi[25:75, 25:75] = True
    reference_roi[50:150, 50:150] = True
    angle = np.deg2rad(170.0)
    scale = 2.0
    linear = scale * np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    translation = np.array([99.5, 99.5]) - linear @ np.array([49.5, 49.5])
    initial = pr.initial_warp_for_dof(6, moving_shape, reference_shape, np.array([49.5, 49.5]), np.array([99.5, 99.5]), 170)
    assert initial.shape == (2, 3)
    assert initial[0, 0] == pytest.approx(linear[0, 0], abs=0.1)


def test_similarity_validation_rejects_scale_reflection_and_centre_jump():
    roi = np.ones((100, 100), dtype=bool)
    too_large = pr._validate_ecc_transform(np.array([[3, 0, -100], [0, 3, -100], [0, 0, 1]], dtype=float), (100, 100), (100, 100), roi, roi, 6)
    assert not too_large["valid"]
    assert "scale" in too_large["error"]
    reflected = pr._validate_ecc_transform(np.array([[-1, 0, 99], [0, 1, 0], [0, 0, 1]], dtype=float), (100, 100), (100, 100), roi, roi, 6)
    assert not reflected["valid"]
    assert not reflected["orientation_preserving"]
    displaced = pr._validate_ecc_transform(np.array([[1, 0, 90], [0, 1, 90], [0, 0, 1]], dtype=float), (100, 100), (100, 100), roi, roi, 3)
    assert not displaced["valid"]
    assert "centre" in displaced["error"]


def test_registration_cache_signature_changes_when_mask_changes(tmp_path):
    from PIL import Image

    image_path = tmp_path / "photo.jpg"
    Image.new("RGB", (8, 8), color="white").save(image_path)
    mask_path = tmp_path / "masks.npz"
    np.savez_compressed(mask_path, outside_mask=np.zeros((8, 8), dtype=bool))
    reference = PhotoRecord(photo_id="ref", source_path=str(image_path), role="reference", masks={"path": str(mask_path)})
    moving = PhotoRecord(photo_id="moving", source_path=str(image_path), masks={"path": str(mask_path)})
    first = pr._registration_cache_signature(moving, reference, pr.PhotoRegistrationMethod.SIMILARITY_4DOF)
    np.savez_compressed(mask_path, outside_mask=np.ones((8, 8), dtype=bool))
    stat = mask_path.stat()
    mask_path.touch()
    os.utime(mask_path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    second = pr._registration_cache_signature(moving, reference, pr.PhotoRegistrationMethod.SIMILARITY_4DOF)
    assert first != second


def test_ecc_registration_uses_both_masks_and_all_rotation_starts(monkeypatch, tmp_path):
    import cv2
    from PIL import Image

    reference_path = tmp_path / "reference.png"
    moving_path = tmp_path / "moving.png"
    Image.new("RGB", (100, 100), color="gray").save(reference_path)
    Image.new("RGB", (100, 100), color="gray").save(moving_path)
    reference_mask = tmp_path / "reference.npz"
    moving_mask = tmp_path / "moving.npz"
    np.savez_compressed(reference_mask, outside_mask=np.zeros((100, 100), bool))
    np.savez_compressed(moving_mask, outside_mask=np.zeros((100, 100), bool), resection_mask=np.zeros((100, 100), bool))
    reference = PhotoRecord("reference", str(reference_path), role="reference", masks={"path": str(reference_mask)})
    moving = PhotoRecord("moving", str(moving_path), masks={"path": str(moving_mask)})
    calls = []

    def fake_ecc(template, input_image, template_mask, input_mask, warp, motion_type):
        calls.append((template_mask.copy(), input_mask.copy(), motion_type, warp.copy()))
        return 0.9, np.eye(2, 3, dtype=np.float32)

    monkeypatch.setattr(pr, "_ecc_with_mask", fake_ecc)
    result = pr.register_ecc(moving, reference, dof=6)

    assert len(calls) == 8
    assert all(call[0].dtype == np.uint8 and call[1].dtype == np.uint8 for call in calls)
    assert all(call[2] == cv2.MOTION_AFFINE for call in calls)
    assert result.metadata["backend"] == "opencv_ecc"
    assert result.metadata["dof"] == 6


def _make_registration_photos(tmp_path, moving_image, reference_image):
    import cv2

    moving_path = tmp_path / "moving.png"
    reference_path = tmp_path / "reference.png"
    assert cv2.imwrite(str(moving_path), moving_image)
    assert cv2.imwrite(str(reference_path), reference_image)
    moving_mask_path = tmp_path / "moving_masks.npz"
    reference_mask_path = tmp_path / "reference_masks.npz"
    np.savez_compressed(moving_mask_path, outside_mask=np.zeros(moving_image.shape[:2], dtype=bool))
    np.savez_compressed(reference_mask_path, outside_mask=np.zeros(reference_image.shape[:2], dtype=bool))
    return (
        PhotoRecord("moving", str(moving_path), masks={"path": str(moving_mask_path)}),
        PhotoRecord("reference", str(reference_path), role="reference", masks={"path": str(reference_mask_path)}),
    )


def _asymmetric_image(size=160):
    import cv2

    image = np.zeros((size, size, 3), dtype=np.uint8)
    cv2.circle(image, (35, 42), 13, (220, 90, 30), -1)
    cv2.rectangle(image, (96, 25), (132, 58), (40, 190, 240), -1)
    cv2.fillPoly(image, [np.array([[50, 108], [78, 130], [32, 140]])], (170, 50, 200))
    cv2.line(image, (102, 105), (135, 137), (255, 255, 255), 5)
    return cv2.GaussianBlur(image, (5, 5), 0)


def test_ecc_registration_returns_moving_to_reference_translation(monkeypatch, tmp_path):
    import cv2

    moving_image = _asymmetric_image()
    known = np.array([[1, 0, 7], [0, 1, -5]], dtype=np.float32)
    reference_image = cv2.warpAffine(moving_image, known, (160, 160))
    moving, reference = _make_registration_photos(tmp_path, moving_image, reference_image)
    monkeypatch.setattr(pr, "ECC_INITIAL_ROTATIONS_DEG", (0,))

    result = pr.register_photo_to_reference(moving, reference, dof=6)
    mapped = pr.transform_points_to_reference([[35, 42]], result)[0]

    assert mapped == pytest.approx([42, 37], abs=1.5)
    warped = pr.warp_image_to_reference(moving_image, result)
    assert np.mean(np.abs(warped.astype(float) - reference_image.astype(float))) < 8


def test_ecc_registration_returns_moving_to_reference_affine(monkeypatch, tmp_path):
    import cv2

    moving_image = _asymmetric_image()
    known = cv2.getRotationMatrix2D((80, 80), 12, 1.0).astype(np.float32)
    known[:, 2] += [4, -3]
    reference_image = cv2.warpAffine(moving_image, known, (160, 160))
    moving, reference = _make_registration_photos(tmp_path, moving_image, reference_image)
    monkeypatch.setattr(pr, "ECC_INITIAL_ROTATIONS_DEG", (0,))

    result = pr.register_photo_to_reference(moving, reference, dof=6)
    point = np.array([35, 42, 1], dtype=float)
    expected = (np.vstack([known, [0, 0, 1]]) @ point)[:2]

    assert pr.transform_points_to_reference([point[:2]], result)[0] == pytest.approx(expected, abs=2.0)


@pytest.mark.parametrize("full_transform", [
    np.array([[1.1, 0.15, 40], [-0.08, 0.9, -20], [0, 0, 1]], dtype=float),
    np.array([[1.0, 0.05, 40], [-0.03, 0.95, -20], [0.0002, -0.0001, 1]], dtype=float),
])
def test_coarse_to_full_transform_preserves_moving_to_reference_points(full_transform):
    moving_shape = (1200, 1600, 3)
    reference_shape = (900, 1200, 3)
    moving_width, moving_height = pr._coarse_shape(moving_shape)
    reference_width, reference_height = pr._coarse_shape(reference_shape)
    moving_scale = np.diag([moving_width / moving_shape[1], moving_height / moving_shape[0], 1.0])
    reference_scale = np.diag([reference_width / reference_shape[1], reference_height / reference_shape[0], 1.0])
    coarse_transform = reference_scale @ full_transform @ np.linalg.inv(moving_scale)

    recovered = pr._to_full_resolution_transform(coarse_transform, moving_shape, reference_shape)
    point = np.array([340, 520, 1.0])
    assert (recovered @ point) / (recovered @ point)[2] == pytest.approx((full_transform @ point) / (full_transform @ point)[2])
