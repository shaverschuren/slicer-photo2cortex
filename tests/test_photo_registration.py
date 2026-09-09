"""Ordinary-Python tests for photo registration routing and mask warping. Claude-generated."""

import numpy as np
import pytest

from photo_preparation import PhotoRecord, PatientPhotoSet
from util import photo_registration as pr


def _fake_register_factory(reference_dimensions=None, transform=None):
    def fake_register(moving_photo, reference_photo, method=None, **kwargs):
        return pr.PhotoRegistrationResult(
            reference_photo_id=reference_photo.photo_id,
            moving_photo_id=moving_photo.photo_id,
            method=pr._normalise_registration_method(method),
            status="registered",
            reference_dimensions=reference_dimensions,
            transform=transform,
        )
    return fake_register


def test_register_photo_set_respects_per_photo_registration_method(monkeypatch, tmp_path):
    calls = {}

    def fake_register(moving_photo, reference_photo, method=None, **kwargs):
        calls[moving_photo.photo_id] = pr._normalise_registration_method(method)
        return pr.PhotoRegistrationResult(
            reference_photo_id=reference_photo.photo_id,
            moving_photo_id=moving_photo.photo_id,
            method=pr._normalise_registration_method(method),
            status="registered",
        )

    monkeypatch.setattr(pr, "register_photo_to_reference", fake_register)

    reference = PhotoRecord(photo_id="ref", source_path="ref.jpg", role="reference")
    aux_default = PhotoRecord(photo_id="aux_default", source_path="a.jpg", role="secondary")
    aux_explicit = PhotoRecord(
        photo_id="aux_explicit", source_path="b.jpg", role="secondary",
        registration_method=pr.PhotoRegistrationMethod.AFFINE_6DOF,
    )
    photo_set = PatientPhotoSet(
        patient_id="P1", reference_photo=reference, secondary_photos=[aux_default, aux_explicit]
    )

    pr.register_photo_set(photo_set, str(tmp_path), method=pr.PhotoRegistrationMethod.SIMILARITY_4DOF)

    # A photo without its own registration_method falls back to the caller's default...
    assert calls["aux_default"] == pr.PhotoRegistrationMethod.SIMILARITY_4DOF
    # ...but a per-photo method always takes precedence over that default.
    assert calls["aux_explicit"] == pr.PhotoRegistrationMethod.AFFINE_6DOF


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


def test_warp_mask_to_reference_rejects_non_projective_method():
    result = pr.PhotoRegistrationResult(
        reference_photo_id="ref", moving_photo_id="m",
        method=pr.PhotoRegistrationMethod.AFFINE_6DOF, status="registered",
    )
    with pytest.raises(NotImplementedError):
        pr.warp_mask_to_reference(np.zeros((4, 4), dtype=bool), result)


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
