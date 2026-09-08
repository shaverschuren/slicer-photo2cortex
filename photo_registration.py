"""Photo-to-photo registration API and patient photo-set model.

This module prepares the architecture for a multi-photo workflow while keeping
all numerical registration and warping algorithms intentionally unimplemented.
The invariant is that there is exactly one reference photograph per patient and
all secondary photographs are mapped into that reference coordinate system.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional dependency in minimal environments
    yaml = None  # type: ignore


class PhotoRegistrationMethod(Enum):
    """Supported photo-to-reference registration methods."""

    RIGID_3DOF = "rigid_3dof"
    SIMILARITY_4DOF = "similarity_4dof"
    AFFINE_6DOF = "affine_6dof"
    PROJECTIVE_8DOF = "projective_8dof"
    NONLINEAR = "nonlinear"


@dataclass
class PhotoRegistrationResult:
    """Metadata describing a secondary-to-reference registration result."""

    reference_photo_id: str
    moving_photo_id: str
    method: PhotoRegistrationMethod
    transform_direction: str = "secondary_to_reference"
    status: str = "registered"
    source_image_path: Optional[str] = None
    reference_image_path: Optional[str] = None
    source_dimensions: Optional[Tuple[int, int]] = None
    reference_dimensions: Optional[Tuple[int, int]] = None
    registered_image_path: Optional[str] = None
    saved_path: Optional[str] = None
    transform: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PhotoRecord:
    """Metadata for one photograph in a patient photo set."""

    photo_id: str
    source_path: str
    role: str = "secondary"
    photo_type: Optional[str] = None
    registration_method: Optional[PhotoRegistrationMethod] = None
    registration_status: str = "pending"
    registration_result_path: Optional[str] = None
    registered_image_path: Optional[str] = None
    masks: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_reference(self) -> bool:
        return self.role == "reference"

    @property
    def is_secondary(self) -> bool:
        return self.role == "secondary"


@dataclass
class PatientPhotoSet:
    """The complete patient-level set of reference and secondary photographs."""

    patient_id: str
    reference_photo: PhotoRecord
    post_resection_photo: Optional[PhotoRecord] = None
    secondary_photos: List[PhotoRecord] = field(default_factory=list)
    manifest_path: Optional[str] = None

    def all_photos(self) -> List[PhotoRecord]:
        photos = [self.reference_photo]
        if self.post_resection_photo is not None:
            photos.append(self.post_resection_photo)
        return photos + list(self.secondary_photos)


def _normalise_photo_role(role: Optional[str]) -> str:
    if role is None:
        return "secondary"
    value = str(role).strip().lower()
    if value in {"reference", "fixed"}:
        return "reference"
    if value in {"secondary", "moving"}:
        return "secondary"
    return value


def _normalise_registration_method(method: Optional[Union[str, PhotoRegistrationMethod]]) -> PhotoRegistrationMethod:
    if method is None:
        return PhotoRegistrationMethod.PROJECTIVE_8DOF
    if isinstance(method, PhotoRegistrationMethod):
        return method
    if isinstance(method, str):
        value = method.strip().lower().replace("-", "_").replace(" ", "_")
        alias_map = {
            "rigid": PhotoRegistrationMethod.RIGID_3DOF,
            "rigid_3dof": PhotoRegistrationMethod.RIGID_3DOF,
            "similarity": PhotoRegistrationMethod.SIMILARITY_4DOF,
            "similarity_4dof": PhotoRegistrationMethod.SIMILARITY_4DOF,
            "affine": PhotoRegistrationMethod.AFFINE_6DOF,
            "affine_6dof": PhotoRegistrationMethod.AFFINE_6DOF,
            "projective": PhotoRegistrationMethod.PROJECTIVE_8DOF,
            "projective_8dof": PhotoRegistrationMethod.PROJECTIVE_8DOF,
            "homography": PhotoRegistrationMethod.PROJECTIVE_8DOF,
            "homography_8dof": PhotoRegistrationMethod.PROJECTIVE_8DOF,
            "nonlinear": PhotoRegistrationMethod.NONLINEAR,
            "deformable": PhotoRegistrationMethod.NONLINEAR,
            "deformable_registration": PhotoRegistrationMethod.NONLINEAR,
        }
        method_key = alias_map.get(value)
        if method_key is not None:
            return method_key
    raise ValueError(f"Unsupported photo registration method: {method!r}")


def _read_manifest_file(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Photo-set manifest not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()
    if path.lower().endswith(".json"):
        return json.loads(text)
    if yaml is not None:
        return yaml.safe_load(text) or {}
    raise RuntimeError("YAML support is unavailable and the manifest is not JSON.")


def _resolve_photo_path(source_path: str, base_dir: str) -> str:
    if os.path.isabs(source_path):
        return source_path
    candidate = os.path.join(base_dir, source_path)
    if os.path.exists(candidate):
        return candidate
    return source_path


def _manifest_entry_to_record(entry: Dict[str, Any], base_dir: str) -> PhotoRecord:
    photo_id = str(entry.get("id") or entry.get("photo_id") or os.path.splitext(os.path.basename(str(entry.get("path", ""))))[0] or "photo")
    source_path = _resolve_photo_path(str(entry.get("path") or entry.get("source_path") or entry.get("image_path") or ""), base_dir)
    if not source_path:
        raise ValueError(f"Manifest photo entry is missing a valid path: {entry!r}")
    role = _normalise_photo_role(entry.get("role"))
    method = _normalise_registration_method(entry.get("registration_method")) if entry.get("registration_method") else None
    status = str(entry.get("registration_status") or "pending")
    return PhotoRecord(
        photo_id=photo_id,
        source_path=source_path,
        role=role,
        photo_type=(entry.get("photo_type") or entry.get("type") or None),
        registration_method=method,
        registration_status=status,
        registration_result_path=(entry.get("registration_result_path") or None),
        registered_image_path=(entry.get("registered_image_path") or None),
        masks=dict(entry.get("masks") or {}),
        metadata={k: v for k, v in entry.items() if k not in {"id", "photo_id", "path", "source_path", "image_path", "role", "photo_type", "type", "registration_method", "registration_status", "registration_result_path", "registered_image_path", "masks"}},
    )


def _discover_files_in_directory(photo_dir: str) -> List[str]:
    extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    matches: List[str] = []
    for root, _, files in os.walk(photo_dir):
        for fname in files:
            if fname.lower().endswith(extensions):
                matches.append(os.path.join(root, fname))
    return sorted(matches)


def validate_photo_set(photo_set: PatientPhotoSet) -> bool:
    """Validate the patient photo set invariants."""

    if photo_set.reference_photo is None:
        raise ValueError("Patient photo set is missing a reference photo.")
    if not os.path.exists(photo_set.reference_photo.source_path):
        raise FileNotFoundError(f"Reference photo not found: {photo_set.reference_photo.source_path}")
    references = [photo for photo in photo_set.all_photos() if photo.is_reference]
    if len(references) != 1:
        raise ValueError(f"Patient photo set must contain exactly one reference photo. Found {len(references)}.")
    for photo in photo_set.secondary_photos:
        if not os.path.exists(photo.source_path):
            raise FileNotFoundError(f"Secondary photo not found: {photo.source_path}")
    return True


def discover_patient_photo_set(patient_id: str, patient_photo_dir: str, picture_root: Optional[str] = None, manifest_path: Optional[str] = None) -> PatientPhotoSet:
    """Discover and validate patient photos with single-photo backward compatibility.

    When only one photograph exists and no manifest was supplied, the photo is
    treated as the reference photograph. When multiple photographs exist without a
    manifest or explicit reference, this raises a clear ValueError instead of
    silently guessing.
    """

    if picture_root is None:
        picture_root = patient_photo_dir

    candidate_dirs = []
    patient_dir = os.path.join(picture_root, patient_id)
    if os.path.isdir(patient_dir):
        candidate_dirs.append(patient_dir)
    if os.path.isdir(patient_photo_dir):
        candidate_dirs.append(patient_photo_dir)
    seen_dirs = set()
    base_dir = None
    for directory in candidate_dirs:
        if directory not in seen_dirs:
            seen_dirs.add(directory)
            if os.path.isdir(directory):
                base_dir = directory
                break

    if base_dir is None:
        base_dir = patient_photo_dir

    manifest_candidates = []
    if manifest_path is not None:
        manifest_candidates.append(manifest_path)
    manifest_candidates.extend([
        os.path.join(base_dir, "photos.yaml"),
        os.path.join(base_dir, "photos.yml"),
        os.path.join(base_dir, "photos.json"),
        os.path.join(base_dir, patient_id, "photos.yaml"),
        os.path.join(base_dir, patient_id, "photos.json"),
    ])
    manifest_file = next((candidate for candidate in manifest_candidates if os.path.exists(candidate)), None)

    if manifest_file is not None:
        manifest = _read_manifest_file(manifest_file)
        photo_entries = manifest.get("photos") if isinstance(manifest.get("photos"), list) else []
        if not photo_entries:
            raise ValueError(f"Manifest '{manifest_file}' does not contain a valid 'photos' list.")

        photos = [_manifest_entry_to_record(entry, os.path.dirname(manifest_file)) for entry in photo_entries]
        refs = [photo for photo in photos if photo.is_reference]
        if len(refs) > 1:
            raise ValueError(f"Manifest '{manifest_file}' contains more than one reference photo. There must be exactly one reference.")

        reference_id = manifest.get("reference")
        if reference_id is not None:
            matching = [photo for photo in photos if photo.photo_id == str(reference_id)]
            if len(matching) != 1:
                raise ValueError(f"Manifest '{manifest_file}' references reference '{reference_id}', but it does not resolve to exactly one photograph.")
            reference_photo = matching[0]
            reference_photo.role = "reference"
            if len(refs) > 0 and refs[0].photo_id != reference_photo.photo_id:
                raise ValueError(f"Manifest '{manifest_file}' has multiple photos marked as reference. There must be exactly one reference photo.")
        else:
            if len(refs) == 1:
                reference_photo = refs[0]
            elif len(refs) > 1:
                raise ValueError(f"Manifest '{manifest_file}' contains more than one reference photo. There must be exactly one reference.")
            elif len(photos) == 1:
                photos[0].role = "reference"
                reference_photo = photos[0]
            else:
                raise ValueError("Multiple photographs found but no reference was specified. Add a 'reference' field or set one photo role to 'reference' in the manifest.")

        post_resection_photos = [photo for photo in photos if photo.photo_type in {"resection", "post_resection"} and photo.photo_id != reference_photo.photo_id]
        if len(post_resection_photos) > 1:
            raise ValueError(f"Manifest '{manifest_file}' contains more than one post-resection photo.")
        post_resection_photo = post_resection_photos[0] if post_resection_photos else None
        secondary_photos = [photo for photo in photos if photo.photo_id != reference_photo.photo_id and photo is not post_resection_photo]
        for photo in secondary_photos:
            photo.role = "secondary"
        if post_resection_photo is not None:
            post_resection_photo.role = "secondary"
        photo_set = PatientPhotoSet(patient_id=patient_id, reference_photo=reference_photo, post_resection_photo=post_resection_photo, secondary_photos=secondary_photos, manifest_path=manifest_file)
        validate_photo_set(photo_set)
        return photo_set

    photo_files = _discover_files_in_directory(base_dir)
    if not photo_files:
        raise FileNotFoundError(f"No photographs found for patient '{patient_id}' under '{base_dir}'.")
    if len(photo_files) == 1:
        photo = PhotoRecord(photo_id=os.path.splitext(os.path.basename(photo_files[0]))[0], source_path=photo_files[0], role="reference")
        return PatientPhotoSet(patient_id=patient_id, reference_photo=photo, post_resection_photo=None, secondary_photos=[], manifest_path=None)
    raise ValueError(
        f"Multiple photographs found for patient '{patient_id}' in '{base_dir}' and no explicit reference was specified. "
        "Add a photos.yaml/photos.json manifest with one photo marked as 'reference'."
    )


def build_photo_set_manifest(photo_set: PatientPhotoSet) -> Dict[str, Any]:
    """Create a normalized photo-set manifest for Slicer and other downstream consumers."""

    photos = []
    for photo in photo_set.all_photos():
        entry = {
            "id": photo.photo_id,
            "path": os.path.relpath(photo.source_path, os.path.dirname(photo_set.manifest_path or os.getcwd())) if photo_set.manifest_path else photo.source_path,
            "role": photo.role,
            "photo_type": photo.photo_type,
            "registration_method": photo.registration_method.value if isinstance(photo.registration_method, PhotoRegistrationMethod) else photo.registration_method,
            "registration_status": photo.registration_status,
            "registration_result_path": photo.registration_result_path,
            "registered_image_path": photo.registered_image_path,
            "masks": photo.masks,
        }
        photos.append({k: v for k, v in entry.items() if v is not None and not (k == "masks" and not v)})
    return {
        "patient_id": photo_set.patient_id,
        "reference_photo_id": photo_set.reference_photo.photo_id,
        "reference": photo_set.reference_photo.photo_id,
        "photos": photos,
    }


def save_photo_set_manifest(photo_set: PatientPhotoSet, manifest_path: str) -> str:
    """Persist a patient photo-set manifest to JSON or YAML."""

    payload = build_photo_set_manifest(photo_set)
    directory = os.path.dirname(manifest_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    if manifest_path.lower().endswith(".yaml") or manifest_path.lower().endswith(".yml"):
        if yaml is None:
            raise RuntimeError("PyYAML is required for YAML output.")
        with open(manifest_path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(payload, fh, sort_keys=False)
    else:
        with open(manifest_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
    return manifest_path


def _method_stub_doc(method_name: str, degrees: str) -> str:
    return (
        f"{method_name} is a placeholder for {degrees} photo-to-reference registration. "
        "The numerical implementation is intentionally not yet added."
    )


def register_rigid_3dof(moving_photo: Any, reference_photo: Any, **kwargs: Any) -> PhotoRegistrationResult:
    """Register a moving photo to a reference photo using rigid 3-DOF motion."""
    raise NotImplementedError(_method_stub_doc("register_rigid_3dof", "3 DOF rigid"))


def register_similarity_4dof(moving_photo: Any, reference_photo: Any, **kwargs: Any) -> PhotoRegistrationResult:
    """Register a moving photo to a reference photo using similarity 4-DOF motion."""
    raise NotImplementedError(_method_stub_doc("register_similarity_4dof", "4 DOF similarity"))


def register_affine_6dof(moving_photo: Any, reference_photo: Any, **kwargs: Any) -> PhotoRegistrationResult:
    """Register a moving photo to a reference photo using affine 6-DOF motion."""
    raise NotImplementedError(_method_stub_doc("register_affine_6dof", "6 DOF affine"))


def register_projective_8dof(moving_photo: Any, reference_photo: Any, **kwargs: Any) -> PhotoRegistrationResult:
    """Register a moving photo to a reference photo using feature-based homography."""
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required for projective photo registration.") from exc

    moving_path = str(getattr(moving_photo, "source_path", moving_photo))
    reference_path = str(getattr(reference_photo, "source_path", reference_photo))
    moving_image = cv2.imread(moving_path, cv2.IMREAD_COLOR)
    reference_image = cv2.imread(reference_path, cv2.IMREAD_COLOR)
    if moving_image is None or reference_image is None:
        raise ValueError(f"Could not read registration images: {moving_path}, {reference_path}")

    detector = cv2.SIFT_create() if hasattr(cv2, "SIFT_create") else cv2.ORB_create(nfeatures=5000)
    moving_keypoints, moving_descriptors = detector.detectAndCompute(cv2.cvtColor(moving_image, cv2.COLOR_BGR2GRAY), None)
    reference_keypoints, reference_descriptors = detector.detectAndCompute(cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY), None)
    if moving_descriptors is None or reference_descriptors is None:
        raise ValueError("Could not find usable features in one of the registration images.")

    norm = cv2.NORM_L2 if moving_descriptors.dtype == np.float32 else cv2.NORM_HAMMING
    matcher = cv2.BFMatcher(norm)
    matches = matcher.knnMatch(moving_descriptors, reference_descriptors, k=2)
    good_matches = [first for first, second in matches if first.distance < 0.75 * second.distance]
    if len(good_matches) < 4:
        raise ValueError(f"Only {len(good_matches)} reliable feature matches found; at least 4 are required.")

    moving_points = np.float32([moving_keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    reference_points = np.float32([reference_keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    transform, inlier_mask = cv2.findHomography(moving_points, reference_points, cv2.RANSAC, 5.0)
    if transform is None:
        raise ValueError("Could not estimate a projective transform from the matched features.")

    registered_image = cv2.warpPerspective(moving_image, transform, (reference_image.shape[1], reference_image.shape[0]))
    result = PhotoRegistrationResult(
        reference_photo_id=str(getattr(reference_photo, "photo_id", reference_path)),
        moving_photo_id=str(getattr(moving_photo, "photo_id", moving_path)),
        method=PhotoRegistrationMethod.PROJECTIVE_8DOF,
        source_image_path=moving_path,
        reference_image_path=reference_path,
        source_dimensions=(moving_image.shape[1], moving_image.shape[0]),
        reference_dimensions=(reference_image.shape[1], reference_image.shape[0]),
        transform=transform.tolist(),
        metadata={
            "match_count": len(good_matches),
            "inlier_count": int(inlier_mask.sum()) if inlier_mask is not None else None,
        },
    )
    output_path = kwargs.get("output_path")
    if output_path:
        if not cv2.imwrite(str(output_path), registered_image):
            raise OSError(f"Could not save registered image: {output_path}")
        result.registered_image_path = str(output_path)
    return result


def register_nonlinear(moving_photo: Any, reference_photo: Any, **kwargs: Any) -> PhotoRegistrationResult:
    """Register a moving photo to a reference photo using a nonlinear deformation model."""
    raise NotImplementedError(_method_stub_doc("register_nonlinear", "nonlinear"))


def register_photo_to_reference(moving_photo: Any, reference_photo: Any, method: Optional[Union[str, PhotoRegistrationMethod]] = None, **kwargs: Any) -> PhotoRegistrationResult:
    """Dispatch to the appropriate photo-to-reference registration implementation.

    The numerical algorithm is intentionally not implemented yet, but the routing
    and result metadata are prepared so the application can later save and reuse
    the registration result without changing the rest of the application.
    """

    method_enum = _normalise_registration_method(method)
    moving_id = str(getattr(moving_photo, "photo_id", moving_photo))
    reference_id = str(getattr(reference_photo, "photo_id", reference_photo))
    dispatcher = {
        PhotoRegistrationMethod.RIGID_3DOF: register_rigid_3dof,
        PhotoRegistrationMethod.SIMILARITY_4DOF: register_similarity_4dof,
        PhotoRegistrationMethod.AFFINE_6DOF: register_affine_6dof,
        PhotoRegistrationMethod.PROJECTIVE_8DOF: register_projective_8dof,
        PhotoRegistrationMethod.NONLINEAR: register_nonlinear,
    }[method_enum]
    try:
        return dispatcher(moving_photo, reference_photo, **kwargs)
    except NotImplementedError as exc:
        return PhotoRegistrationResult(
            reference_photo_id=reference_id,
            moving_photo_id=moving_id,
            method=method_enum,
            status="registration_pending",
            metadata={"error": str(exc), "transform_direction": "secondary_to_reference"},
        )


def register_photo_set(photo_set: PatientPhotoSet, output_dir: str, method: Optional[Union[str, PhotoRegistrationMethod]] = None) -> List[PhotoRegistrationResult]:
    """Register all non-reference photos into the reference grid."""
    os.makedirs(output_dir, exist_ok=True)
    results = []
    for photo in photo_set.all_photos()[1:]:
        output_path = os.path.join(output_dir, f"{photo.photo_id}_registered.png")
        try:
            result = register_photo_to_reference(
                photo,
                photo_set.reference_photo,
                method=method or PhotoRegistrationMethod.PROJECTIVE_8DOF,
                output_path=output_path,
            )
        except (ImportError, OSError, ValueError, RuntimeError) as exc:
            result = PhotoRegistrationResult(
                reference_photo_id=photo_set.reference_photo.photo_id,
                moving_photo_id=photo.photo_id,
                method=_normalise_registration_method(method),
                status="registration_pending",
                source_image_path=photo.source_path,
                reference_image_path=photo_set.reference_photo.source_path,
                metadata={"error": str(exc)},
            )
        photo.registration_method = result.method
        photo.registration_status = result.status
        photo.registered_image_path = result.registered_image_path
        photo.registration_result_path = os.path.join(output_dir, f"{photo.photo_id}_registration.json")
        save_registration_result(photo.registration_result_path, result)
        results.append(result)
    return results


def warp_image_to_reference(moving_image: Any, registration: PhotoRegistrationResult, **kwargs: Any) -> Any:
    """Warp an image into the reference-photo grid using the saved registration result."""
    raise NotImplementedError("warp_image_to_reference is not implemented yet; registration algorithms are intentionally deferred.")


def warp_mask_to_reference(mask: Any, registration: PhotoRegistrationResult, **kwargs: Any) -> Any:
    """Warp a mask into the reference-photo grid using the same registration result."""
    raise NotImplementedError("warp_mask_to_reference is not implemented yet; the API is prepared for later warping support.")


def transform_points_to_reference(points: Any, registration: PhotoRegistrationResult, **kwargs: Any) -> Any:
    """Transform points from secondary-photo coordinates to reference-photo coordinates."""
    raise NotImplementedError("transform_points_to_reference is not implemented yet; only the API contract is defined for now.")


def save_registration(path: str, reference_photo_id: str, moving_photo_id: str, method: Union[str, PhotoRegistrationMethod], metadata: Optional[Dict[str, Any]] = None, **kwargs: Any) -> str:
    """Save a registration result and related metadata to disk."""

    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    method_enum = _normalise_registration_method(method)
    payload = {
        "reference_photo_id": reference_photo_id,
        "moving_photo_id": moving_photo_id,
        "method": method_enum.value,
        "transform_direction": "secondary_to_reference",
        "status": "registered",
        "metadata": metadata or {},
    }
    payload.update(kwargs)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    return path


def load_registration(path: str) -> PhotoRegistrationResult:
    """Load a registration metadata file."""

    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    return PhotoRegistrationResult(
        reference_photo_id=str(payload.get("reference_photo_id") or payload.get("reference_id") or "unknown_reference"),
        moving_photo_id=str(payload.get("moving_photo_id") or payload.get("secondary_photo_id") or "unknown_moving"),
        method=_normalise_registration_method(payload.get("method", PhotoRegistrationMethod.RIGID_3DOF)),
        transform_direction=str(payload.get("transform_direction") or "secondary_to_reference"),
        status=str(payload.get("status") or "registered"),
        source_image_path=payload.get("source_image_path"),
        reference_image_path=payload.get("reference_image_path"),
        source_dimensions=tuple(payload.get("source_dimensions")) if payload.get("source_dimensions") else None,
        reference_dimensions=tuple(payload.get("reference_dimensions")) if payload.get("reference_dimensions") else None,
        registered_image_path=payload.get("registered_image_path"),
        saved_path=str(path),
        metadata=dict(payload.get("metadata") or {}),
        transform=payload.get("transform"),
    )


def load_photo_registration(path: Optional[str] = None, *, reference_photo_id: Optional[str] = None, moving_photo_id: Optional[str] = None, method: Optional[Union[str, PhotoRegistrationMethod]] = None, transform_direction: Optional[str] = None, status: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None, **kwargs: Any) -> PhotoRegistrationResult:
    """Load a saved registration or construct a result directly from metadata.

    The function accepts either a path to a saved registration metadata file or,
    when the caller already has the key values, a direct metadata-based result.
    """
    if path is None:
        return PhotoRegistrationResult(
            reference_photo_id=str(reference_photo_id or "unknown_reference"),
            moving_photo_id=str(moving_photo_id or "unknown_moving"),
            method=_normalise_registration_method(method) if method is not None else PhotoRegistrationMethod.RIGID_3DOF,
            transform_direction=str(transform_direction or "secondary_to_reference"),
            status=str(status or "registered"),
            metadata=dict(metadata or {}),
            source_image_path=kwargs.get("source_image_path"),
            reference_image_path=kwargs.get("reference_image_path"),
            source_dimensions=tuple(kwargs.get("source_dimensions")) if kwargs.get("source_dimensions") else None,
            reference_dimensions=tuple(kwargs.get("reference_dimensions")) if kwargs.get("reference_dimensions") else None,
            registered_image_path=kwargs.get("registered_image_path"),
            saved_path=kwargs.get("saved_path"),
            transform=kwargs.get("transform"),
        )
    return load_registration(path)


def save_registration_result(path: str, result: PhotoRegistrationResult) -> str:
    """Save a PhotoRegistrationResult object to disk."""

    save_registration(
        path,
        reference_photo_id=result.reference_photo_id,
        moving_photo_id=result.moving_photo_id,
        method=result.method,
        metadata=result.metadata,
        status=result.status,
        transform_direction=result.transform_direction,
        source_image_path=result.source_image_path,
        reference_image_path=result.reference_image_path,
        source_dimensions=result.source_dimensions,
        reference_dimensions=result.reference_dimensions,
        registered_image_path=result.registered_image_path,
        transform=result.transform,
    )
    return path


def discover_photo_set_manifest(patient_id: str, patient_photo_dir: str, manifest_path: Optional[str] = None) -> PatientPhotoSet:
    """Compatibility wrapper used by the workflow when the patient photo set is specified by manifest."""
    return discover_patient_photo_set(patient_id=patient_id, patient_photo_dir=patient_photo_dir, picture_root=patient_photo_dir, manifest_path=manifest_path)
