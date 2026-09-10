"""Photo-to-photo registration algorithms and result persistence.

File discovery, manifest IO, and the `PhotoRecord`/`PatientPhotoSet` data model
live in `photo_preparation`. This module only concerns itself with registering
a secondary photograph into the reference-photo coordinate system and with
saving/loading the resulting transforms.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from photo_preparation import PatientPhotoSet


REGISTRATION_ALGORITHM_VERSION = "opencv2_masked_ecc"
DEFAULT_REGISTRATION_DOF = 6
ECC_MOTION_BY_DOF = {2: "MOTION_TRANSLATION", 3: "MOTION_EUCLIDEAN", 6: "MOTION_AFFINE", 8: "MOTION_HOMOGRAPHY"}
ECC_INITIAL_ROTATIONS_DEG = (0, 45, 90, 135, 180, 225, 270, 315)
ECC_MAX_ITERATIONS = 100
ECC_EPSILON = 1e-6
ECC_GAUSS_FILTER_SIZE = 5
ECC_COARSE_MAX_DIM = 768
MIN_VALID_ROI_PIXELS = 64
MIN_ROI_OVERLAP_FRACTION = 0.05
MAX_AFFINE_ANISOTROPY = 1.30
MIN_RELATIVE_SCALE = 0.50
MAX_RELATIVE_SCALE = 2.00
MAX_CENTRE_OFFSET_FRACTION = 0.50


def set_registration_qc_pending(photo: Any) -> None:
    """Reset QC status for a freshly recomputed registration so it must be reviewed again."""
    photo.registration_qc_status = "pending"
    photo.registration_qc_reviewed_at = None


def registration_qc_is_approved(photo_set: PatientPhotoSet) -> bool:
    """Return True only when every selected auxiliary registration has been approved."""
    for photo in photo_set.all_photos()[1:]:
        if photo.registration_status != "registered":
            return False
        if photo.registration_qc_status != "approved":
            return False
    return True


class PhotoRegistrationMethod(Enum):
    """Supported photo-to-reference registration methods."""

    RIGID_3DOF = "rigid_3dof"
    SIMILARITY_4DOF = "similarity_4dof"
    AFFINE_6DOF = "affine_6dof"
    PROJECTIVE_8DOF = "projective_8dof"
    NONLINEAR = "nonlinear"


def validate_registration_dof(dof: int) -> int:
    try:
        value = int(dof)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Unsupported registration DOF {dof!r}. Supported ECC models are 2, 3, 6, and 8 DOF.") from exc
    if value not in {2, 3, 6, 8}:
        raise ValueError(f"Unsupported registration DOF {value}. Supported ECC models are 2, 3, 6, and 8 DOF.")
    return value


def ecc_motion_type(dof: int) -> int:
    import cv2
    return {
        2: cv2.MOTION_TRANSLATION,
        3: cv2.MOTION_EUCLIDEAN,
        6: cv2.MOTION_AFFINE,
        8: cv2.MOTION_HOMOGRAPHY,
    }[validate_registration_dof(dof)]


def _method_from_dof(dof: int) -> PhotoRegistrationMethod:
    return {2: PhotoRegistrationMethod.RIGID_3DOF, 3: PhotoRegistrationMethod.RIGID_3DOF, 6: PhotoRegistrationMethod.AFFINE_6DOF, 8: PhotoRegistrationMethod.PROJECTIVE_8DOF}[validate_registration_dof(dof)]


def _dof_from_method(method: Any) -> int:
    return {
        PhotoRegistrationMethod.RIGID_3DOF: 3,
        PhotoRegistrationMethod.SIMILARITY_4DOF: 6,
        PhotoRegistrationMethod.AFFINE_6DOF: 6,
        PhotoRegistrationMethod.PROJECTIVE_8DOF: 8,
    }[_normalise_registration_method(method)]


def _normalise_registration_method(method: Optional[Union[str, PhotoRegistrationMethod]]) -> PhotoRegistrationMethod:
    if method is None:
        return PhotoRegistrationMethod.AFFINE_6DOF
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


def build_registration_roi(outside_mask: Any, resection_mask: Any = None) -> np.ndarray:
    """Return the intact exposed-field ROI from source-space masks."""
    outside = np.asarray(outside_mask, dtype=bool)
    if outside.ndim != 2:
        raise ValueError(f"outside_mask must be a 2D array; got shape {outside.shape}.")
    if resection_mask is None:
        resection = np.zeros_like(outside)
    else:
        resection = np.asarray(resection_mask, dtype=bool)
        if resection.shape != outside.shape:
            raise ValueError(f"resection_mask shape {resection.shape} does not match outside_mask shape {outside.shape}.")
    return np.logical_not(outside) & np.logical_not(resection)


def load_registration_roi(photo: Any, image_shape: Tuple[int, ...]) -> np.ndarray:
    """Load a photo's source-space masks and build its intact-cortex ROI."""
    expected_shape = tuple(image_shape[:2])
    masks = getattr(photo, "masks", None) or {}
    mask_path = masks.get("path")
    if not mask_path or not os.path.exists(mask_path):
        raise ValueError(f"Registration mask file is required for '{getattr(photo, 'photo_id', photo)}'.")
    try:
        with np.load(mask_path) as payload:
            if "outside_mask" not in payload.files:
                raise ValueError(f"Registration mask file '{mask_path}' is missing outside_mask.")
            outside_mask = np.asarray(payload["outside_mask"], dtype=bool)
            resection_mask = payload["resection_mask"] if "resection_mask" in payload.files else None
    except (OSError, ValueError) as exc:
        raise ValueError(f"Could not load registration masks from '{mask_path}': {exc}") from exc
    if outside_mask.shape != expected_shape:
        raise ValueError(f"outside_mask shape {outside_mask.shape} does not match source image shape {expected_shape}.")
    if resection_mask is not None and np.asarray(resection_mask).shape != expected_shape:
        raise ValueError(f"resection_mask shape {np.asarray(resection_mask).shape} does not match source image shape {expected_shape}.")
    return build_registration_roi(outside_mask, resection_mask)


def _coarse_shape(shape: Tuple[int, ...]) -> Tuple[int, int]:
    scale = min(1.0, ECC_COARSE_MAX_DIM / float(max(shape[:2])))
    return max(1, int(round(shape[1] * scale))), max(1, int(round(shape[0] * scale)))


def initial_warp_for_dof(dof: int, moving_shape: Tuple[int, ...], reference_shape: Tuple[int, ...], moving_centre: Any, reference_centre: Any, rotation_deg: float = 0.0) -> np.ndarray:
    """Create a coarse moving-to-reference initialization in project convention."""
    import cv2
    value = validate_registration_dof(dof)
    moving_centre = np.asarray(moving_centre, dtype=np.float64)
    reference_centre = np.asarray(reference_centre, dtype=np.float64)
    resolution_scale = np.hypot(*reference_shape[:2]) / np.hypot(*moving_shape[:2])
    if value == 2:
        matrix = np.eye(2, 3, dtype=np.float32)
        matrix[:, 2] = reference_centre - moving_centre
    else:
        matrix = cv2.getRotationMatrix2D(tuple(moving_centre), float(rotation_deg), float(resolution_scale)).astype(np.float32)
        matrix[:, 2] += reference_centre - moving_centre
    return np.vstack([matrix, [0.0, 0.0, 1.0]]).astype(np.float32) if value == 8 else matrix


def invert_transform(transform: Any) -> np.ndarray:
    """Invert an affine or projective transform, preserving its input shape."""
    matrix = np.asarray(transform, dtype=np.float64)
    if matrix.shape == (2, 3):
        homogeneous = np.vstack([matrix, [0.0, 0.0, 1.0]])
        return np.linalg.inv(homogeneous)[:2, :].astype(np.float32)
    if matrix.shape == (3, 3):
        return np.linalg.inv(matrix).astype(np.float32)
    raise ValueError(f"Transform must have shape (2, 3) or (3, 3); got {matrix.shape}.")


def _prepare_ecc_image(image: np.ndarray, roi: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    import cv2
    width, height = _coarse_shape(image.shape)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    gray = cv2.resize(gray, (width, height), interpolation=cv2.INTER_AREA)
    mask = cv2.resize(roi.astype(np.uint8) * 255, (width, height), interpolation=cv2.INTER_NEAREST)
    ys, xs = np.where(mask > 0)
    centre = np.array([xs.mean(), ys.mean()]) if len(xs) else np.array([(width - 1) / 2.0, (height - 1) / 2.0])
    return gray, mask.astype(np.uint8), centre


def _to_full_resolution_transform(coarse_transform: np.ndarray, moving_shape: Tuple[int, ...], reference_shape: Tuple[int, ...]) -> np.ndarray:
    """Lift a coarse moving-to-reference transform into full-resolution coordinates.

    With ``x_coarse = S @ x_full`` and ``x_reference_coarse = H_coarse @
    x_moving_coarse``, the project-convention full-resolution transform is
    ``H_full = inv(S_reference) @ H_coarse @ S_moving``.
    """
    moving_width, moving_height = _coarse_shape(moving_shape)
    reference_width, reference_height = _coarse_shape(reference_shape)
    moving_scale = np.diag([moving_width / moving_shape[1], moving_height / moving_shape[0], 1.0])
    reference_scale = np.diag([reference_width / reference_shape[1], reference_height / reference_shape[0], 1.0])
    coarse = np.asarray(coarse_transform, dtype=np.float64)
    if coarse.shape == (2, 3):
        coarse = np.vstack([coarse, [0.0, 0.0, 1.0]])
    return np.linalg.inv(reference_scale) @ coarse @ moving_scale


def _validate_ecc_transform(transform: np.ndarray, moving_shape: Tuple[int, ...], reference_shape: Tuple[int, ...], moving_roi: np.ndarray, reference_roi: np.ndarray, dof: int) -> Dict[str, Any]:
    import cv2
    matrix = np.asarray(transform, dtype=np.float64)
    finite = bool(np.all(np.isfinite(matrix)))
    linear = matrix[:2, :2]
    determinant = float(np.linalg.det(linear)) if finite else float("nan")
    singular_values = np.linalg.svd(linear, compute_uv=False) if finite else np.array([np.nan, np.nan])
    anisotropy = float(max(singular_values) / min(singular_values)) if np.all(singular_values > 0) else float("inf")
    expected_scale = np.hypot(*reference_shape[:2]) / np.hypot(*moving_shape[:2])
    estimated_scale = float(np.sqrt(abs(determinant))) if np.isfinite(determinant) else float("nan")
    relative_scale = estimated_scale / expected_scale if expected_scale else float("nan")
    moving_centre = _roi_centroid(moving_roi, moving_shape)
    reference_centre = _roi_centroid(reference_roi, reference_shape)
    transformed_centre = cv2.perspectiveTransform(np.float32([[moving_centre]]), matrix)[0, 0]
    centre_offset = float(np.linalg.norm(transformed_centre - reference_centre) / np.hypot(*reference_shape[:2]))
    warped_roi = cv2.warpPerspective(moving_roi.astype(np.uint8), matrix, (reference_shape[1], reference_shape[0]), flags=cv2.INTER_NEAREST) > 0
    intersection = reference_roi & warped_roi
    reference_coverage = float(intersection.sum() / max(1, reference_roi.sum()))
    moving_coverage = float(intersection.sum() / max(1, warped_roi.sum()))
    diagnostics = {"orientation_preserving": bool(np.isfinite(determinant) and determinant > 0), "estimated_scale": estimated_scale, "relative_scale": relative_scale, "anisotropy": anisotropy, "centre_offset_fraction": centre_offset, "roi_overlap_pixels": int(intersection.sum()), "roi_overlap_fraction": min(reference_coverage, moving_coverage), "rotation_deg": float(np.degrees(np.arctan2(linear[1, 0], linear[0, 0]))) if finite else float("nan"), "valid": True}
    if not finite or (dof in (6, 8) and determinant <= 0):
        diagnostics.update(valid=False, error="ECC transform is reflected, degenerate, or non-finite.")
    elif dof == 6 and (not MIN_RELATIVE_SCALE <= relative_scale <= MAX_RELATIVE_SCALE or anisotropy > MAX_AFFINE_ANISOTROPY):
        diagnostics.update(valid=False, error=f"ECC affine geometry is implausible: relative scale {relative_scale:.2f}, anisotropy {anisotropy:.2f}.")
    elif centre_offset > MAX_CENTRE_OFFSET_FRACTION:
        diagnostics.update(valid=False, error=f"Transformed surgical-field centre lies {centre_offset:.2f} reference diagonals from the reference ROI centre.")
    elif diagnostics["roi_overlap_fraction"] < MIN_ROI_OVERLAP_FRACTION:
        diagnostics.update(valid=False, error=f"ECC transform leaves only {diagnostics['roi_overlap_fraction']:.2f} valid-ROI overlap.")
    return diagnostics


def _ecc_with_mask(template: np.ndarray, moving: np.ndarray, template_mask: np.ndarray, moving_mask: np.ndarray, warp: np.ndarray, motion_type: int) -> Tuple[float, np.ndarray]:
    """Run ECC while keeping the public warp convention moving-to-reference.

    OpenCV's ECC warp maps template/reference coordinates to moving coordinates.
    Convert on both sides of this boundary so callers only handle the project's
    secondary-to-reference convention.
    """
    import cv2
    if not hasattr(cv2, "findTransformECCWithMask"):
        raise RuntimeError("This OpenCV build does not provide findTransformECCWithMask.")
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, ECC_MAX_ITERATIONS, ECC_EPSILON)
    ecc_warp = invert_transform(warp)
    score, ecc_warp = cv2.findTransformECCWithMask(template, moving, template_mask, moving_mask, ecc_warp, motion_type, criteria, ECC_GAUSS_FILTER_SIZE)
    return score, invert_transform(ecc_warp)


def _load_registration_images(moving_photo: Any, reference_photo: Any) -> Tuple[Any, Any, str, str]:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required for photo registration.") from exc
    moving_path = str(getattr(moving_photo, "source_path", moving_photo))
    reference_path = str(getattr(reference_photo, "source_path", reference_photo))
    moving_image = cv2.imread(moving_path, cv2.IMREAD_COLOR)
    reference_image = cv2.imread(reference_path, cv2.IMREAD_COLOR)
    if moving_image is None or reference_image is None:
        raise ValueError(f"Could not read registration images: {moving_path}, {reference_path}")
    return moving_image, reference_image, moving_path, reference_path


def _roi_centroid(roi: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
    coordinates = np.column_stack(np.where(roi))
    if not len(coordinates):
        return np.array([(shape[1] - 1) / 2.0, (shape[0] - 1) / 2.0], dtype=np.float64)
    return np.array([coordinates[:, 1].mean(), coordinates[:, 0].mean()], dtype=np.float64)


def register_ecc(moving_photo: Any, reference_photo: Any, *, dof: int = DEFAULT_REGISTRATION_DOF, **kwargs: Any) -> PhotoRegistrationResult:
    """Register a moving photo using masked, multi-start OpenCV ECC."""
    import cv2

    dof = validate_registration_dof(dof)
    moving_image, reference_image, moving_path, reference_path = _load_registration_images(moving_photo, reference_photo)
    moving_roi = load_registration_roi(moving_photo, moving_image.shape)
    reference_roi = load_registration_roi(reference_photo, reference_image.shape)
    moving_gray, moving_mask, moving_centre = _prepare_ecc_image(moving_image, moving_roi)
    reference_gray, reference_mask, reference_centre = _prepare_ecc_image(reference_image, reference_roi)
    candidates = []
    for angle in ECC_INITIAL_ROTATIONS_DEG:
        try:
            initial = initial_warp_for_dof(dof, moving_gray.shape, reference_gray.shape, moving_centre, reference_centre, angle)
            score, coarse_transform = _ecc_with_mask(reference_gray, moving_gray, reference_mask, moving_mask, initial, ecc_motion_type(dof))
            transform = _to_full_resolution_transform(coarse_transform, moving_image.shape, reference_image.shape)
            diagnostics = _validate_ecc_transform(transform, moving_image.shape, reference_image.shape, moving_roi, reference_roi, dof)
            candidates.append({"initial_rotation_deg": angle, "ecc_score": float(score), "converged": True, "ecc_succeeded": True, **diagnostics, "transform": transform})
        except (cv2.error, ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            candidates.append({"initial_rotation_deg": angle, "converged": False, "ecc_succeeded": False, "error": str(exc)})
    valid_candidates = [candidate for candidate in candidates if candidate.get("ecc_succeeded") and candidate.get("valid")]
    if not valid_candidates:
        raise ValueError(f"ECC registration failed: none of {len(candidates)} initialization angles converged to a geometrically valid transform.")
    best = max(valid_candidates, key=lambda candidate: candidate["ecc_score"])
    transform = best["transform"]
    registered_image = cv2.warpPerspective(moving_image, transform, (reference_image.shape[1], reference_image.shape[0]), flags=cv2.INTER_LINEAR)
    metadata = {key: value for key, value in best.items() if key not in {"transform", "valid", "error"}}
    metadata.update({"backend": "opencv_ecc", "dof": dof, "motion_type": ECC_MOTION_BY_DOF[dof], "initialization_candidates": len(candidates), "converged_candidates": sum(candidate.get("ecc_succeeded", False) for candidate in candidates), "valid_candidates": len(valid_candidates), "gauss_filter_size": ECC_GAUSS_FILTER_SIZE, "max_iterations": ECC_MAX_ITERATIONS, "epsilon": ECC_EPSILON, "algorithm_version": REGISTRATION_ALGORITHM_VERSION, "candidate_summary": [{key: candidate.get(key) for key in ("initial_rotation_deg", "ecc_score", "converged", "ecc_succeeded", "valid")} for candidate in candidates], "convergence_note": "converged means ECC returned successfully; OpenCV does not expose whether termination was caused by epsilon or max_iterations."})
    result = PhotoRegistrationResult(str(getattr(reference_photo, "photo_id", reference_path)), str(getattr(moving_photo, "photo_id", moving_path)), _method_from_dof(dof), source_image_path=moving_path, reference_image_path=reference_path, source_dimensions=(moving_image.shape[1], moving_image.shape[0]), reference_dimensions=(reference_image.shape[1], reference_image.shape[0]), transform=transform.tolist(), metadata=metadata)
    output_path = kwargs.get("output_path")
    if output_path:
        if not cv2.imwrite(str(output_path), registered_image):
            raise OSError(f"Could not save registered image: {output_path}")
        result.registered_image_path = str(output_path)
    return result


def register_photo_to_reference(moving_photo: Any, reference_photo: Any, *, dof: int = DEFAULT_REGISTRATION_DOF, method: Optional[Any] = None, **kwargs: Any) -> PhotoRegistrationResult:
    """Register a moving photo using masked ECC; ``method`` is legacy compatibility only."""
    if method is not None:
        dof = _dof_from_method(method)
    return register_ecc(moving_photo, reference_photo, dof=dof, **kwargs)


def _file_fingerprint(path: Optional[str]) -> Optional[Dict[str, Any]]:
    if not path or not os.path.exists(path):
        return None
    absolute_path = os.path.abspath(path)
    stat = os.stat(absolute_path)
    return {"path": absolute_path, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _registration_cache_signature(moving_photo: Any, reference_photo: Any, dof: int) -> Optional[Dict[str, Any]]:
    if isinstance(dof, PhotoRegistrationMethod) or isinstance(dof, str):
        dof = _dof_from_method(dof)
    moving_masks = getattr(moving_photo, "masks", None) or {}
    reference_masks = getattr(reference_photo, "masks", None) or {}
    signature = {
        "algorithm_version": REGISTRATION_ALGORITHM_VERSION,
        "backend": "opencv_ecc",
        "dof": validate_registration_dof(dof),
        "moving_image": _file_fingerprint(getattr(moving_photo, "source_path", None)),
        "reference_image": _file_fingerprint(getattr(reference_photo, "source_path", None)),
        "moving_mask": _file_fingerprint(moving_masks.get("path")),
        "reference_mask": _file_fingerprint(reference_masks.get("path")),
    }
    if any(signature[key] is None for key in ("moving_image", "reference_image", "moving_mask", "reference_mask")):
        return None
    return signature


def register_photo_set(photo_set: PatientPhotoSet, output_dir: str, *, dof: int = DEFAULT_REGISTRATION_DOF, method: Optional[Any] = None) -> List[PhotoRegistrationResult]:
    """Register all auxiliary photos into the reference grid using ECC."""
    os.makedirs(output_dir, exist_ok=True)
    if method is not None:
        dof = _dof_from_method(method)
    dof = validate_registration_dof(dof)
    results = []
    for photo in photo_set.all_photos()[1:]:
        photo_dof = validate_registration_dof(getattr(photo, "registration_dof", None) or photo.metadata.get("registration_dof") or dof)
        cache_signature = _registration_cache_signature(photo, photo_set.reference_photo, photo_dof)
        output_path = os.path.join(output_dir, f"{photo.photo_id}_registered.png")
        result_path = photo.registration_result_path or os.path.join(output_dir, f"{photo.photo_id}_registration.json")
        if photo.registered_image_path and os.path.exists(photo.registered_image_path) and os.path.exists(result_path):
            try:
                cached_result = load_registration(result_path)
                cached_image_path = cached_result.registered_image_path
                if cached_image_path and not os.path.isabs(cached_image_path):
                    cached_image_path = os.path.join(os.path.dirname(result_path), cached_image_path)
                if (
                    cached_result.status == "registered"
                    and cached_result.reference_photo_id == photo_set.reference_photo.photo_id
                    and cached_result.moving_photo_id == photo.photo_id
                    and cached_result.metadata.get("dof") == photo_dof
                    and cached_result.metadata.get("algorithm_version") == REGISTRATION_ALGORITHM_VERSION
                    and cache_signature is not None
                    and cached_result.metadata.get("cache_signature") == cache_signature
                    and cached_image_path
                    and os.path.exists(cached_image_path)
                ):
                    cached_result.registered_image_path = cached_image_path
                    result = cached_result
                    photo.registration_dof = photo_dof
                    photo.registration_method = result.method
                    photo.registration_status = result.status
                    photo.registered_image_path = result.registered_image_path
                    photo.registration_result_path = result_path
                    _reuse_or_warp_registered_masks(photo, result, output_dir)
                    results.append(result)
                    continue
            except (OSError, ValueError, KeyError, json.JSONDecodeError):
                pass
        try:
            result = register_photo_to_reference(
                photo,
                photo_set.reference_photo,
                dof=photo_dof,
                output_path=output_path,
            )
        except (ImportError, OSError, ValueError, RuntimeError) as exc:
            result = PhotoRegistrationResult(
                reference_photo_id=photo_set.reference_photo.photo_id,
                moving_photo_id=photo.photo_id,
                method=_method_from_dof(photo_dof),
                status="registration_pending",
                source_image_path=photo.source_path,
                reference_image_path=photo_set.reference_photo.source_path,
                metadata={"backend": "opencv_ecc", "dof": photo_dof, "algorithm_version": REGISTRATION_ALGORITHM_VERSION, "cache_signature": cache_signature, "error": str(exc)},
            )
        photo.registration_dof = photo_dof
        photo.registration_method = result.method
        photo.registration_status = result.status
        photo.registered_image_path = result.registered_image_path
        photo.registration_result_path = result_path
        if result.status == "registered":
            set_registration_qc_pending(photo)
        else:
            photo.registration_qc_status = "pending" if photo.is_secondary else "not_required"
            photo.registration_qc_reviewed_at = None
        result.metadata.setdefault("cache_signature", cache_signature)
        save_registration_result(photo.registration_result_path, result)

        _reuse_or_warp_registered_masks(photo, result, output_dir, force=True)

        results.append(result)
    return results


def _reuse_or_warp_registered_masks(photo: Any, result: PhotoRegistrationResult, output_dir: str, force: bool = False) -> None:
    """Reuse an existing registered mask or create it once for a registered photo."""
    if result.status != "registered" or not photo.masks:
        return
    registered_mask_path = photo.masks.get("registered_path")
    if not force and registered_mask_path and os.path.exists(registered_mask_path):
        return
    mask_path = photo.masks.get("path")
    if not mask_path or not os.path.exists(mask_path):
        return
    try:
        registered_mask_path = os.path.join(output_dir, f"{photo.photo_id}_registered_masks.npz")
        warp_mask_file_to_reference(mask_path, result, registered_mask_path)
        photo.masks["registered_path"] = registered_mask_path
    except (NotImplementedError, ImportError, OSError, ValueError) as exc:
        print(f"[photo_registration] Could not warp masks for '{photo.photo_id}' into the reference grid: {exc}")


def warp_image_to_reference(moving_image: Any, registration: PhotoRegistrationResult, **kwargs: Any) -> Any:
    """Warp an image into the reference-photo grid using the saved registration result."""
    if registration.transform is None or registration.reference_dimensions is None:
        raise ValueError("Registration result must contain a transform and reference dimensions.")
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required for image warping.") from exc
    width, height = registration.reference_dimensions
    return cv2.warpPerspective(np.asarray(moving_image), np.asarray(registration.transform, dtype=np.float64), (width, height), flags=kwargs.get("interpolation", cv2.INTER_LINEAR))


def warp_mask_to_reference(mask: Any, registration: PhotoRegistrationResult, **kwargs: Any) -> Any:
    """Warp a boolean mask into the reference-photo grid using the same registration result.

    Similarity and projective transforms are stored as homogeneous matrices and
    both use nearest-neighbour interpolation for boolean ROI/label pixels.
    """
    if registration.transform is None:
        raise ValueError("Registration result has no transform to warp masks with.")
    if registration.reference_dimensions is None:
        raise ValueError("Registration result has no reference image dimensions to warp masks to.")

    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required for registration mask warping.") from exc

    transform = np.asarray(registration.transform, dtype=np.float64)
    width, height = registration.reference_dimensions
    mask_uint8 = np.asarray(mask).astype(np.uint8)
    warped = cv2.warpPerspective(mask_uint8, transform, (width, height), flags=cv2.INTER_NEAREST)
    return warped.astype(bool)


def warp_mask_file_to_reference(mask_path: str, registration: PhotoRegistrationResult, output_path: str) -> str:
    """Warp a saved `.npz` mask file (resection_mask/outside_mask) into the reference grid and save it."""
    with np.load(mask_path) as masks:
        warped = {
            name: warp_mask_to_reference(masks[name], registration)
            for name in masks.files
        }
    directory = os.path.dirname(output_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    np.savez_compressed(output_path, **warped)
    return output_path


def transform_points_to_reference(points: Any, registration: PhotoRegistrationResult, **kwargs: Any) -> Any:
    """Transform points from secondary-photo coordinates to reference-photo coordinates."""
    if registration.transform is None:
        raise ValueError("Registration result has no transform.")
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required for point transformation.") from exc
    points_array = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(points_array, np.asarray(registration.transform, dtype=np.float64)).reshape(-1, 2)


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
