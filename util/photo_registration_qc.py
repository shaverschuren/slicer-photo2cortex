from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None

from PIL import Image


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _photo_registered_rows(photo_set: Any) -> List[Any]:
    registered: List[Any] = []
    for photo in photo_set.all_photos()[1:]:
        if photo.registration_status == "registered":
            registered.append(photo)
    return registered


def _as_rgb_array(path: str) -> np.ndarray:
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")
    image = Image.open(path).convert("RGB")
    try:
        return np.asarray(image)
    finally:
        image.close()


def _load_optional_masks(photo: Any) -> Dict[str, np.ndarray]:
    masks: Dict[str, np.ndarray] = {}
    if not getattr(photo, "masks", None):
        return masks
    mask_path = photo.masks.get("registered_path") or photo.masks.get("path")
    if not mask_path or not os.path.exists(mask_path):
        return masks
    try:
        with np.load(mask_path) as payload:
            for key in payload.files:
                masks[key] = np.asarray(payload[key], dtype=bool)
    except (OSError, ValueError, KeyError):
        return {}
    return masks


def _valid_roi_mask_from_maybe_mask(mask_payload: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if mask_payload is None:
        return None
    mask = np.asarray(mask_payload, dtype=bool)
    return np.logical_not(mask)


def _resize_for_qc(image_array: np.ndarray, max_image_size: int = 1000) -> np.ndarray:
    height, width = image_array.shape[:2]
    scale = min(1.0, float(max_image_size) / max(height, width))
    if scale >= 1.0:
        return image_array.astype(np.uint8)
    target_w = max(1, int(round(width * scale)))
    target_h = max(1, int(round(height * scale)))
    if cv2 is not None:
        return cv2.resize(image_array.astype(np.uint8), (target_w, target_h), interpolation=cv2.INTER_AREA)
    return np.asarray(Image.fromarray(image_array.astype(np.uint8)).resize((target_w, target_h), Image.Resampling.BILINEAR))


def _make_overlay_image(reference_rgb: np.ndarray, registered_rgb: np.ndarray, reference_valid: Optional[np.ndarray], registered_valid: Optional[np.ndarray]) -> np.ndarray:
    base = reference_rgb.astype(np.float32)
    overlay = registered_rgb.astype(np.float32)
    visible = np.clip(0.5 * base + 0.5 * overlay, 0, 255).astype(np.uint8)
    if reference_valid is None and registered_valid is None:
        return visible

    contour_canvas = np.zeros((*visible.shape[:2], 3), dtype=np.uint8)
    if reference_valid is not None and reference_valid.size:
        if cv2 is not None:
            contours, _ = cv2.findContours(reference_valid.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(contour_canvas, contours, -1, (0, 255, 255), 1)
        else:
            contour_canvas[..., 1] = np.where(reference_valid.astype(bool), 255, 0)
    if registered_valid is not None and registered_valid.size:
        if cv2 is not None:
            contours, _ = cv2.findContours(registered_valid.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(contour_canvas, contours, -1, (255, 0, 0), 1)
        else:
            contour_canvas[..., 0] = np.where(registered_valid.astype(bool), 255, 0)
    return np.maximum(visible, contour_canvas)


def _make_edge_panel(reference_rgb: np.ndarray, registered_rgb: np.ndarray) -> np.ndarray:
    if cv2 is not None:
        ref_gray = cv2.cvtColor(reference_rgb, cv2.COLOR_RGB2GRAY)
        reg_gray = cv2.cvtColor(registered_rgb, cv2.COLOR_RGB2GRAY)
        ref_gray = cv2.GaussianBlur(ref_gray, (3, 3), 0)
        reg_gray = cv2.GaussianBlur(reg_gray, (3, 3), 0)
        ref_edges = cv2.Canny(ref_gray, 50, 150)
        reg_edges = cv2.Canny(reg_gray, 50, 150)
        edge_canvas = np.zeros((*ref_edges.shape, 3), dtype=np.uint8)
        edge_canvas[..., 0] = reg_edges
        edge_canvas[..., 1] = ref_edges
        return edge_canvas

    ref_gray = np.mean(reference_rgb, axis=2).astype(np.uint8)
    reg_gray = np.mean(registered_rgb, axis=2).astype(np.uint8)
    ref_edges = np.asarray(ref_gray > 128, dtype=np.uint8) * 255
    reg_edges = np.asarray(reg_gray > 128, dtype=np.uint8) * 255
    edge_canvas = np.zeros((ref_edges.shape[0], ref_edges.shape[1], 3), dtype=np.uint8)
    edge_canvas[..., 0] = reg_edges
    edge_canvas[..., 1] = ref_edges
    return edge_canvas


def _registration_label(photo: Any, result: Optional[Any] = None) -> str:
    if result is None:
        return getattr(photo, "photo_type", None) or getattr(photo, "role", "secondary")
    metadata = getattr(result, "metadata", {}) or {}
    method_name = f"ECC {metadata.get('dof', '?')}-DOF" if metadata.get("backend") == "opencv_ecc" else (getattr(result.method, "value", str(result.method)) if hasattr(result.method, "value") else str(result.method))
    if metadata.get("backend") == "opencv_ecc" and metadata.get("ecc_score") is not None:
        return f"{method_name} | rho={metadata['ecc_score']:.2f} | rot {metadata.get('rotation_deg', 0):.0f} deg | scale {metadata.get('relative_scale', 0):.2f} | overlap {100 * metadata.get('roi_overlap_fraction', 0):.0f}%"
    match_count = metadata.get("mutual_match_count", metadata.get("match_count"))
    inlier_count = metadata.get("inlier_count")
    if match_count is not None and inlier_count is not None and match_count:
        ratio = 100.0 * inlier_count / match_count
        details = f"{method_name} | {inlier_count} / {match_count} inliers ({ratio:.0f}%)"
        if metadata.get("estimated_scale") is not None and metadata.get("rotation_deg") is not None:
            details += f" | scale {metadata['estimated_scale']:.2f} | rot {metadata['rotation_deg']:.0f} deg"
        return details
    return method_name


def create_registration_qc_montage(photo_set: Any, registration_results: Sequence[Any], output_path: str, max_image_size: int = 1000) -> Optional[str]:
    """Create a subject-level montage of the registered auxiliary photos for human review."""
    if plt is None:
        return None

    registered_photos = _photo_registered_rows(photo_set)
    if not registered_photos:
        return None

    reference_photo = photo_set.reference_photo
    reference_path = getattr(reference_photo, "source_path", None)
    if not reference_path or not os.path.exists(reference_path):
        return None

    reference_rgb = _resize_for_qc(_as_rgb_array(reference_path), max_image_size=max_image_size)
    result_by_photo = {str(getattr(result, "moving_photo_id", "")): result for result in registration_results if getattr(result, "status", "") == "registered"}

    rows = len(registered_photos)
    fig = plt.figure(figsize=(15, 4.0 * rows + 1.5))
    fig.suptitle(f"{photo_set.patient_id} — photo-to-reference registration QC\nReference: {reference_photo.photo_id}", fontsize=12, y=0.98)
    gs = fig.add_gridspec(rows, 3)

    for row_index, photo in enumerate(registered_photos):
        result = result_by_photo.get(str(photo.photo_id))
        registered_path = getattr(photo, "registered_image_path", None)
        if not registered_path or not os.path.exists(registered_path):
            continue

        registered_rgb = _resize_for_qc(_as_rgb_array(registered_path), max_image_size=max_image_size)

        ref_valid = None
        reg_valid = None
        if getattr(reference_photo, "masks", None):
            ref_masks = _load_optional_masks(reference_photo)
            if "outside_mask" in ref_masks:
                ref_valid = _valid_roi_mask_from_maybe_mask(ref_masks.get("outside_mask"))
        if getattr(photo, "masks", None):
            reg_masks = _load_optional_masks(photo)
            if "outside_mask" in reg_masks:
                reg_valid = _valid_roi_mask_from_maybe_mask(reg_masks.get("outside_mask"))

        if ref_valid is not None and ref_valid.shape != reference_rgb.shape[:2]:
            ref_valid = None
        if reg_valid is not None and reg_valid.shape != registered_rgb.shape[:2]:
            reg_valid = None

        ax1 = fig.add_subplot(gs[row_index, 0])
        ax1.imshow(registered_rgb)
        ax1.set_title("Registered photo")
        ax1.axis("off")

        ax2 = fig.add_subplot(gs[row_index, 1])
        ax2.imshow(_make_overlay_image(reference_rgb, registered_rgb, ref_valid, reg_valid))
        ax2.set_title("50/50 overlay")
        ax2.axis("off")

        ax3 = fig.add_subplot(gs[row_index, 2])
        ax3.imshow(_make_edge_panel(reference_rgb, registered_rgb))
        ax3.set_title("Edge alignment")
        ax3.axis("off")

        fig.text(0.02, 0.93 - row_index * (0.90 / max(rows, 1)), f"{photo.photo_id}\n{photo.photo_type or photo.role}", fontsize=9, ha="left", va="top", weight="bold")
        label = _registration_label(photo, result)
        ax3.text(0.5, -0.12, label, transform=ax3.transAxes, ha="center", va="top", fontsize=8)

    directory = os.path.dirname(output_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    fig.tight_layout(rect=[0.05, 0.02, 1, 0.96])
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def _apply_qc_decision(photo_set: Any, decision: str, reviewed_at: Optional[str] = None) -> None:
    reviewed_at = reviewed_at or _utc_now_iso()
    for photo in photo_set.all_photos()[1:]:
        if photo.registration_status != "registered":
            continue
        photo.registration_qc_status = decision
        photo.registration_qc_reviewed_at = reviewed_at


def review_registration_qc(montage_path: str, photo_set: Any, registration_results: Optional[Sequence[Any]] = None) -> str:
    """Display a subject-level registration QC montage and wait for an explicit approval/rejection decision."""
    if plt is None or not montage_path or not os.path.exists(montage_path):
        return "aborted"

    img = plt.imread(montage_path)
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.imshow(img)
    ax.set_axis_off()
    fig.suptitle(
        f"{photo_set.patient_id} — registration QC review\nApprove: Y / Enter | Reject: R / N | Abort / close: Esc",
        fontsize=11,
    )
    decision = {"value": "aborted"}

    def _set_decision(value: str) -> None:
        if decision["value"] != "aborted":
            return
        decision["value"] = value
        plt.close(fig)

    def _on_key(event):
        if event.key in {"y", "Y", "enter"}:
            _set_decision("approved")
        elif event.key in {"r", "R", "n", "N"}:
            _set_decision("rejected")
        elif event.key == "escape":
            _set_decision("aborted")

    fig.canvas.mpl_connect("key_press_event", _on_key)
    fig.canvas.mpl_connect("close_event", lambda event: _set_decision("aborted"))
    plt.show(block=True)

    if decision["value"] == "approved":
        _apply_qc_decision(photo_set, "approved")
    elif decision["value"] == "rejected":
        _apply_qc_decision(photo_set, "rejected")
    else:
        for photo in photo_set.all_photos()[1:]:
            if photo.registration_status == "registered":
                photo.registration_qc_status = "pending"
                photo.registration_qc_reviewed_at = None
    return decision["value"]
