from __future__ import annotations

import os
import json
import yaml
import shutil
import subprocess
import platform
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import numpy as np
import matplotlib

matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.widgets import PolygonSelector
from matplotlib.path import Path
import tkinter as tk
from tkinter import filedialog

@dataclass
class PhotoRecord:
    """Metadata for one photograph in a patient photo set."""

    photo_id: str
    source_path: str
    role: str = "secondary"
    photo_type: Optional[str] = None
    registration_method: Optional["PhotoRegistrationMethod"] = None  # type: ignore
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
    from util.photo_registration import _normalise_registration_method

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

    photo_files = _discover_image_files(base_dir)
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

    from util.photo_registration import PhotoRegistrationMethod

    photos = []
    for photo in photo_set.all_photos():
        entry = dict(photo.metadata)
        entry.update({
            "id": photo.photo_id,
            "path": os.path.relpath(photo.source_path, os.path.dirname(photo_set.manifest_path or os.getcwd())) if photo_set.manifest_path else photo.source_path,
            "role": photo.role,
            "photo_type": photo.photo_type,
            "registration_method": photo.registration_method.value if isinstance(photo.registration_method, PhotoRegistrationMethod) else photo.registration_method,
            "registration_status": photo.registration_status,
            "registration_result_path": photo.registration_result_path,
            "registered_image_path": photo.registered_image_path,
            "masks": photo.masks,
        })
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


def discover_photo_set_manifest(patient_id: str, patient_photo_dir: str, manifest_path: Optional[str] = None) -> PatientPhotoSet:
    """Compatibility wrapper used by the workflow when the patient photo set is specified by manifest."""
    return discover_patient_photo_set(patient_id=patient_id, patient_photo_dir=patient_photo_dir, picture_root=patient_photo_dir, manifest_path=manifest_path)


def open_image_viewer(path):
    """
    Open an image file in the system's default image viewer.
    
    Parameters
    ----------
    path : str
        Path to the image file to open
    
    Returns
    -------
    subprocess.Popen
        Process handle for the opened viewer (platform-dependent)
    """
    system = platform.system()

    if system == "Windows":
        # Use 'start' through cmd so we get a handle
        return subprocess.Popen(["explorer", path])
    elif system == "Darwin":  # macOS
        # Preview stays open, but we can kill it later by name
        return subprocess.Popen(["open", "-a", "Preview", path])
    else:  # Linux
        # Use eog (Eye of GNOME) or fallback viewer
        return subprocess.Popen(["eog", path])


def close_image_viewer(process):
    """
    Close an image viewer process opened with open_image_viewer.
    
    Parameters
    ----------
    process : subprocess.Popen
        Process handle returned by open_image_viewer
    """
    system = platform.system()

    if system == "Windows":
        process.terminate()  # Try to close process (doesn't always work, needs manual close)
    elif system == "Darwin":  # macOS
        subprocess.run(["pkill", "Preview"])  # Close Preview
    else:  # Linux
        process.terminate()  # Close eog


def _discover_image_files(photo_dir):
    """List image files under a patient photo directory."""
    ext_set = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    matches = []
    for root, _, files in os.walk(photo_dir):
        for name in files:
            if os.path.splitext(name)[1].lower() in ext_set:
                matches.append(os.path.join(root, name))
    return sorted(matches)


def prompt_select_reference_photo(patient_id, patient_photo_dir, parent_title="Select reference photo"):
    """Open a file dialog so the user selects the patient reference photo."""
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    reference_path = filedialog.askopenfilename(
        title=f"{parent_title} for patient {patient_id}",
        initialdir=patient_photo_dir,
        filetypes=[("All files", "*.*"), ("JPEG files", "*.jpg *.jpeg"), ("PNG files", "*.png")],
    )
    root.destroy()
    return reference_path or None


def prompt_select_secondary_photos(patient_id, patient_photo_dir, reference_photo_path=None, parent_title="Select secondary photos"):
    """Open a file dialog so the user selects one or more secondary photos."""
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    secondary_paths = filedialog.askopenfilenames(
        title=f"{parent_title} for patient {patient_id}",
        initialdir=patient_photo_dir,
        filetypes=[("All files", "*.*"), ("JPEG files", "*.jpg *.jpeg"), ("PNG files", "*.png")],
    )
    paths = root.tk.splitlist(secondary_paths) if secondary_paths else []
    root.destroy()
    if reference_photo_path:
        paths = [path for path in paths if os.path.abspath(path) != os.path.abspath(reference_photo_path)]
    return paths


def prompt_select_post_resection_photo(patient_id, patient_photo_dir, reference_photo_path=None, parent_title="Select post-resection photo (optional)"):
    """Let the user choose at most one post-resection photo, or cancel."""
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    path = filedialog.askopenfilename(
        title=f"{parent_title} for patient {patient_id}",
        initialdir=patient_photo_dir,
        filetypes=[("All files", "*.*"), ("JPEG files", "*.jpg *.jpeg"), ("PNG files", "*.png")],
    )
    root.destroy()
    if not path or (reference_photo_path and os.path.abspath(path) == os.path.abspath(reference_photo_path)):
        return None
    return path


def build_photo_set_from_selection(patient_id, patient_photo_dir, manifest_path=None, reference_photo_path=None, post_resection_photo_path=None, secondary_photo_paths=None):
    """Create a photo set through reference, optional post-resection, then secondary selection."""
    if patient_photo_dir is None or not os.path.isdir(patient_photo_dir):
        raise FileNotFoundError(f"Photo directory not found: {patient_photo_dir}")

    if reference_photo_path is None:
        reference_photo_path = prompt_select_reference_photo(patient_id, patient_photo_dir)
    if not reference_photo_path:
        raise ValueError(f"No reference photo selected for patient '{patient_id}'.")

    if post_resection_photo_path is None:
        post_resection_photo_path = prompt_select_post_resection_photo(patient_id, patient_photo_dir, reference_photo_path)
    if post_resection_photo_path and os.path.abspath(post_resection_photo_path) == os.path.abspath(reference_photo_path):
        raise ValueError("The post-resection photo must differ from the reference photo.")

    if secondary_photo_paths is None:
        secondary_photo_paths = prompt_select_secondary_photos(patient_id, patient_photo_dir, reference_photo_path=reference_photo_path)
    excluded_paths = {os.path.abspath(reference_photo_path)}
    if post_resection_photo_path:
        excluded_paths.add(os.path.abspath(post_resection_photo_path))
    secondary_photo_paths = [path for path in secondary_photo_paths if os.path.abspath(path) not in excluded_paths]

    reference_record = PhotoRecord(
        photo_id=os.path.splitext(os.path.basename(reference_photo_path))[0],
        source_path=reference_photo_path,
        role="reference",
        photo_type="reference",
        registration_status="reference_ready",
    )
    secondary_records = []
    for path in secondary_photo_paths:
        secondary_records.append(
            PhotoRecord(
                photo_id=os.path.splitext(os.path.basename(path))[0],
                source_path=path,
                role="secondary",
                photo_type="secondary",
                registration_status="pending",
            )
        )

    post_resection_record = None
    if post_resection_photo_path:
        post_resection_record = PhotoRecord(
            photo_id=os.path.splitext(os.path.basename(post_resection_photo_path))[0],
            source_path=post_resection_photo_path,
            role="secondary",
            photo_type="post_resection",
            registration_status="pending",
        )

    photo_set = PatientPhotoSet(
        patient_id=patient_id,
        reference_photo=reference_record,
        post_resection_photo=post_resection_record,
        secondary_photos=secondary_records,
        manifest_path=manifest_path,
    )
    if manifest_path is not None:
        save_photo_set_manifest(photo_set, manifest_path)
    return photo_set


def discover_photo_set(patient_id, picture_root, patient_photo_dir=None, manifest_path=None):
    """Returns the patient photo set.

    `manifest_path` is the single manifest location (typically the patient's
    output directory), used both to reuse a previous selection and, when a
    patient has multiple photos and no manifest exists yet, as the destination
    for the interactively-constructed photo-set manifest.
    """
    if patient_photo_dir is None:
        patient_photo_dir = os.path.join(picture_root, patient_id)

    if not os.path.isdir(patient_photo_dir):
        patient_photo_dir = os.path.join(picture_root, patient_id)

    if manifest_path and os.path.exists(manifest_path):
        return discover_patient_photo_set(
            patient_id=patient_id,
            patient_photo_dir=patient_photo_dir,
            picture_root=picture_root,
            manifest_path=manifest_path,
        )

    photo_files = _discover_image_files(patient_photo_dir)
    if not photo_files:
        raise FileNotFoundError(f"No images found for patient '{patient_id}' in '{patient_photo_dir}'.")
    if len(photo_files) == 1:
        return discover_patient_photo_set(
            patient_id=patient_id,
            patient_photo_dir=patient_photo_dir,
            picture_root=picture_root,
            manifest_path=None,
        )

    photo_set = build_photo_set_from_selection(
        patient_id=patient_id,
        patient_photo_dir=patient_photo_dir,
        manifest_path=manifest_path,
    )
    return photo_set


def copy_photo_to_output(photo_path, output_dir, patient_id, preferred_name=None):
    """Copy a photograph into the output directory and return the copied path."""
    os.makedirs(output_dir, exist_ok=True)
    base_name = preferred_name or os.path.basename(photo_path)
    dest_path = os.path.join(output_dir, base_name)
    if os.path.abspath(photo_path) != os.path.abspath(dest_path):
        shutil.copy2(photo_path, dest_path)
    return dest_path


def draw_photo_masks(photo_path, save_path=None, tqdm_handle=None, photo_role="reference", include_resection=None):
    """Draw photo masks for the reference or a secondary image.

    For the reference photo, the outside mask remains the main useful manual
    segmentation. For a secondary photo, the resection mask is only relevant if
    the image is genuinely a post-resection photograph; otherwise the workflow may
    skip resection annotations entirely.
    """

    # Define logging function
    log = tqdm_handle.write if tqdm_handle else print

    if include_resection is None:
        include_resection = photo_role != "reference"

    img = np.array(Image.open(photo_path).convert("RGB"))
    h, w, _ = img.shape
    resection_mask = np.zeros((h, w), bool)
    outside_mask = np.zeros((h, w), bool)

    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img)

    if include_resection:
        ax.set_title("Draw resection area (double-click to close, ENTER to confirm)")
    else:
        ax.set_title("Draw outside area (double-click to close, ENTER to confirm)")
    plt.tight_layout()
    mask_done = [False]
    abort = [False]

    def polygon_to_mask(verts):
        path = Path(verts)
        y, x = np.mgrid[:h, :w]
        coords = np.stack((x.ravel(), y.ravel()), axis=-1)
        return path.contains_points(coords).reshape(h, w)

    def onselect_resection(verts):
        nonlocal resection_mask
        resection_mask[:] = polygon_to_mask(verts)
        ax.imshow(np.dstack((img / 255.0, np.where(resection_mask, 0.4, 0.0))))
        ax.set_title("Resection drawn. Press ENTER to continue.")
        fig.canvas.draw_idle()

    if include_resection:
        selector = PolygonSelector(ax, onselect_resection, useblit=True)

        def on_key(event):
            if event.key == "enter":
                mask_done[0] = True

        def on_close(event):
            abort[0] = True

        fig.canvas.mpl_connect("key_press_event", on_key)
        fig.canvas.mpl_connect("close_event", on_close)

        fig.canvas.draw_idle()
        plt.show(block=False)
        log("Draw resection area → double-click to close → press ENTER to continue.")
        while not mask_done[0] and not abort[0]:
            fig.canvas.start_event_loop(0.1)

        if abort[0]:
            plt.close(fig)
            log("Mask drawing aborted by user.")
            return False

        selector.disconnect_events()
        mask_done[0] = False

    def onselect_outside(verts):
        outside_mask[:] = np.logical_not(polygon_to_mask(verts))
        ax.imshow(np.dstack((img / 255.0, np.where(outside_mask, 0.4, 0.0))))
        ax.set_title("Outside area drawn. Press ENTER to finish.")
        fig.canvas.draw_idle()

    ax.imshow(img)
    ax.set_title("Draw OUTSIDE area (double-click to close, ENTER to confirm)")
    selector = PolygonSelector(ax, onselect_outside, useblit=True)

    def on_key(event):
        if event.key == "enter":
            mask_done[0] = True

    def on_close(event):
        abort[0] = True

    fig.canvas.mpl_connect("key_press_event", on_key)
    fig.canvas.mpl_connect("close_event", on_close)

    fig.canvas.draw_idle()
    plt.show(block=False)
    log("Draw outside area → double-click to close → press ENTER to finish.")
    while not mask_done[0] and not abort[0]:
        fig.canvas.start_event_loop(0.1)

    if abort[0]:
        selector.disconnect_events()
        plt.close(fig)
        log("Mask drawing aborted by user.")
        return False

    selector.disconnect_events()
    plt.close(fig)

    if save_path is None:
        base = os.path.splitext(photo_path)[0]
        save_path = base + "_masks.npz"
    np.savez_compressed(save_path, resection_mask=resection_mask, outside_mask=outside_mask)
    log(f"Masks saved to {save_path}")
    return True

def show_photo_with_masks(photo_path, mask_path, save_path=None,
                          resection_color=(0, 0.7, 0, 0.4),
                          outside_color=(0, 0, 0, 0.6),
                          figsize=(10, 10), tqdm_handle=None):
    """
    Display the photograph with resection and outside masks overlayed.

    Parameters
    ----------
    photo_path : str
        Path to the photograph file (e.g., .jpg, .png).
    mask_path : str
        Path to the .npz file containing 'resection_mask' and 'outside_mask'.
    resection_color : tuple
        RGBA color for the resection mask (default: semi-transparent red).
    outside_color : tuple
        RGBA color for the outside mask (default: semi-transparent blue).
    figsize : tuple
        Figure size in inches.

    Returns
    -------
    (fig, ax)
        Matplotlib figure and axes handles for further customization.
    """

    # Define logging function
    log = tqdm_handle.write if tqdm_handle else print

    with plt.ioff():
        # Load image and masks
        img = np.array(Image.open(photo_path).convert("RGB"))
        masks = np.load(mask_path)
        resection_mask = masks.get("resection_mask", None)
        outside_mask = masks.get("outside_mask", None)

        # Safety check
        if resection_mask is None or outside_mask is None:
            raise ValueError("Mask file must contain 'resection_mask' and 'outside_mask' arrays.")

        # Plot base image
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(img)
        ax.set_title("Pre-operative Photograph (+ masked areas)")
        ax.axis("off")

        # Overlay resection mask (semi-transparent red)
        if resection_mask.any():
            overlay_r = np.zeros((*resection_mask.shape, 4))
            overlay_r[..., 0] = resection_color[0]
            overlay_r[..., 1] = resection_color[1]
            overlay_r[..., 2] = resection_color[2]
            overlay_r[..., 3] = resection_mask.astype(float) * resection_color[3]
            ax.imshow(overlay_r)

        # Overlay outside mask (semi-transparent blue)
        if outside_mask.any():
            overlay_o = np.zeros((*outside_mask.shape, 4))
            overlay_o[..., 0] = outside_color[0]
            overlay_o[..., 1] = outside_color[1]
            overlay_o[..., 2] = outside_color[2]
            overlay_o[..., 3] = outside_mask.astype(float) * outside_color[3]
            ax.imshow(overlay_o)

        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path, dpi=300)
            log(f"Figure saved to {save_path}")
            plt.close()
        else:
            plt.show()

    return fig, ax
