import os
import glob
import shutil
import subprocess
import platform
import json
import numpy as np
import matplotlib

matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.widgets import PolygonSelector
from matplotlib.path import Path
import tkinter as tk
from tkinter import filedialog

from photo_registration import (
    PatientPhotoSet,
    PhotoRecord,
    discover_patient_photo_set,
    save_photo_set_manifest,
)

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

def find_post_resection_photo_path(patient_id, picture_root, copy_dir=None, tqdm_handle=None):
    """
    Find the path to the intraoperative photograph for a given patient ID.
    
    Parameters
    ----------
    patient_id : str
        Patient identifier (e.g., 'RESP001')
    picture_root : str
        Root directory containing patient photograph subdirectories
    copy_dir : str, optional
        If provided, copy the found photograph to this directory
    tqdm_handle : tqdm, optional
        tqdm progress bar handle for logging output
    
    Returns
    -------
    str or None
        Path to the photograph file (or copied file if copy_dir specified), 
        or None if no photograph found or user cancelled selection
    """

    # Define logging function
    log = tqdm_handle.write if tqdm_handle else print

    # Define patient root
    patient_root = os.path.join(picture_root, patient_id)
    if not os.path.exists(patient_root):
        raise FileNotFoundError(f"Patient directory not found: {patient_root}")

    # Check for "raw" subdirectory
    if os.path.exists(os.path.join(patient_root, "raw")):
        patient_root = os.path.join(patient_root, "raw")

    # Check expected path, take that one if exists
    expected_paths = [
        os.path.join(patient_root, f"{patient_id}_postresection.jpg"),
        os.path.join(patient_root, f"{patient_id}_Postresection.jpg"),
        os.path.join(patient_root, f"{patient_id}_post.jpg"),
        os.path.join(patient_root, f"{patient_id}_Post.jpg")
    ]
    for expected_path in expected_paths:
        if os.path.exists(expected_path):
            # Optionally copy
            if copy_dir:
                dest_path = os.path.join(copy_dir, f"{patient_id}_post_resection_photo.jpg")
                shutil.copy2(expected_path, dest_path)
                return dest_path
            # Else, just return path
            return expected_path

    # Else, search further: Define search pattern and find files
    search_pattern_1 = os.path.join(patient_root, f"*post*.jpg")
    search_pattern_2 = os.path.join(patient_root, f"*Post*.jpg")
    search_pattern_3 = os.path.join(patient_root, f"*post*.png")
    search_pattern_4 = os.path.join(patient_root, f"*Post*.png")
    matching_files = glob.glob(search_pattern_1) + glob.glob(search_pattern_2) + glob.glob(search_pattern_3) + glob.glob(search_pattern_4)
    if len(matching_files) == 1:
        matching_file = matching_files[0]
    else:
        # Prompt user to select image if multiple or none found
        log(f"Could not uniquely identify photograph for patient ID {patient_id}.")
        log(f"Please select the correct photograph file from the dialog.")
        # Open dialog
        root = tk.Tk()
        root.withdraw()  # hide main window
        root.attributes("-topmost", True)  # bring dialog to front
        # Ask user to select file
        matching_file = filedialog.askopenfilename(
            title=f"Select post-resection photograph for patient {patient_id}",
            initialdir=patient_root,
            filetypes=[("All files", "*.*"), ("JPEG files", "*.jpg *.jpeg"), ("PNG files", "*.png")]
        )
        if not matching_file:
            log("No file selected. Aborting patient.")
            return None

    # Optionally copy
    if copy_dir:
        dest_path = os.path.join(copy_dir, f"{patient_id}_post_resection_photo.jpg")
        shutil.copy2(matching_file, dest_path)
        return dest_path
    # Else, just return path
    else:
        return matching_file

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
    """Compatibility wrapper returning the patient photo set.

    If a patient has multiple photos and no manifest is available, the user is
    asked to select the reference and secondary images interactively so the
    photo-set manifest is constructed on-the-fly.
    """
    if patient_photo_dir is None:
        patient_photo_dir = os.path.join(picture_root, patient_id)

    if not os.path.isdir(patient_photo_dir):
        patient_photo_dir = os.path.join(picture_root, patient_id)

    manifest_candidates = []
    if manifest_path:
        manifest_candidates.append(manifest_path)
    manifest_candidates.extend([
        os.path.join(patient_photo_dir, "photos.yaml"),
        os.path.join(patient_photo_dir, "photos.yml"),
        os.path.join(patient_photo_dir, "photos.json"),
    ])
    for candidate in manifest_candidates:
        if candidate and os.path.exists(candidate):
            return discover_patient_photo_set(
                patient_id=patient_id,
                patient_photo_dir=patient_photo_dir,
                picture_root=picture_root,
                manifest_path=candidate,
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

    if not manifest_path:
        manifest_path = os.path.join(patient_photo_dir, "photos.yaml")
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