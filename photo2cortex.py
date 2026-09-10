"""Command-line entry point for photo-to-cortex processing."""

from __future__ import annotations

import argparse
import glob
import os
import shutil
import subprocess
import sys
from enum import Enum

import photo_preparation
import yaml
from tqdm import tqdm

from util.photo_registration import register_photo_set, registration_qc_is_approved
from util.photo_registration_qc import create_registration_qc_montage, review_registration_qc
from util.projection_manifest import PROJECTION_MANIFEST_FILENAME, projection_set_is_complete


class SubjectProcessStatus(Enum):
    COMPLETED = "completed"
    SKIPPED = "skipped"
    FAILED = "failed"
    ABORT_BATCH = "abort_batch"


def subject_id_from_dir(subject_dir):
    """Return the subject identifier represented by a directory path."""
    return os.path.basename(os.path.normpath(os.path.abspath(subject_dir)))


def _default_config_path():
    return os.path.join(os.path.dirname(__file__), "config.yaml")


def load_config(config_path=None, *, validate_batch=True):
    """Load and validate shared configuration, optionally including batch paths."""
    config_path = config_path or _default_config_path()
    if not os.path.exists(config_path):
        print(f"Software expects config file at '{os.path.abspath(config_path)}'")
        print("Creating template config file. Please edit the paths accordingly and re-run.")
        with open(config_path, "w", encoding="utf-8") as file:
            yaml.safe_dump(
                {
                    "slicer_exe_path": r"C:\Path\To\Slicer.exe",
                    "photo_data_dir": r"C:\Path\To\Photograph\Data\Directory",
                    "batch": {
                        "mri_data_dir": r"C:\Path\To\FreeSurfer\Subjects",
                        "subject_dir_regex": "RESP*",
                        "reprocess": False,
                        "process_only_photo": False,
                        "process_only_envelope": False,
                    },
                },
                file,
                sort_keys=False,
            )
        raise FileNotFoundError(f"Configuration template created at '{config_path}'")

    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}

    batch = dict(config.get("batch") or {})
    if "mri_data_dir" in config:
        batch.setdefault("mri_data_dir", config["mri_data_dir"])
    batch.setdefault("subject_dir_regex", "RESP*")
    batch.setdefault("reprocess", False)
    batch.setdefault("process_only_photo", False)
    batch.setdefault("process_only_envelope", False)
    config["batch"] = batch
    config.setdefault("mri_data_dir", batch.get("mri_data_dir"))

    for key, label in (("slicer_exe_path", "Slicer executable"), ("photo_data_dir", "Photograph root directory")):
        value = config.get(key)
        if not value or not os.path.exists(value):
            raise FileNotFoundError(f"{label} not found at '{value}'")
    if validate_batch:
        mri_data_dir = batch.get("mri_data_dir")
        if not mri_data_dir or not os.path.isdir(mri_data_dir):
            raise FileNotFoundError(f"Batch MRI root directory not found at '{mri_data_dir}'")
    return config


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Register photographs or generate FreeSurfer envelopes in 3D Slicer.",
        epilog=(
            "Batch settings come from config.yaml: reprocess repeats completed work, "
            "process_only_photo prepares photographs without Slicer, and "
            "process_only_envelope generates only surface envelopes. Check README.md and config_template.yaml for details."
        ),
    )
    parser.add_argument("subject_dir", nargs="?", help="FreeSurfer subject directory to process instead of batch mode")
    parser.add_argument("--config", default=_default_config_path(), help="Path to the YAML configuration file")
    return parser.parse_args(argv)


def discover_batch_subjects(mri_data_dir, subject_dir_regex="RESP*"):
    """Return matching subject directories in deterministic order."""
    return sorted(
        path for path in glob.glob(os.path.join(os.path.abspath(mri_data_dir), subject_dir_regex)) if os.path.isdir(path)
    )


def process_subject(
    subject_dir,
    *,
    photo_data_dir,
    slicer_executable,
    reprocess=False,
    process_only_photo=False,
    process_only_envelope=False,
):
    """Prepare one subject and run its photo-to-cortex Slicer workflow."""
    subject_dir = os.path.abspath(subject_dir)
    subject_id = subject_id_from_dir(subject_dir)
    output_dir = os.path.join(subject_dir, "photo2cortex_output")
    t1s = [os.path.join(subject_dir, "mri", name) for name in ("T1.mgz", "T1.nii", "T1.nii.gz")]
    ribbons = [os.path.join(subject_dir, "mri", name) for name in ("ribbon.mgz", "ribbon.nii", "ribbon.nii.gz")]
    lh_pials = [os.path.join(subject_dir, "surf", name) for name in ("lh.pial.T1", "lh.pial")]
    rh_pials = [os.path.join(subject_dir, "surf", name) for name in ("rh.pial.T1", "rh.pial")]
    lh_envelope = os.path.join(subject_dir, "lh_envelope.stl")
    rh_envelope = os.path.join(subject_dir, "rh_envelope.stl")
    brain_envelope = os.path.join(subject_dir, "brain_envelope.stl")
    mask_path = os.path.join(output_dir, f"{subject_id}_photo_masks.npz")
    figure_path = os.path.join(output_dir, f"{subject_id}_photo_with_masks.png")
    skip_flag_path = os.path.join(output_dir, "skip.txt")
    photo_set_manifest_path = os.path.join(output_dir, "photo_set_manifest.json")
    projection_manifest_path = os.path.join(output_dir, PROJECTION_MANIFEST_FILENAME)

    tqdm.write(f"\n====== {subject_id}: Start processing ======\n")
    os.makedirs(output_dir, exist_ok=True)

    t1 = next((path for path in t1s if os.path.exists(path)), None)
    lh_pial = next((path for path in lh_pials if os.path.exists(path)), None)
    rh_pial = next((path for path in rh_pials if os.path.exists(path)), None)
    if process_only_envelope:
        if not all((t1, lh_pial, rh_pial)):
            tqdm.write(f"Missing required MRI/surface files for {subject_id}, skipping subject.")
            return SubjectProcessStatus.SKIPPED
        envelope_paths = (lh_envelope, rh_envelope, brain_envelope)
        if all(os.path.exists(path) for path in envelope_paths) and not reprocess:
            tqdm.write(f"Envelopes already exist for {subject_id}, skipping subject.")
            return SubjectProcessStatus.SKIPPED
        try:
            tqdm.write(f"Generating envelopes for {subject_id} in 3D Slicer...")
            result = subprocess.run([
                slicer_executable,
                "--no-main-window",
                "--python-script", os.path.join(os.path.dirname(__file__), "slicer_script.py"),
                "--t1_path", t1,
                "--lh_pial_path", lh_pial,
                "--rh_pial_path", rh_pial,
                "--lh_envelope_path", lh_envelope,
                "--rh_envelope_path", rh_envelope,
                "--brain_envelope_path", brain_envelope,
                "--create_envelope_mode",
            ], capture_output=True, text=True)
        except OSError as error:
            tqdm.write(f"Slicer failed for {subject_id}: {error}")
            return SubjectProcessStatus.FAILED
        if result.returncode == 0:
            return SubjectProcessStatus.COMPLETED
        if result.returncode == 2:
            return SubjectProcessStatus.ABORT_BATCH
        tqdm.write(f"Envelope generation failed for {subject_id}: {result.stderr}")
        return SubjectProcessStatus.FAILED

    try:
        photo_set = photo_preparation.discover_photo_set(
            patient_id=subject_id,
            picture_root=photo_data_dir,
            patient_photo_dir=os.path.join(photo_data_dir, subject_id),
            manifest_path=photo_set_manifest_path,
        )
    except (FileNotFoundError, ValueError, OSError) as error:
        tqdm.write(f"Failed to discover photos for {subject_id}: {error}")
        return SubjectProcessStatus.FAILED

    reference_photo = photo_set.reference_photo
    reference_photo_path = reference_photo.source_path
    reference_photo_output_path = os.path.join(output_dir, f"{subject_id}_reference_photo.jpg")
    if not os.path.exists(reference_photo_path) and os.path.exists(reference_photo_output_path):
        reference_photo_path = reference_photo_output_path
    if not os.path.exists(reference_photo_path):
        tqdm.write(f"Reference photo not found for {subject_id}: {reference_photo_path}")
        return SubjectProcessStatus.FAILED

    if os.path.abspath(reference_photo_path) != os.path.abspath(reference_photo_output_path):
        try:
            shutil.copy2(reference_photo_path, reference_photo_output_path)
        except OSError as error:
            tqdm.write(f"Failed to copy reference photo for {subject_id}: {error}")
            reference_photo_output_path = reference_photo_path
    reference_photo.metadata.setdefault("original_source_path", reference_photo_path)
    reference_photo.source_path = reference_photo_output_path
    photo_path = reference_photo_output_path

    post_resection_photo = photo_set.post_resection_photo
    if post_resection_photo is not None:
        tqdm.write(f"Drawing resection mask for selected post-resection photo of {subject_id}...")
        resection_mask_path = os.path.join(output_dir, f"{subject_id}_{post_resection_photo.photo_id}_masks.npz")
        if not os.path.exists(resection_mask_path):
            photo_preparation.draw_photo_masks(
                post_resection_photo.source_path,
                save_path=resection_mask_path,
                tqdm_handle=tqdm,
                photo_role="secondary",
                include_resection=True,
            )
        post_resection_photo.masks["path"] = resection_mask_path

    if photo_set.secondary_photos:
        tqdm.write(
            f"Subject {subject_id} has {len(photo_set.secondary_photos)} secondary photo(s). "
            "Drawing outside-area ROI masks for photo-to-reference registration."
        )
        for secondary in photo_set.secondary_photos:
            secondary_mask_path = os.path.join(output_dir, f"{subject_id}_{secondary.photo_id}_masks.npz")
            if not os.path.exists(secondary_mask_path):
                photo_preparation.draw_photo_masks(
                    secondary.source_path,
                    save_path=secondary_mask_path,
                    tqdm_handle=tqdm,
                    photo_role="secondary",
                    include_resection=False,
                )
            secondary.masks["path"] = secondary_mask_path

    if not os.path.exists(mask_path):
        tqdm.write(f"Drawing masks for reference photo of {subject_id}...")
        masks_drawn = photo_preparation.draw_photo_masks(photo_path, save_path=mask_path, tqdm_handle=tqdm)
    else:
        tqdm.write(f"Masks already exist for {subject_id}, skipping drawing.")
        masks_drawn = True
    reference_photo.masks["path"] = mask_path

    tqdm.write("Starting photo-to-photo registration...")
    registration_dir = os.path.join(output_dir, "registered_photos")
    registration_results = register_photo_set(photo_set, registration_dir, dof=6)
    tqdm.write(
        f"Photo-to-reference registration completed for {sum(result.status == 'registered' for result in registration_results)} "
        f"of {len(registration_results)} selected secondary photo(s)."
    )
    pending_photos = [result.moving_photo_id for result in registration_results if result.status != "registered"]
    if pending_photos:
        tqdm.write(f"Registration pending/failed for: {', '.join(pending_photos)}. These photos will not receive a projected envelope until registered.")

    selected_auxiliaries = [photo for photo in photo_set.all_photos()[1:] if photo.registration_status in {"registered", "registration_pending", "pending"}]
    if selected_auxiliaries:
        unregistered = [photo.photo_id for photo in selected_auxiliaries if photo.registration_status != "registered"]
        if unregistered:
            tqdm.write(f"Registration incomplete for: {', '.join(unregistered)}. Cannot continue to Slicer.")
            photo_preparation.save_photo_set_manifest(photo_set, photo_set_manifest_path)
            return SubjectProcessStatus.FAILED

        if not registration_qc_is_approved(photo_set):
            qc_path = os.path.join(registration_dir, "registration_qc.png")
            photo_set.registration_qc["montage_path"] = qc_path
            if not os.path.exists(qc_path):
                create_registration_qc_montage(photo_set, registration_results, qc_path)
            if os.path.exists(qc_path):
                decision = review_registration_qc(qc_path, photo_set, registration_results)
                photo_set.registration_qc["montage_path"] = qc_path
                photo_preparation.save_photo_set_manifest(photo_set, photo_set_manifest_path)
                if decision == "approved":
                    tqdm.write("Registration QC approved for current auxiliary registrations.")
                elif decision == "rejected":
                    tqdm.write("Registration QC rejected; subject will not continue to Slicer.")
                    return SubjectProcessStatus.FAILED
                else:
                    tqdm.write("Registration QC was not explicitly approved; subject will not continue to Slicer.")
                    return SubjectProcessStatus.FAILED
            else:
                tqdm.write("Registration QC montage could not be generated; subject will not continue to Slicer.")
                photo_preparation.save_photo_set_manifest(photo_set, photo_set_manifest_path)
                return SubjectProcessStatus.FAILED
        else:
            tqdm.write("Registration QC already approved; reusing cached registrations.")
            photo_set.registration_qc.setdefault("montage_path", os.path.join(registration_dir, "registration_qc.png"))

    photo_preparation.save_photo_set_manifest(photo_set, photo_set_manifest_path)

    if not masks_drawn:
        tqdm.write(f"Skipping {subject_id} (no masks drawn).")
        return SubjectProcessStatus.SKIPPED
    if not os.path.exists(figure_path):
        photo_preparation.show_photo_with_masks(photo_path, mask_path, save_path=figure_path, tqdm_handle=tqdm)
    if process_only_photo:
        tqdm.write(f"Only processing photograph for {subject_id}, skipping Slicer step.")
        return SubjectProcessStatus.COMPLETED

    ribbon = next((path for path in ribbons if os.path.exists(path)), None)
    if not all((t1, ribbon, lh_pial, rh_pial)):
        tqdm.write(f"Missing FreeSurfer data for {subject_id}, skipping subject.")
        return SubjectProcessStatus.SKIPPED
    if os.path.exists(skip_flag_path) and not reprocess:
        tqdm.write(f"Skip flag found for {subject_id}, skipping subject.")
        return SubjectProcessStatus.SKIPPED
    if projection_set_is_complete(projection_manifest_path, photo_set_manifest_path) and not reprocess:
        tqdm.write(f"Photo-projection set already complete for {subject_id}, skipping subject.")
        return SubjectProcessStatus.SKIPPED

    viewer_process = photo_preparation.open_image_viewer(figure_path)
    try:
        sys.stdout.write(f"\rProcessing {subject_dir} in 3D Slicer...\033[K")
        sys.stdout.flush()
        result = subprocess.run([
            slicer_executable,
            "--python-script", os.path.join(os.path.dirname(__file__), "slicer_script.py"),
            "--t1_path", t1, "--ribbon_path", ribbon, "--lh_pial_path", lh_pial, "--rh_pial_path", rh_pial,
            "--lh_envelope_path", lh_envelope, "--rh_envelope_path", rh_envelope,
            "--brain_envelope_path", brain_envelope, "--photo_path", photo_path,
            "--mask_path", mask_path, "--output_dir", output_dir,
            "--photo_set_manifest", photo_set_manifest_path,
        ])
    except OSError as error:
        tqdm.write(f"Slicer failed for {subject_id}: {error}")
        return SubjectProcessStatus.FAILED
    finally:
        photo_preparation.close_image_viewer(viewer_process)

    if result.returncode == 0:
        sys.stdout.write(f"\rProcessing {subject_dir} in 3D Slicer... \033[92mDONE\033[0m\033[K\n")
        return SubjectProcessStatus.COMPLETED
    if result.returncode == 1:
        sys.stdout.write(f"\rProcessing {subject_dir} in 3D Slicer... \033[93mFAILED\033[0m\033[K\n")
        return SubjectProcessStatus.FAILED
    if result.returncode == 2:
        sys.stdout.write(f"\rProcessing {subject_dir} in 3D Slicer... \033[94mLOOP ABORTED\033[0m\033[K\n")
        return SubjectProcessStatus.ABORT_BATCH
    sys.stdout.write(f"\rProcessing {subject_dir} in 3D Slicer... \033[91mUNKNOWN RETURN CODE {result.returncode}\033[0m\033[K\n")
    return SubjectProcessStatus.FAILED


def process_batch(config):
    """Discover configured subjects and delegate each one to process_subject."""
    batch = config["batch"]
    subject_dirs = discover_batch_subjects(batch["mri_data_dir"], batch["subject_dir_regex"])
    print(f"Found {len(subject_dirs)} subject directories to process.")
    for subject_dir in tqdm(subject_dirs, desc="Processing subjects", unit="subject"):
        status = process_subject(
            subject_dir,
            photo_data_dir=config["photo_data_dir"],
            slicer_executable=config["slicer_exe_path"],
            reprocess=batch["reprocess"],
            process_only_photo=batch["process_only_photo"],
            process_only_envelope=batch.get("process_only_envelope", False),
        )
        if status is SubjectProcessStatus.ABORT_BATCH:
            break
    return 0


def exit_code_for_status(status):
    return 0 if status in (SubjectProcessStatus.COMPLETED, SubjectProcessStatus.SKIPPED) else 1


def main(argv=None):
    args = parse_args(argv)
    config = load_config(args.config, validate_batch=args.subject_dir is None)
    if args.subject_dir:
        if not os.path.isdir(args.subject_dir):
            raise FileNotFoundError(f"Subject directory not found at '{args.subject_dir}'")
        status = process_subject(
            args.subject_dir,
            photo_data_dir=config["photo_data_dir"],
            slicer_executable=config["slicer_exe_path"],
            reprocess=config["batch"]["reprocess"],
            process_only_photo=config["batch"]["process_only_photo"],
            process_only_envelope=config["batch"]["process_only_envelope"],
        )
        return exit_code_for_status(status)
    return process_batch(config)


if __name__ == "__main__":
    sys.exit(main())