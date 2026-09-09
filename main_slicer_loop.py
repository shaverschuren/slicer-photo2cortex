"""
Main loop to open each patient in 3D Slicer for manual photo to MRI registration.
To use, make sure to set the correct paths in the __main__ section.
- `slicer_path`: Path to the Slicer executable.
- `root`: Root directory containing patient subdirectories.
"""

import os
import glob
import shutil
import subprocess
import sys
import photo_preparation
import yaml
from tqdm import tqdm

from util.photo_registration import register_photo_set

def load_config(config_path=os.path.join(os.path.dirname(__file__), "config.yaml")):
    """
    Load configuration from YAML file.
    
    If the config file does not exist, creates a template and exits.
    
    Parameters
    ----------
    config_path : str, optional
        Path to the configuration YAML file. Defaults to 'config.yaml' in the package directory.
    
    Returns
    -------
    dict
        Configuration dictionary with keys: 'slicer_exe_path', 'mri_data_dir', 'photo_data_dir'
    
    Raises
    ------
    FileNotFoundError
        If any of the paths specified in the config file do not exist.
    """
    
    # Check if config exists
    if not os.path.exists(config_path):
        print(f"Software expects config file at '{os.path.abspath(config_path)}'")
        print("Creating template config file. Please edit the paths accordingly and re-run.")
        with open(config_path, "w") as f:
            yaml.dump({
            "slicer_exe_path": "C:\\Path\\To\\Slicer.exe",
            "mri_data_dir": "C:\\Path\\To\\MRI\\Data\\Directory",
            "photo_data_dir": "C:\\Path\\To\\Photograph\\Data\\Directory"
            }, f)
        exit(0)  # Exit so user can edit paths
    
    # Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Check if paths exist
    if not os.path.exists(config["slicer_exe_path"]):
        raise FileNotFoundError(f"Slicer executable not found at '{config['slicer_exe_path']}'")
    if not os.path.exists(config["mri_data_dir"]):
        raise FileNotFoundError(f"Root directory not found at '{config['mri_data_dir']}'")
    if not os.path.exists(config["photo_data_dir"]):
        raise FileNotFoundError(f"Photograph root directory not found at '{config['photo_data_dir']}'")

    return config

def main_slicer_loop(mri_data_dir, photo_data_dir, slicer_executable, patient_dir_regex="RESP*", reprocess=False, process_only_photo=False):
    """
    Open each patient in 3D Slicer for manual photo to MRI registration.
    
    Parameters
    ----------
    mri_data_dir : str
        Root directory containing FreeSurfer patient subdirectories (e.g., RESP001, RESP002, etc.)
    photo_data_dir : str
        Root directory containing photograph subdirectories matching patient IDs
    slicer_executable : str
        Path to the 3D Slicer executable
    patient_dir_regex : str, optional
        Glob pattern for patient directories. Defaults to "RESP*"
    reprocess : bool, optional
        If True, reprocess patients even if resection mask already exists. Defaults to False
    """

    # Get patient dirs
    patient_dirs = sorted(glob.glob(os.path.join(mri_data_dir, patient_dir_regex)))

    # ---------- MAIN LOOP -------------
    # Iterates over each patient directory and process the photos and MRI data
    # ----------------------------------

    print(f"Found {len(patient_dirs)} patient directories to process.")
    for patient_dir in tqdm(patient_dirs, desc="Processing patients", unit="pt"):
        # Get patient ID and set up default paths
        patient_id = os.path.basename(patient_dir)
        output_dir = os.path.join(patient_dir, "photo2cortex_output")
        fs_dir = patient_dir
        t1s = [os.path.join(fs_dir, "mri", fname) for fname in ["T1.mgz", "T1.nii", "T1.nii.gz"]]
        ribbons = [os.path.join(fs_dir, "mri", fname) for fname in ["ribbon.mgz", "ribbon.nii", "ribbon.nii.gz"]]
        lh_pials = [os.path.join(fs_dir, "surf", "lh.pial.T1"), os.path.join(fs_dir, "surf", "lh.pial")]
        rh_pials = [os.path.join(fs_dir, "surf", "rh.pial.T1"), os.path.join(fs_dir, "surf", "rh.pial")]
        lh_envelope = os.path.join(patient_dir, "lh_envelope.stl")
        rh_envelope = os.path.join(patient_dir, "rh_envelope.stl")
        brain_envelope = os.path.join(patient_dir, "brain_envelope.stl")
        mask_path = os.path.join(output_dir, f"{patient_id}_photo_masks.npz")
        figure_path = os.path.join(output_dir, f"{patient_id}_photo_with_masks.png")
        resection_mask_path = os.path.join(output_dir, f"photo2cortex_resection_mask.nii.gz")
        atlas_based_flag_path = os.path.join(output_dir, "atlas_based.txt")
        skip_flag_path = os.path.join(output_dir, "skip.txt")
        photo_set_manifest_path = os.path.join(output_dir, "photo_set_manifest.json")

        tqdm.write(f"\n====== {patient_id}: Start processing ======\n")

        # Single manifest lives in the output dir; reused across runs to avoid re-prompting
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        photo_set = photo_preparation.discover_photo_set(
            patient_id=patient_id,
            picture_root=photo_data_dir,
            patient_photo_dir=os.path.join(photo_data_dir, patient_id),
            manifest_path=photo_set_manifest_path,
        )
        # Get the reference photo from the photo set
        reference_photo = photo_set.reference_photo
        reference_photo_path = reference_photo.source_path
        reference_photo_output_path = os.path.join(output_dir, f"{patient_id}_reference_photo.jpg")
        # A cached manifest may only know the previously-copied output file if the
        # original selection is no longer reachable (e.g. input drive unavailable)
        if not os.path.exists(reference_photo_path) and os.path.exists(reference_photo_output_path):
            reference_photo_path = reference_photo_output_path
        # Skip if not there
        if not os.path.exists(reference_photo_path):
            tqdm.write(f"Reference photo not found for {patient_id}: {reference_photo_path}")
            continue

        if os.path.abspath(reference_photo_path) != os.path.abspath(reference_photo_output_path):
            try:
                shutil.copy2(reference_photo_path, reference_photo_output_path)
            except Exception as e:
                tqdm.write(f"Failed to copy reference photo for {patient_id}: {e}")
                reference_photo_output_path = reference_photo_path

        # Keep the originally-selected file traceable
        reference_photo.metadata.setdefault("original_source_path", reference_photo_path)
        reference_photo.source_path = reference_photo_output_path

        # Use the reference photo output path as the main photo path for further processing
        photo_path = reference_photo_output_path

        # Process post-resection photo if it exists, for generating resected area masks
        post_resection_photo = photo_set.post_resection_photo
        if post_resection_photo is not None:
            tqdm.write(f"Drawing resection mask for selected post-resection photo of {patient_id}...")
            resection_mask_path_photo = os.path.join(output_dir, f"{patient_id}_{post_resection_photo.photo_id}_masks.npz")
            if not os.path.exists(resection_mask_path_photo):
                photo_preparation.draw_photo_masks(
                    post_resection_photo.source_path,
                    save_path=resection_mask_path_photo,
                    tqdm_handle=tqdm,
                    photo_role="secondary",
                    include_resection=True,
                )
            post_resection_photo.masks["path"] = resection_mask_path_photo

        if len(photo_set.secondary_photos) > 0:
            tqdm.write(
                f"Patient {patient_id} has {len(photo_set.secondary_photos)} secondary photo(s). "
                "Drawing outside-area ROI masks for photo-to-reference registration."
            )
            for secondary in photo_set.secondary_photos:
                secondary_mask_path = os.path.join(output_dir, f"{patient_id}_{secondary.photo_id}_masks.npz")
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
            tqdm.write(f"Drawing masks for reference photo of {patient_id}...")
            masks_drawn = photo_preparation.draw_photo_masks(photo_path, save_path=mask_path, tqdm_handle=tqdm)
        else:
            tqdm.write(f"Masks already exist for {patient_id}, skipping drawing.")
            masks_drawn = True

        reference_photo.masks["path"] = mask_path
        registration_dir = os.path.join(output_dir, "registered_photos")
        registration_results = register_photo_set(photo_set, registration_dir)
        tqdm.write(
            f"Photo-to-reference registration completed for {sum(result.status == 'registered' for result in registration_results)} "
            f"of {len(registration_results)} selected secondary photo(s)."
        )
        photo_preparation.save_photo_set_manifest(photo_set, photo_set_manifest_path)

        if not masks_drawn:
            tqdm.write(f"Skipping {patient_id} (no masks drawn).")
            continue

        if not os.path.exists(figure_path):
            photo_preparation.show_photo_with_masks(photo_path, mask_path, save_path=figure_path, tqdm_handle=tqdm)

        if process_only_photo:
            tqdm.write(f"Only processing photograph for {patient_id}, skipping Slicer step.")
            continue

        t1 = next((path for path in t1s if os.path.exists(path)), None)
        ribbon = next((path for path in ribbons if os.path.exists(path)), None)
        lh_pial = next((path for path in lh_pials if os.path.exists(path)), None)
        rh_pial = next((path for path in rh_pials if os.path.exists(path)), None)
        if (not os.path.exists(t1) or not os.path.exists(ribbon) or not os.path.exists(lh_pial) or not os.path.exists(rh_pial)) \
            and not process_only_photo:
            tqdm.write(f"Missing FreeSurfer data for {patient_id}, skipping patient.")
            continue

        # Skip conditions
        if os.path.exists(resection_mask_path) and not reprocess:
            tqdm.write(f"Resection mask already exists for {patient_id}, skipping patient.")
            continue
        if os.path.exists(atlas_based_flag_path) and not reprocess:
            tqdm.write(f"Atlas-based resection mask flagged for {patient_id}, skipping patient.")
            continue
        if os.path.exists(skip_flag_path) and not reprocess:
            tqdm.write(f"Skip flag found for {patient_id}, skipping patient.")
            continue
        if os.path.exists(os.path.join(output_dir, "brain_envelope.vtk")) and not reprocess:
            tqdm.write(f"Registration already done for {patient_id}, skipping patient.")
            continue

        # When processing, show the reference photo to help with manual alignment
        viewer_process_id = photo_preparation.open_image_viewer(figure_path)
        # Open the 3D Slicer process for this patient
        sys.stdout.write(f"\rProcessing {patient_dir} in 3D Slicer...\033[K")
        sys.stdout.flush()
        result = subprocess.run([
            slicer_executable,
            "--python-script", os.path.join(os.path.dirname(__file__), "slicer_script.py"),
            "--t1_path", t1, "--ribbon_path", ribbon, "--lh_pial_path", lh_pial, "--rh_pial_path", rh_pial,
            "--lh_envelope_path", lh_envelope, "--rh_envelope_path", rh_envelope,
            "--brain_envelope_path", brain_envelope, "--photo_path", photo_path,
            "--mask_path", mask_path, "--output_dir", output_dir,
            "--photo_set_manifest", photo_set_manifest_path
        ])
        # When done, close the reference photo viewer
        photo_preparation.close_image_viewer(viewer_process_id)
        # Check result
        if result.returncode == 0:
            # Success and continue
            sys.stdout.write(f"\rProcessing {patient_dir} in 3D Slicer... \033[92mDONE\033[0m\033[K\n")
            sys.stdout.flush()
        elif result.returncode == 1:
            # Failed
            sys.stdout.write(f"\rProcessing {patient_dir} in 3D Slicer... \033[93mFAILED\033[0m\033[K\n")
            sys.stdout.flush()
        elif result.returncode == 2:
            # Loop aborted manually
            sys.stdout.write(f"\rProcessing {patient_dir} in 3D Slicer... \033[94mLOOP ABORTED\033[0m\033[K\n")
            sys.stdout.flush()
            break
        else:
            # Unknown return code
            sys.stdout.write(f"\rProcessing {patient_dir} in 3D Slicer... \033[91mUNKNOWN RETURN CODE {result.returncode}\033[0m\033[K\n")
            sys.stdout.flush()


if __name__ == "__main__":
    # Create or load config, set path variables
    config = load_config()
    slicer_exe_path = config["slicer_exe_path"]
    mri_data_dir = config["mri_data_dir"]
    photo_data_dir = config["photo_data_dir"]
    # Run main function
    main_slicer_loop(mri_data_dir, photo_data_dir, slicer_exe_path)