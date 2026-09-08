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
import process_photograph
import yaml
from tqdm import tqdm

from photo_registration import PhotoRegistrationMethod, register_photo_set

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

    # Loop
    print(f"Found {len(patient_dirs)} patient directories to process.")
    for patient_dir in tqdm(patient_dirs, desc="Processing patients:", unit="pt"):
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

        tqdm.write(f"\n====== {patient_id}: Start processing ======\n")

        manifest_path = None
        candidate_manifests = [
            os.path.join(patient_dir, "photos.yaml"),
            os.path.join(patient_dir, "photos.yml"),
            os.path.join(patient_dir, "photos.json"),
            os.path.join(photo_data_dir, patient_id, "photos.yaml"),
            os.path.join(photo_data_dir, patient_id, "photos.json"),
        ]
        manifest_path = next((path for path in candidate_manifests if os.path.exists(path)), None)

        photo_set = process_photograph.discover_photo_set(
            patient_id=patient_id,
            picture_root=photo_data_dir,
            patient_photo_dir=os.path.join(photo_data_dir, patient_id),
            manifest_path=manifest_path,
        )
        reference_photo = photo_set.reference_photo
        reference_photo_path = reference_photo.source_path
        post_resection_photo = photo_set.post_resection_photo

        if not os.path.exists(reference_photo_path):
            tqdm.write(f"Reference photo not found for {patient_id}: {reference_photo_path}")
            continue

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        reference_photo_output_path = os.path.join(output_dir, f"{patient_id}_reference_photo.jpg")
        if os.path.abspath(reference_photo_path) != os.path.abspath(reference_photo_output_path):
            try:
                shutil.copy2(reference_photo_path, reference_photo_output_path)
            except Exception:
                reference_photo_output_path = reference_photo_path

        photo_path = reference_photo_output_path

        if post_resection_photo is not None:
            tqdm.write(f"Drawing resection mask for selected post-resection photo of {patient_id}...")
            resection_mask_path_photo = os.path.join(output_dir, f"{patient_id}_{post_resection_photo.photo_id}_masks.npz")
            if not os.path.exists(resection_mask_path_photo):
                process_photograph.draw_photo_masks(
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
                "Only the reference photo is registered to cortex; secondary photos are routed through photo-to-reference registration metadata."
            )
            for secondary in photo_set.secondary_photos:
                if secondary.registration_status == "pending" or secondary.registration_method is not None:
                    tqdm.write(
                        f"Secondary photo '{secondary.photo_id}' is pending or configured for '{secondary.registration_method.value if secondary.registration_method else 'unspecified'}' "
                        "registration; continuing with the reference-photo workflow."
                    )

        if not os.path.exists(mask_path):
            tqdm.write(f"Drawing masks for reference photo of {patient_id}...")
            masks_drawn = process_photograph.draw_photo_masks(photo_path, save_path=mask_path, tqdm_handle=tqdm)
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
        process_photograph.save_photo_set_manifest(photo_set, os.path.join(output_dir, "photo_set_manifest.json"))

        if not masks_drawn:
            tqdm.write(f"Skipping {patient_id} (no masks drawn).")
            continue

        if not os.path.exists(figure_path):
            process_photograph.show_photo_with_masks(photo_path, mask_path, save_path=figure_path, tqdm_handle=tqdm)

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

        viewer_process_id = process_photograph.open_image_viewer(figure_path)

        sys.stdout.write(f"\rProcessing {patient_dir} in 3D Slicer...\033[K")
        sys.stdout.flush()
        result = subprocess.run([
            slicer_executable,
            "--python-script", os.path.join(os.path.dirname(__file__), "slicer_script.py"),
            "--t1_path", t1, "--ribbon_path", ribbon, "--lh_pial_path", lh_pial, "--rh_pial_path", rh_pial,
            "--lh_envelope_path", lh_envelope, "--rh_envelope_path", rh_envelope,
            "--brain_envelope_path", brain_envelope, "--photo_path", photo_path,
            "--mask_path", mask_path, "--output_dir", output_dir,
            "--photo_set_manifest", os.path.join(output_dir, "photo_set_manifest.json")
        ])
        process_photograph.close_image_viewer(viewer_process_id)

        if result.returncode == 0:
            sys.stdout.write(f"\rProcessing {patient_dir} in 3D Slicer... \033[92mDONE\033[0m\033[K\n")
            sys.stdout.flush()
        elif result.returncode == 1:
            sys.stdout.write(f"\rProcessing {patient_dir} in 3D Slicer... \033[93mFAILED\033[0m\033[K\n")
            sys.stdout.flush()
        elif result.returncode == 2:
            sys.stdout.write(f"\rProcessing {patient_dir} in 3D Slicer... \033[94mLOOP ABORTED\033[0m\033[K\n")
            sys.stdout.flush()
            break
        else:
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