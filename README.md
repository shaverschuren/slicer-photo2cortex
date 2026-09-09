# Slicer Photo2Cortex

Utilities and scripts for manual registration of intraoperative photographs to MRI-derived cortical models in 3D Slicer.

## Overview

The core concept is a single reference photo per patient:

```text
                      secondary photo A
                             |
                             | 2D registration
                             v
MRI / FreeSurfer <---- reference photograph <---- secondary photo B
       ^                    |
       |                    |
       +--- photo→cortex ---+
```

The reference photograph is the one manually aligned to the cortex. Every secondary photograph is first mapped into the reference-photo pixel coordinate system and then reuses the same reference-photo→cortex geometry.

This architecture explicitly separates:

- reference photo → cortex registration (manual, one per patient)
- secondary photo → reference photo registration (feature-based projective registration)
- optimizer.py, which remains a future reference-photo→cortex optimisation tool and is not the same as 2D photo-to-photo registration

### Key Features

- **Reference-photo-first workflow**: exactly one photo defines the cortex geometry
- **Secondary-photo routing**: secondary photos are registered into the reference-photo grid before cortex projection
- **FreeSurfer Integration**: Surface IO and envelope handling for cortical models
- **Interactive Photo Alignment**: Manual registration tools in 3D Slicer with transform controls
- **Surface-to-Volume Projection**: Convert surface masks into volumetric representations
- **Photo Preprocessing**: Automated tools for image preparation and initial alignment
- **Batch Processing**: Process multiple patients efficiently with automated loops
- **Envelope Generation**: Create brain surface envelopes for visualization

## Prerequisites

### Required Software

- **[FreeSurfer](https://surfer.nmr.mgh.harvard.edu/)**: For cortical surface reconstruction
  - Patient data must follow FreeSurfer's directory structure
- **[3D Slicer](https://www.slicer.org/) 5.8.1+**: Main visualization and registration platform
  - Tested with version 5.8.1
- **[SlicerFreeSurfer Extension](https://github.com/PerkLab/SlicerFreeSurfer)**: For FreeSurfer surface import
  - Tested with commit `05eccb9`
- **[MATLAB](https://www.mathworks.com/products/matlab.html)**: For envelope creation
  - Required for `create_envelopes.m` script

### Python Dependencies

- Python 3.13+ (tested with Python 3.13)
- Core libraries:
  - `numpy` - Numerical computing
  - `scipy` - Optimization and image processing
  - `nibabel` - Neuroimaging file I/O
  - `matplotlib` - Visualization and interactive tools
  - `Pillow (PIL)` - Image processing
  - `PyYAML` - Configuration management
  - `tqdm` - Progress bars
  - `tkinter` - GUI file dialogs
  - `vtk` - 3D visualization (bundled with Slicer)

Additional dependencies (installed automatically when needed):
- `opencv-python` - Computer vision for auto-alignment

### Operating System

- Tested on **Windows 11**
- Should work on macOS and Linux with minor modifications to path handling

## Installation

1. **Clone the repository**:
   ```bash
  git clone https://github.com/shaverschuren/slicer-photo2cortex.git
  cd slicer-photo2cortex
   ```

2. **Install Python dependencies**:
   ```bash
   pip install numpy scipy nibabel matplotlib pillow pyyaml tqdm
   ```
   
   Note: VTK and Slicer-specific modules are provided by 3D Slicer's Python environment.

3. **Install 3D Slicer**:
   - Download from [slicer.org](https://www.slicer.org/)
   - Install the SlicerFreeSurfer extension via the Extension Manager

4. **Install MATLAB** (if not already installed):
   - Required for envelope creation functionality

## Configuration

Before running the scripts, you need to configure the paths to your data and software:

1. **Copy the configuration template**:
   - The first time you run `main_slicer_loop.py`, it will create a `config.yaml` file
   - Alternatively, copy `config_template.yaml` to `config.yaml`

2. **Edit `config.yaml`** with your system paths:
   ```yaml
   slicer_exe_path: C:\Path\To\Slicer.exe
   mri_data_dir: C:\Path\To\FreeSurfer\Subjects\Directory
  photo_data_dir: C:\Path\To\Photographs\Root\Directory
   ```

   - `slicer_exe_path`: Full path to the Slicer executable
   - `mri_data_dir`: Root directory containing FreeSurfer subject folders (e.g., `RESP0001`, `RESP0002`, etc.)
  - `photo_data_dir`: Root directory containing corresponding photograph folders

## Usage

### Quick Start

The typical workflow consists of two main steps:

1. **Pre-compute Surface Envelopes** (Optional but Recommended):
   ```python
   python fs_envelope_loop.py
   ```
   This pre-generates envelope STL files for all patients. Without this step, envelopes are created on-the-fly, adding ~30 seconds per patient.

2. **Run Manual Registration Loop**:
   ```python
   python main_slicer_loop.py
   ```
   This opens 3D Slicer for each patient to perform manual photo-to-MRI registration.

### Photo set / reference workflow

A patient may contain several images, but one photo is designated as the reference photo. It is the only photograph manually aligned to the cortical surface. Secondary photographs are not independently registered to cortex; instead they are mapped into the reference-photo image grid using a 2D transform, and then the same photo→cortex projection is reused.

A patient photo set may be represented by a manifest such as:

```yaml
reference: pre_resection

photos:
  - id: pre_resection
    path: pre_resection.jpg
    role: reference

  - id: grid_configuration_2
    path: grid2.jpg
    role: secondary
    registration_method: projective_8dof

  - id: post_resection
    path: post_resection.jpg
    role: secondary
    registration_method: nonlinear
```

If a patient contains exactly one usable photograph and no manifest, that single photo is automatically treated as the reference photograph. If multiple photos are present and no reference is specified, the workflow stops with a clear error instead of guessing.

### Detailed Workflow

#### Step 1: Envelope Generation (Optional)

Run this first to save time during the main registration process:

```python
python fs_envelope_loop.py
```

This script:
- Scans for FreeSurfer patient directories (matching `RESP*` pattern)
- Opens Slicer in no-main-window mode for each patient
- Calls MATLAB's `create_envelopes.m` to generate:
  - `lh_envelope.stl` - Left hemisphere envelope
  - `rh_envelope.stl` - Right hemisphere envelope  
  - `brain_envelope.stl` - Whole brain envelope
- Skips patients that already have envelopes

#### Step 2: Photo Registration

Run the main registration loop:

```python
python main_slicer_loop.py
```

This script:
- Loads configuration from `config.yaml`
- Finds all patient directories matching the pattern
- For each patient:
  - Locates the corresponding photograph
  - Preprocesses the image (manual drawing of resection + outside)
  - Opens 3D Slicer with:
    - T1 MRI volume
    - FreeSurfer pial surfaces (left/right hemispheres)
    - Brain envelopes
    - Textured photo plane with transform controls
  - Enables manual alignment via Slicer's transform widget
  - Allows annotation with markups curves
  - Saves registration results

### Interactive Registration in Slicer

When Slicer opens for each patient:

1. **View Setup**: The scene loads with MRI, surfaces, and the photo plane
2. **Transform Control**: Use Slicer's transform widget to align the photo:
   - Translate (move position)
   - Rotate (adjust orientation)
   - The photo appears as a textured plane in 3D space
3. **Volumetric Mask Creation**: Project the now-aligned surface mask to the underlying brain volume
4. **Export**: Save scene and volumetric resection mask

### Advanced: Auto-Alignment (Experimental)

The `optimizer.py` module includes experimental auto-alignment functionality:

```python
import optimizer
# Auto-alignment code (see optimizer.py for details)
```

⚠️ **Warning**: The auto-alignment feature is not yet stable and requires additional development before production use.

## Project Structure

```
photo2cortex/
├── __init__.py                 # Package initialization and documentation
├── config_template.yaml        # Configuration template
├── main_slicer_loop.py        # Main entry point for registration workflow
├── fs_envelope_loop.py        # Pre-compute envelopes for all patients
├── slicer_script.py           # Slicer automation script (runs inside Slicer)
├── photo_preparation.py      # Photo preprocessing and file handling
├── surf2vol.py                # Surface-to-volume projection utilities
├── optimizer.py               # Auto-alignment optimization (experimental)
├── util/                      # Utility package with specialized modules
│   ├── __init__.py            # Package exports and documentation
│   ├── io.py                  # Scene and file I/O operations
│   ├── geometry.py            # Geometry and matrix utilities
│   ├── projection.py          # Photo projection onto surfaces
│   └── interaction.py         # UI, camera, and interaction handling
├── MATLAB/
│   └── create_envelopes.m     # MATLAB script for envelope creation
└── .gitignore                 # Git ignore rules
```

### Module Descriptions

- **main_slicer_loop.py**: Orchestrates the batch processing workflow, loading configuration and iterating through patients
- **fs_envelope_loop.py**: Standalone script to pre-generate surface envelopes
- **slicer_script.py**: Executed inside Slicer's Python environment to set up the registration scene
- **photo_preparation.py**: Handles photograph file discovery, copying, and preprocessing
- **surf2vol.py**: Converts FreeSurfer surface masks to volumetric representations
- **util/**: Utility package containing specialized modules:
  - **io.py**: Scene and file I/O operations, STL handling, envelope creation
  - **geometry.py**: Matrix conversions, rotation utilities, VTK polydata operations
  - **projection.py**: Photo projection onto 3D surfaces with perspective/orthographic support
  - **interaction.py**: UI widgets, camera controls, and interactive transform handling
- **optimizer.py**: Experimental auto-alignment using image registration techniques
- **create_envelopes.m**: MATLAB function to create brain surface envelopes from pial surfaces

## Data Organization

### Expected Directory Structure

Your FreeSurfer subjects should be organized as:

```
mri_data_dir/
├── RESP0001/
│   ├── mri/
│   │   └── T1.nii
│   └── surf/
│       ├── lh.pial
│       ├── rh.pial
│       ├── lh_envelope.stl (generated)
│       ├── rh_envelope.stl (generated)
│       └── brain_envelope.stl (generated)
├── RESP0002/
│   └── ...
└── ...
```

Photographs should be organized with matching patient IDs:

```
photo_data_dir/
├── RESP0001/
│   └── photo.jpg (or similar)
├── RESP0002/
│   └── photo.jpg
└── ...
```

## Notes and Limitations

### Known Issues

- **Auto-alignment**: The automatic alignment functionality in `optimizer.py` is experimental and not yet stable. Manual alignment is recommended for production use.
  
### Performance Tips

- **Pre-compute Envelopes**: Run `fs_envelope_loop.py` first to avoid 30-second delays per patient
- **Batch Processing**: The scripts are designed for batch processing; configure all paths once and process multiple patients efficiently

### Platform Notes

- **Path Separators**: The code handles both Windows (`\`) and Unix (`/`) path separators
- **File Viewers**: Cross-platform support for image viewing (Windows Explorer, macOS Preview, Linux EOG)

### Requirements

- FreeSurfer output must include:
  - `mri/T1.nii` (or T1.mgz, will be converted)
  - `surf/lh.pial` and `surf/rh.pial`
- MATLAB with the following toolboxes:
  - Basic MATLAB (for `stlread`, array operations)
  - Image Processing Toolbox (for morphological operations)

## Development and Testing

This software was developed and tested with:
- **OS**: Windows 11
- **Python**: 3.13
- **3D Slicer**: 5.8.1
- **SlicerFreeSurfer**: commit `05eccb9`

## Author

**Sjors Verschuren**  
October 2025

## Acknowledgments

This software builds upon:
- [FreeSurfer](https://surfer.nmr.mgh.harvard.edu/) for cortical surface reconstruction
- [3D Slicer](https://www.slicer.org/) for visualization and registration
- [SlicerFreeSurfer](https://github.com/PerkLab/SlicerFreeSurfer) extension for FreeSurfer integration
