"""
photo2cortex
===========

Utilities and scripts for registering photographs to MRI-derived cortical models in 3D Slicer.

This package groups helper functions for:
- `FreeSurfer` surface IO and envelope handling
- Interactive photo plane transforms in Slicer
- Projection of surface masks into volumes
- Photo preprocessing and automatic alignment utilities
- Batch runners to open Slicer for multiple patients

Usage
-----
- The software relies on `FreeSurfer` outputs and expects subject directories to follow `FreeSurfer`'s structure.
- The main entry point is `photo2cortex.py`, which supports batch and single-subject processing.
- For quicker processing, set `batch.process_only_envelope: true` to precompute surface envelopes before photo registration.
- Most of the code is meant to be run within 3D Slicer's Python environment, which is handled automatically by the batch scripts.

Notes
-----
- The auto-alignment functionality (in `optimizer.py`) is *not* stable and requires some more work.
- The code has been tested on `Windows 11` with `Python 3.13`, `3D Slicer 5.8.1` and `SlicerFreesurfer` extension `05eccb9`.

Author: Sjors Verschuren

Date: October 2025
"""