"""
Utility package for 3D Slicer visualization and interaction.

This package is organized into focused modules:
- io: Scene and file I/O operations
- geometry: Geometry and matrix utilities
- projection: Photo projection onto surfaces
- interaction: UI, camera, and interaction handling
- photo_registration: Photo-to-reference registration algorithms

`io`, `geometry`, `projection`, and `interaction` require Slicer's `vtk`/`slicer`
modules and are only importable from within 3D Slicer. `photo_registration` has
no such dependency and remains importable from a plain Python environment (e.g.
`main_slicer_loop.py`), so the Slicer-only imports below are best-effort.
"""

__all__ = []

try:  # pragma: no cover - only available inside 3D Slicer
    from .io import save_scene_to_directory, create_envelopes, load_stl_surface, write_stl_surface
    from .geometry import (
        vtkMatrixToNumpy, numpyToVtkMatrix, extractRotationScale, rotationFromVectors,
        get_poly_normals, subdivide_model, sample_scalar_along_normals,
        ras_to_lps_polydata, load_photo_masks
    )
    from .projection import Projection
    from .interaction import (
        PhotoTransformObserver, setup_interactive_transform,
        center_camera_on_projection, setup_ui_widgets, setup_interactor
    )
except ImportError:
    pass
else:
    __all__ += [
        # I/O
        'save_scene_to_directory', 'create_envelopes', 'load_stl_surface', 'write_stl_surface',
        # Geometry
        'vtkMatrixToNumpy', 'numpyToVtkMatrix', 'extractRotationScale', 'rotationFromVectors',
        'get_poly_normals', 'subdivide_model', 'sample_scalar_along_normals',
        'ras_to_lps_polydata', 'load_photo_masks',
        # Projection
        'Projection',
        # Interaction
        'PhotoTransformObserver', 'setup_interactive_transform',
        'center_camera_on_projection', 'setup_ui_widgets', 'setup_interactor',
    ]

try:  # pragma: no cover - photo_preparation's GUI deps may be unavailable in Slicer
    from .photo_registration import (
        PhotoRegistrationResult, register_photo_to_reference, register_photo_set,
        save_registration_result, load_photo_registration
    )
except ImportError:
    pass
else:
    __all__ += [
        'PhotoRegistrationResult', 'register_photo_to_reference', 'register_photo_set',
        'save_registration_result', 'load_photo_registration',
    ]
