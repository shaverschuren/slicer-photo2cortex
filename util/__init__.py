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
    from .io import save_scene_to_directory, create_envelopes, load_stl_surface, write_stl_surface, load_photo_volume
    from .geometry import (
        vtkMatrixToNumpy, numpyToVtkMatrix, extractRotationScale, rotationFromVectors,
        get_poly_normals, subdivide_model, sample_scalar_along_normals,
        ras_to_lps_polydata, load_photo_masks, clone_model_node
    )
    from .projection import Projection, create_textured_plane
    from .interaction import (
        PhotoTransformObserver, setup_interactive_transform,
        center_camera_on_projection, setup_ui_widgets, setup_interactor,
        ensure_photo_projection_state, finalize_photo_projections, save_photo_projection_scene
    )
    from .photo_state import PhotoProjectionState
except ImportError:
    pass
else:
    __all__ += [
        # I/O
        'save_scene_to_directory', 'create_envelopes', 'load_stl_surface', 'write_stl_surface', 'load_photo_volume',
        # Geometry
        'vtkMatrixToNumpy', 'numpyToVtkMatrix', 'extractRotationScale', 'rotationFromVectors',
        'get_poly_normals', 'subdivide_model', 'sample_scalar_along_normals',
        'ras_to_lps_polydata', 'load_photo_masks', 'clone_model_node',
        # Projection
        'Projection', 'create_textured_plane',
        # Interaction
        'PhotoTransformObserver', 'setup_interactive_transform',
        'center_camera_on_projection', 'setup_ui_widgets', 'setup_interactor',
        'ensure_photo_projection_state', 'finalize_photo_projections', 'save_photo_projection_scene',
        # Photo projection state
        'PhotoProjectionState',
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

# No slicer/vtk/cv2 dependency: importable from ordinary Python and from Slicer alike.
from .projection_manifest import (
    PROJECTION_MANIFEST_FILENAME, build_projection_manifest, save_projection_manifest,
    load_projection_manifest, projection_set_is_complete
)
__all__ += [
    'PROJECTION_MANIFEST_FILENAME', 'build_projection_manifest', 'save_projection_manifest',
    'load_projection_manifest', 'projection_set_is_complete',
]
