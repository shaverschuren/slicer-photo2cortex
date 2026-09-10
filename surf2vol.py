"""Module to project a color-coded surface mask into a grey matter volume along surface normals within Slicer-Python."""

import numpy as np
import slicer  # type: ignore
import vtk  # type: ignore
from vtkmodules.util import numpy_support  # type: ignore
from scipy.ndimage import binary_closing  # type: ignore

def project_surface_to_volume_mask(surfaceNode, ribbonVolumeNode, max_depth_mm=20.0, include_wm=True,
                                   target_rgb=(1,128,1), gm_labels=[3, 42], wm_labels=[2, 41],
                                   scalar_name="ProjectedPhoto", samples=None,
                                   morph_closing=True, closing_radius_mm=3., visualize=True):
    """
    Projects a color-coded surface mask (RGB) into a grey matter volume
    strictly along the surface normals, using specified GM labels.
    
    Parameters
    ----------
    surfaceNode : vtkMRMLModelNode
        Surface containing RGB scalars.
    ribbonVolumeNode : vtkMRMLLabelMapVolumeNode
        Volume containing grey matter labels (e.g., ribbon.mgz).
    max_depth_mm : float
        Maximum distance along normals to project.
    include_wm : bool
        Whether to include white matter labels in the projection.
    target_rgb : tuple
        RGB color of surface mask to select.
    gm_labels : list of int
        Grey matter labels in the volume to consider.
    wm_labels : list of int
        White matter labels in the volume to consider if include_wm is True.
    scalar_name : str
        Name of the scalar array on the surface.
    samples : int | None
        Number of samples along each normal. If None, defaults to 2 * max_depth_mm.
    morph_closing : bool
        Whether to apply morphological closing to the resulting mask.
    closing_radius_mm : float
        Size of the structuring element for morphological closing in mm.
    visualize : bool
        Whether to visualize the resulting mask in Slicer.
    """

    # Setup default samples
    if samples is None:
        samples = int(max_depth_mm * 2)  # 0.5 mm steps

    # Announce
    volume_type = "grey matter" if not include_wm else "grey and white matter"
    print(f"Projecting surface mask along normals into {volume_type} volume up to {max_depth_mm} mm depth...")

    print("--> Masking surface data...")
    # --- Extract RGB scalar array ---
    pd = surfaceNode.GetPolyData()
    array = pd.GetPointData().GetArray(scalar_name)
    if array is None:
        raise RuntimeError(f"Surface node does not have scalar array '{scalar_name}'")
    rgb_array = numpy_support.vtk_to_numpy(array)
    if rgb_array.ndim == 1:
        rgb_array = rgb_array.reshape(-1, 3)

    # --- Select points matching target RGB ---
    mask_indices = np.where(np.all(rgb_array == target_rgb, axis=1))[0]
    if len(mask_indices) == 0:
        raise RuntimeError(f"No points found with RGB = {target_rgb}")
    points = np.array([pd.GetPoint(i) for i in mask_indices])

    print(f"--> Computing normals at surface points...")
    # --- Compute normals at surface points ---
    normal_filter = vtk.vtkPolyDataNormals()
    normal_filter.SetInputData(pd)
    normal_filter.ComputePointNormalsOn()
    normal_filter.Update()
    normals_vtk = normal_filter.GetOutput().GetPointData().GetNormals()
    normals = np.array([normals_vtk.GetTuple(i) for i in mask_indices])
    normals /= np.linalg.norm(normals, axis=1)[:, None]

    print("--> Preparing volume for projection...")
    # --- Prepare volume info ---
    ribbon_array = slicer.util.arrayFromVolume(ribbonVolumeNode).astype(np.int32)  # Ensure signed datatype (overflow)
    ras2ijk = vtk.vtkMatrix4x4()
    ribbonVolumeNode.GetRASToIJKMatrix(ras2ijk)
    mask_array = np.zeros_like(ribbon_array, dtype=np.uint8)

    print("--> Projecting along normals...")
    # --- Prepare label list ---
    projection_labels = gm_labels.copy()
    if include_wm:
        projection_labels.extend(wm_labels)
    # --- Sample along normals ---
    for pt, nrm in zip(points, normals):
        for t in np.linspace(0, max_depth_mm, samples):
            sample_ras = pt + t * nrm  # inward along normal
            ras_h = np.array([*sample_ras, 1.0])
            ijk_h = ras2ijk.MultiplyPoint(ras_h)
            k, j, i = [int(round(c)) for c in ijk_h[:3]]  # I have no idea why this is swapped, but it works. 
            if (0 <= i < ribbon_array.shape[0] and
                0 <= j < ribbon_array.shape[1] and
                0 <= k < ribbon_array.shape[2]):
                if any(np.abs(ribbon_array[i,j,k] - label) <= 1e-2 for label in projection_labels):
                    mask_array[i,j,k] = 1
        slicer.app.processEvents()  # Keep UI responsive

    # --- Optional morphological closing ---
    if morph_closing:
        print("--> Applying morphological closing to mask...")
        slicer.app.processEvents() 

        # Get voxel spacing
        spacing = ribbonVolumeNode.GetSpacing()
        # Compute kernel half-size (radius) in voxels for each axis
        se_radii_vox = [
            max(1, int(np.ceil(closing_radius_mm / s)))
            for s in spacing
        ]
        # Create a grid for the 3D ellipsoidal structuring element
        grid = np.ogrid[
            -se_radii_vox[0]:se_radii_vox[0] + 1,
            -se_radii_vox[1]:se_radii_vox[1] + 1,
            -se_radii_vox[2]:se_radii_vox[2] + 1,
        ]
        # Equation of an ellipsoid scaled by voxel spacing
        ellipsoid = (
            (grid[0] * spacing[0]) ** 2 +
            (grid[1] * spacing[1]) ** 2 +
            (grid[2] * spacing[2]) ** 2
        ) <= closing_radius_mm ** 2
        se = ellipsoid.astype(np.uint8)

        # Perform binary closing
        mask_array = binary_closing(mask_array, structure=se).astype(np.uint8)

    # --- Create output labelmap ---
    labelNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode", f"projectedMask_{max_depth_mm}mm")
    slicer.util.updateVolumeFromArray(labelNode, mask_array)
    labelNode.CopyOrientation(ribbonVolumeNode)

    # Create segmentation node + display node
    segmentationNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", "photo2cortex_resection_mask")
    segmentationNode.CreateDefaultDisplayNodes()
    # Import the labelmap volume into the segmentation
    slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(labelNode, segmentationNode)

    # Visualize segmentation if requested
    if visualize:
        # Create surf representation and show in 3D viewer
        segmentationNode.CreateClosedSurfaceRepresentation()
        displayNode = segmentationNode.GetDisplayNode()
        displayNode.SetVisibility(True)
        displayNode.SetVisibility3D(True)
        # Force a render
        slicer.app.layoutManager().threeDWidget(0).threeDView().scheduleRender()

    # Remove intermediate labelmap
    slicer.mrmlScene.RemoveNode(labelNode)

    print(f"DONE. Projected surface RGB mask {target_rgb} along normals into grey matter up to {max_depth_mm} mm.")
    print(f"Segmentation node created: {segmentationNode.GetName()}")
    print("You can access and tweak it via the Segmentations module or via `slicer.util.getNode()`.")
    return segmentationNode

def save_resection_mask(segmentationNode, output_path):
    """
    Save a segmentation node as a NIfTI (.nii or .nii.gz) file.

    Parameters
    ----------
    segmentationNode : vtkMRMLSegmentationNode
        The segmentation node to save.
    output_path : str
        Path to the output NIfTI file.
    """

    print(f"Saving segmentation to {output_path}...")

    # --- Create a new empty labelmap node ---
    labelmapVolumeNode = slicer.mrmlScene.AddNewNodeByClass(
        "vtkMRMLLabelMapVolumeNode",
        segmentationNode.GetName() + "_LabelmapTmp"
    )

    # --- Export all segments ---
    slicer.modules.segmentations.logic().ExportAllSegmentsToLabelmapNode(
        segmentationNode,
        labelmapVolumeNode
    )

    # --- Save to disk ---
    success = slicer.util.saveNode(labelmapVolumeNode, output_path)
    if not success:
        raise RuntimeError(f"Failed to save segmentation to {output_path}.")

    print("Segmentation saved successfully:", output_path)

    # --- Clean up ---
    slicer.mrmlScene.RemoveNode(labelmapVolumeNode)
