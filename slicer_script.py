# slicer_script.py
# Author: Sjors Verschuren
# Date: October 2025

"""
Slicer automation script for photo-to-MRI registration.

This script is intended to be run inside 3D Slicer via the --python-script argument.
It automates the loading and setup for manual registration of an intraoperative 
photograph to MRI.

Developed and tested with 3D Slicer 5.8.1 and SlicerFreeSurfer 05eccb9.

Features:
- Loads T1 volume, FreeSurfer pial surfaces, envelope surfaces (or creates them), 
  and intraoperative photograph
- Creates a textured plane with the photograph
- Adds a linear transform for interactive photo alignment
- Provides keyboard shortcuts for common operations
- Supports projection of aligned photo onto cortical surface
- Exports volumetric resection masks
"""

import os
import sys
import json
import argparse
import slicer
import numpy as np
import vtk
import nibabel as nib
import util
from NiBabelModelIO import VolGeom  # type: ignore


def load_photo_set_manifest(photo_set_manifest_path):
    """Load a patient photo-set manifest for multi-photo workflows.

    The reference image remains the only photo used for manual cortex alignment;
    secondary photographs are represented as metadata and may optionally be
    transformed into reference coordinates later.
    """
    if not photo_set_manifest_path or not os.path.exists(photo_set_manifest_path):
        return None
    with open(photo_set_manifest_path, "r", encoding="utf-8") as fh:
        return json.load(fh)

def read_fs_surf(path, modelNode, calculateNormals=True):
    """
    Read a FreeSurfer surface file and set it to the provided model node.
    
    Parameters
    ----------
    path : str
        Path to the FreeSurfer surface file (e.g., lh.pial, rh.pial)
    modelNode : vtkMRMLModelNode
        Slicer model node to store the loaded surface
    calculateNormals : bool, optional
        Whether to compute point normals for the surface. Defaults to True
    
    Returns
    -------
    bool
        True if loading succeeded, False otherwise
    """

    try:
        points, polys, metadata = nib.freesurfer.io.read_geometry(path, True)

        geom = VolGeom.from_surf_footer(metadata)

        points = nib.affines.apply_affine(geom.tkreg2scanner(), points).astype('f4')
    
        polyData = vtk.vtkPolyData()
        pointsVTK = vtk.vtkPoints()
        pointsVTK.SetData(vtk.util.numpy_support.numpy_to_vtk(points))
        polyData.SetPoints(pointsVTK)
    
        cellArray = vtk.vtkCellArray()
        polys = polys.astype(np.int64)
        polys = polys.flatten()
        idTypeArray =  vtk.util.numpy_support.numpy_to_vtkIdTypeArray(polys, deep=True)
        cellArray.SetData(3, idTypeArray)
        polyData.SetPolys(cellArray)

        if calculateNormals:
            normals = vtk.vtkPolyDataNormals()
            normals.SetInputData(polyData)
            normals.ComputePointNormalsOn()
            normals.SplittingOff()
            normals.ConsistencyOn()
            normals.AutoOrientNormalsOn()
            normals.Update()
            polyData = normals.GetOutput()

        # attempt to read corresponding .curv and .sulc files
        curv_path = path.replace("pial", "curv")
        sulc_path = path.replace("pial", "sulc")
        try:
            if os.path.exists(curv_path):
                curv = nib.freesurfer.io.read_morph_data(curv_path)
            else:
                curv = None
        except Exception:
            curv = None
        
        try:
            if os.path.exists(sulc_path):
                sulc = nib.freesurfer.io.read_morph_data(sulc_path)
            else:
                sulc = None
        except Exception:
            sulc = None

        # convert to VTK array for point data
        if curv is not None:
            curvArray = vtk.util.numpy_support.numpy_to_vtk(curv.astype("f4"), deep=True)
            curvArray.SetName("curv")
            polyData.GetPointData().AddArray(curvArray)
        if sulc is not None:
            sulcArray = vtk.util.numpy_support.numpy_to_vtk(sulc.astype("f4"), deep=True)
            sulcArray.SetName("sulc")
            polyData.GetPointData().AddArray(sulcArray)

        modelNode.SetAndObservePolyData(polyData)
    
    except:
        import traceback
        traceback.print_exc()
        return False
    return True

def load_inputs(t1_path, ribbon_path, lh_pial_path, rh_pial_path, lh_envelope_path, rh_envelope_path, brain_envelope_path, photo_path, create_env_mode=False):
    """
    Load all input data: T1 volume, cortical surfaces, envelope models, and photograph.
    
    Parameters
    ----------
    t1_path : str
        Path to T1 MRI volume
    ribbon_path : str
        Path to FreeSurfer ribbon (gray matter mask) volume
    lh_pial_path : str
        Path to left hemisphere pial surface
    rh_pial_path : str
        Path to right hemisphere pial surface
    lh_envelope_path : str
        Path to left hemisphere envelope STL file
    rh_envelope_path : str
        Path to right hemisphere envelope STL file
    brain_envelope_path : str
        Path to whole brain envelope STL file
    photo_path : str
        Path to intraoperative photograph
    create_env_mode : bool, optional
        If True, only creates envelopes and exits. Defaults to False
    
    Returns
    -------
    tuple
        (t1Node, ribbonNode, lh_pialNode, rh_pialNode, lh_envelopeNode, 
         rh_envelopeNode, brain_envelopeNode, photoVolumeNode)
        Returns (None, None, None, None, lh_envelopeNode, rh_envelopeNode, 
         brain_envelopeNode, None) if create_env_mode is True
    
    Raises
    ------
    RuntimeError
        If required files fail to load
    """

    # Pial surfaces
    print("Loading L pial surface (FreeSurfer curve):", lh_pial_path)
    lh_pialNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", "lh_pial")
    load_success_lh = read_fs_surf(lh_pial_path, lh_pialNode, calculateNormals=True)
    if not load_success_lh:
        if create_env_mode:
            raise RuntimeError("Failed to load lh.pial model.")
    else:
        print("Cortical model loaded:", lh_pialNode.GetName())

    print("Loading R pial surface (FreeSurfer curve):", rh_pial_path)
    rh_pialNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", "rh_pial")
    load_success_rh = read_fs_surf(rh_pial_path, rh_pialNode, calculateNormals=True)
    if not load_success_rh:
        if create_env_mode:
            raise RuntimeError("Failed to load rh.pial model.")
    else:
        print("Cortical model loaded:", rh_pialNode.GetName())

    # Envelope surfaces
    if os.path.exists(lh_envelope_path) and os.path.exists(rh_envelope_path) and os.path.exists(brain_envelope_path):
        print("Loading existing envelope surfaces.")
        lh_envelopeNode = util.load_stl_surface(lh_envelope_path, "lh_envelope")
        rh_envelopeNode = util.load_stl_surface(rh_envelope_path, "rh_envelope")
        brain_envelopeNode = util.load_stl_surface(brain_envelope_path, "brain_envelope")
    else:
        print("Envelope surfaces not found. Creating from pial surfaces (this may take a minute)...")
        slicer.app.processEvents()
        # Create envelope models from pial surfaces
        try:
            lh_envelopeNode, rh_envelopeNode, brain_envelopeNode = util.create_envelopes(
                lh_pialNode, rh_pialNode, surf_dir=os.path.dirname(lh_envelope_path))
        except Exception as e:
            print("Error creating envelope surfaces:", e)
            if create_env_mode:
                print("In envelope creation mode, exiting Slicer.")
                slicer.app.processEvents()
                sys.exit(1)

    # If in create envelope mode, skip loading photo and return here
    if create_env_mode:
        return None, None, None, None, lh_envelopeNode, rh_envelopeNode, brain_envelopeNode, None

    # T1 volume
    print("Loading reference T1:", t1_path)
    t1Node = slicer.util.loadVolume(t1_path)
    if not t1Node:
        raise RuntimeError("Failed to load T1 volume.")
    print("T1 loaded:", t1Node.GetName())

    # Ribbon volume
    print("Loading ribbon volume (FreeSurfer GM mask):", ribbon_path)
    ribbonNode = slicer.util.loadVolume(ribbon_path)
    if not ribbonNode:
        raise RuntimeError("Failed to load ribbon volume.")
    print("Ribbon volume loaded:", ribbonNode.GetName())

    # Photograph
    print("Loading photo as volume (will be used as texture):", photo_path)
    # Slicer can load PNG/JPG as a scalar volume node - used as texture source
    photoVolumeNode = slicer.util.loadVolume(photo_path)
    if not photoVolumeNode:
        raise RuntimeError("Failed to load photo image as volume. Check image path.")
    print("Photo image loaded as volume node:", photoVolumeNode.GetName())

    # Show all surfaces except brain envelope
    for modelNode in (lh_pialNode, rh_pialNode, lh_envelopeNode, rh_envelopeNode, brain_envelopeNode):
        displayNode = modelNode.GetDisplayNode()
        if not displayNode:
            displayNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelDisplayNode")
            modelNode.SetAndObserveDisplayNodeID(displayNode.GetID())

    # Set initial display properties
    lh_pialDisplayNode = lh_pialNode.GetDisplayNode()
    rh_pialDisplayNode = rh_pialNode.GetDisplayNode()
    lh_envelopeDisplayNode = lh_envelopeNode.GetDisplayNode()
    rh_envelopeDisplayNode = rh_envelopeNode.GetDisplayNode()
    brain_envelopeDisplayNode = brain_envelopeNode.GetDisplayNode()
    lh_pialDisplayNode.SetColor(0.8, 0.8, 1.0)
    lh_pialDisplayNode.SetOpacity(0.8)
    lh_pialDisplayNode.SetVisibility(True)
    rh_pialDisplayNode.SetColor(0.8, 0.8, 1.0)
    rh_pialDisplayNode.SetOpacity(0.8)
    rh_pialDisplayNode.SetVisibility(True)
    lh_envelopeDisplayNode.SetColor(0.0, 0.0, 1.0)
    lh_envelopeDisplayNode.SetOpacity(0.2)
    lh_envelopeDisplayNode.SetVisibility(False)
    rh_envelopeDisplayNode.SetColor(0.0, 0.0, 1.0)
    rh_envelopeDisplayNode.SetOpacity(0.2)
    rh_envelopeDisplayNode.SetVisibility(False)
    brain_envelopeDisplayNode.SetColor(0.0, 0.0, 1.0)
    brain_envelopeDisplayNode.SetOpacity(0.8)
    brain_envelopeDisplayNode.SetVisibility(True)

    return t1Node, ribbonNode, lh_pialNode, rh_pialNode, lh_envelopeNode, rh_envelopeNode, brain_envelopeNode, photoVolumeNode

def create_textured_plane(photoVolumeNode, planeName="PhotoPlane", width=120.0, height=120.0, opacity=0.6):
    """
    Create a plane model which will receive the photograph as a texture.
    
    Parameters
    ----------
    photoVolumeNode : vtkMRMLScalarVolumeNode
        Volume node containing the photograph data
    planeName : str, optional
        Name for the created plane model. Defaults to "PhotoPlane"
    width : float, optional
        Width of the plane in mm. Defaults to 120.0
    height : float, optional
        Height of the plane in mm. Defaults to 120.0
    opacity : float, optional
        Opacity of the textured plane (0.0 to 1.0). Defaults to 0.6
    
    Returns
    -------
    tuple
        (planeNode, flip) - The model node containing the plane and the VTK flip filter
    """
    # Create vtkPlaneSource
    plane = vtk.vtkPlaneSource()
    plane.SetOrigin(-width/2.0, -height/2.0, 0.0)
    plane.SetPoint1(width/2.0, -height/2.0, 0.0)
    plane.SetPoint2(-width/2.0, height/2.0, 0.0)
    plane.SetXResolution(1)
    plane.SetYResolution(1)
    plane.Update()

    # Add as model node
    planeNode = slicer.modules.models.logic().AddModel(plane.GetOutputPort())
    planeNode.SetName(planeName)

    # Make semi-transparent
    displayNode = planeNode.GetModelDisplayNode()
    displayNode.SetOpacity(opacity)
    displayNode.SetBackfaceCulling(False)
    displayNode.SetSelectable(True)
    displayNode.SetVisibility(False)

    # Texture assignment:
    # flip image vertically (needed for correct orientation)
    flip = vtk.vtkImageFlip()
    flip.SetFilteredAxis(1)  # flip vertical axis
    # Connect the photo volume image data
    flip.SetInputConnection(photoVolumeNode.GetImageDataConnection())
    flip.Update()

    # Connect the flipped image pipeline to model display as texture
    displayNode.SetTextureImageDataConnection(flip.GetOutputPort())

    # Naming note: the texture is referenced via the display node's connection (not saved as separate node)
    print("Created textured plane:", planeNode.GetName(), " (texture connected)")

    return planeNode, flip

def setup_interactive_photo_plane(planeNode, envelopeNode, offset_mm=5.0):
    """
    Set up interactive transform controls for the photo plane.
    
    Places plane on envelope surface with offset, attaches transform,
    enables immediate interactive handles, and constrains dragging to 
    envelope surface with in-plane rotation and uniform scale.
    
    Parameters
    ----------
    planeNode : vtkMRMLModelNode
        The plane model node to make interactive
    envelopeNode : vtkMRMLModelNode
        The envelope surface model node that constrains plane position
    offset_mm : float, optional
        Offset distance from envelope surface in mm. Defaults to 5.0
    
    Returns
    -------
    tuple
        (transformObserver, transformNode) - Observer managing constraints and the transform node
    """
    # Create linear transform and attach
    transformNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLinearTransformNode", "PhotoTransform")
    planeNode.SetAndObserveTransformNodeID(transformNode.GetID())

    # Attach draggable constraint observer
    transformObserver = util.PhotoTransformObserver(transformNode, planeNode, envelopeNode, offset_mm=offset_mm)
    transformObserver.start()

    # Setup interactive handles
    util.setup_interactive_transform(transformNode, visibility=False)

    return transformObserver, transformNode

def main(t1_path, ribbon_path, lh_pial_path, rh_pial_path, lh_envelope_path, rh_envelope_path, brain_envelope_path, photo_path, mask_path, output_dir, create_envelope_mode=False, photo_set_manifest=None):
    """
    Main execution function for photo-to-MRI registration in 3D Slicer.
    
    Sets up the Slicer scene with all necessary data, creates interactive
    transform controls for manual photo alignment, and provides keyboard
    shortcuts for common operations.
    
    Parameters
    ----------
    t1_path : str
        Path to T1 MRI volume
    ribbon_path : str
        Path to FreeSurfer ribbon (gray matter mask) volume
    lh_pial_path : str
        Path to left hemisphere pial surface
    rh_pial_path : str
        Path to right hemisphere pial surface
    lh_envelope_path : str
        Path to left hemisphere envelope STL file
    rh_envelope_path : str
        Path to right hemisphere envelope STL file
    brain_envelope_path : str
        Path to whole brain envelope STL file
    photo_path : str
        Path to intraoperative photograph
    mask_path : str
        Path to .npz file containing photo masks
    output_dir : str
        Directory for saving output files
    create_envelope_mode : bool, optional
        If True, only creates envelopes and exits. Defaults to False
    """

    manifest = load_photo_set_manifest(photo_set_manifest) if photo_set_manifest else None
    if manifest is not None:
        photo_path = photo_path or next(
            (
                photo["path"] if isinstance(photo, dict) and "path" in photo else None
                for photo in manifest.get("photos", [])
                if str(photo.get("role", "")).lower() == "reference"
            ),
            photo_path,
        )
        print(f"[PhotoSet] Loaded manifest for patient '{manifest.get('patient_id')}'. Reference photo: {photo_path}")

    # On startup, set layout to 3D view only and open Models module and Python console
    if not create_envelope_mode:
        slicer.app.layoutManager().setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutOneUp3DView)
        slicer.util.selectModule('Models')
        # TODO: Open python interactor

    # Load all inputs
    try:
        t1Node, ribbonNode, lh_pialNode, rh_pialNode, lh_envelopeNode, rh_envelopeNode, brain_envelopeNode, photoVolumeNode = \
            load_inputs(t1_path, ribbon_path, lh_pial_path, rh_pial_path, lh_envelope_path, rh_envelope_path, brain_envelope_path, photo_path, create_envelope_mode)
    except Exception as e:
        print("Error loading inputs:", e)
        if create_envelope_mode:
            slicer.app.processEvents()
            sys.exit(1)
        return

    Nodes = {
        "t1Node": t1Node,
        "ribbonNode": ribbonNode,
        "lh_pialNode": lh_pialNode,
        "rh_pialNode": rh_pialNode,
        "lh_envelopeNode": lh_envelopeNode,
        "rh_envelopeNode": rh_envelopeNode,
        "brain_envelopeNode": brain_envelopeNode,
        "photoVolumeNode": photoVolumeNode
    }

    # If in envelope creation mode, stop here (after loading and creating envelopes)
    if create_envelope_mode:
        print("Envelope creation mode: envelopes created/loaded. Stopping")
        slicer.app.processEvents()
        sys.exit(0)

    # Load photo masks into photo volume
    if mask_path and os.path.exists(mask_path):
        util.load_photo_masks(mask_path, photoVolumeNode)
    else:
        print("No mask file provided or found; proceeding without masking.")

    # Create textured plane sized roughly to cover the envelope — adjust width/height if needed
    # Make width/height relative to image aspect ratio
    imageData = photoVolumeNode.GetImageData()
    dims = imageData.GetDimensions()
    aspect = dims[0] / dims[1] if dims[1] != 0 else 1.0
    base_height = 100.0
    base_width = base_height * aspect
    plane_dims = (base_width, base_height)
    planeNode, _flip = create_textured_plane(
        photoVolumeNode, planeName="ResectionPhotoPlane", width=base_width, height=base_height, opacity=0.6)
    Nodes["planeNode"] = planeNode

    # Attach transform so user can manipulate
    transformObserver, transformNode = setup_interactive_photo_plane(planeNode, brain_envelopeNode, offset_mm=1.0)
    Nodes["transformNode"] = transformNode

    # Setup projection onto brain envelope
    # util.setup_projection_onto_model(Nodes, plane_dims, cam_dist_mm=150.0, proj_slab_thickness_mm=15.0)
    MainProjection = util.Projection(
        brain_envelopeNode, photoVolumeNode, transformNode, plane_dims,
        cam_dist_mm=150.0, proj_slab_thickness_mm=15.0,
        rgb=True, visualize_camera=False
    )

    # Setup slider to control projection camera distance
    cam_dist_slider, observer_toggle, depth_slider, include_wm_checkbox, _ = util.setup_ui_widgets(
        MainProjection, transformObserver, transformNode,
        min_mm=1.0, max_mm=300.0, initial_mm=150.0
    )

    # Setup keypress interactor
    util.setup_interactor(Nodes, plane_dims, mask_path, MainProjection, transformObserver, output_dir)

    # Provide variables in global namespace to allow interactive commands after script finishes
    globals()['Nodes'] = Nodes
    globals()['planeNode'] = planeNode
    globals()['transformNode'] = transformNode
    globals()['t1Node'] = t1Node
    globals()['ribbonNode'] = ribbonNode
    globals()['lh_pialNode'] = lh_pialNode
    globals()['rh_pialNode'] = rh_pialNode
    globals()['lh_envelopeNode'] = lh_envelopeNode
    globals()['rh_envelopeNode'] = rh_envelopeNode
    globals()['brain_envelopeNode'] = brain_envelopeNode
    globals()['photoVolumeNode'] = photoVolumeNode
    globals()['photoAspect'] = aspect
    globals()['photoDims'] = dims
    globals()['planeDims'] = (base_width, base_height)
    globals()['MainProjection'] = MainProjection
    globals()['transformObserver'] = transformObserver
    globals()['cam_dist_slider'] = cam_dist_slider
    globals()['observer_toggle'] = observer_toggle
    globals()['depth_slider'] = depth_slider
    globals()['include_wm_checkbox'] = include_wm_checkbox
    globals()['output_dir'] = output_dir
    globals()['mask_path'] = mask_path

    print("\n==============================================================================\n")
    print("Setup complete. You can now interactively align the photo projection.")
    print("Also check the right panel for adjustable projection parameters.")
    print("Press <enter> to drop in projection + interactor in current view.")
    print("Press <space> to align the 3D camera with the projection plane.")
    print("Press '1' to toggle the 3D model view.")
    print("Press '2' to toggle the 2D photo view.")
    print("Press 'a' for optional (final) auto-align (WARNING: Not stable yet)")
    print("Press 'v' to create volumetric resection mask.")
    print("Press 's' to save the current scene and resection mask.")
    print("Press 'q' to quit Slicer when done.")
    print("Press 'x' to mark as atlas-based and quit Slicer.")
    print("Press 'Escape' to quit Slicer and break outside loop.")
    print("\n==============================================================================\n")

if __name__ == "__main__":

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Slicer automation script for photo-to-MRI alignment.")
    parser.add_argument("--t1_path", type=str, help="Path to T1 volume")
    parser.add_argument("--ribbon_path", type=str, help="Path to 'ribbon' volume (=FreeSurfer's GM mask)")
    parser.add_argument("--lh_pial_path", type=str, help="Path to left hemisphere pial surface")
    parser.add_argument("--rh_pial_path", type=str, help="Path to right hemisphere pial surface")
    parser.add_argument("--lh_envelope_path", type=str, help="Path to left envelope model")
    parser.add_argument("--rh_envelope_path", type=str, help="Path to right envelope model")
    parser.add_argument("--brain_envelope_path", type=str, help="Path to whole brain envelope model")
    parser.add_argument("--photo_path", type=str, help="Path to intraoperative photograph")
    parser.add_argument("--mask_path", type=str, help="Path to photo masks .npz file")
    parser.add_argument("--output_dir", type=str, help="Path to output directory")
    parser.add_argument("--photo_set_manifest", type=str, help="Optional patient photo-set manifest; reference photo is used for manual cortex alignment")
    parser.add_argument("--create_envelope_mode", action="store_true", help="Flag to enable envelope creation loop")
    args = parser.parse_args()

    # Assign to variables
    t1_path = args.t1_path
    ribbon_path = args.ribbon_path
    lh_pial_path = args.lh_pial_path
    rh_pial_path = args.rh_pial_path
    lh_envelope_path = args.lh_envelope_path
    rh_envelope_path = args.rh_envelope_path
    brain_envelope_path = args.brain_envelope_path
    photo_path = args.photo_path
    mask_path = args.mask_path
    output_dir = args.output_dir
    create_envelope_mode = args.create_envelope_mode

    # Run main function
    main(t1_path, ribbon_path, lh_pial_path, rh_pial_path, lh_envelope_path, rh_envelope_path, brain_envelope_path, photo_path, mask_path, output_dir, create_envelope_mode)