"""
I/O operations for 3D Slicer including scene management, STL file handling, 
and envelope creation.
"""

import os
import subprocess
import vtk  # type: ignore
import slicer  # type: ignore


def ras_to_lps_polydata(polyData):
    """Convert polydata from RAS to LPS coordinate system."""
    transform = vtk.vtkTransform()
    transform.Scale(-1, -1, 1)
    transformFilter = vtk.vtkTransformPolyDataFilter()
    transformFilter.SetInputData(polyData)
    transformFilter.SetTransform(transform)
    transformFilter.Update()
    return transformFilter.GetOutput()


def write_stl_surface(modelNode, stl_path):
    """Write polydata to STL file in LPS coordinate system."""
    polyData = modelNode.GetPolyData()
    transform = vtk.vtkTransform()
    transform.Scale(-1, -1, 1)
    transformFilter = vtk.vtkTransformPolyDataFilter()
    transformFilter.SetInputData(polyData)
    transformFilter.SetTransform(transform)
    transformFilter.Update()
    stl_writer = vtk.vtkSTLWriter()
    stl_writer.SetFileName(stl_path)
    stl_writer.SetInputData(transformFilter.GetOutput())
    stl_writer.Write()
    print(f"Exported model to STL (LPS): {stl_path}")


def load_stl_surface(stl_path, modelNodeName):
    """Load an STL surface into a model node."""
    # Check for file
    if not os.path.exists(stl_path):
        raise FileNotFoundError(f"STL file not found: {stl_path}")
    # Add node
    modelNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", modelNodeName)
    # Read STL
    stl_reader = vtk.vtkSTLReader()
    stl_reader.SetFileName(stl_path)
    stl_reader.Update()
    polyData = stl_reader.GetOutput()
    # Set data to node in LPS
    modelNode.SetAndObservePolyData(ras_to_lps_polydata(polyData))
    # Check and return
    if not modelNode:
        raise RuntimeError(f"Failed to load STL model from {stl_path}. Check model path.")
    print(f"STL model loaded: {modelNode.GetName()}")
    return modelNode


def load_photo_volume(path, node_name=None):
    """Load a photograph file as a Slicer scalar volume node (used as a projection texture source).

    Used for the reference photo as well as auxiliary/post-resection photos once
    they are expressed in the reference-photo pixel grid.
    """
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"Photo image not found: {path}")
    volumeNode = slicer.util.loadVolume(path)
    if not volumeNode:
        raise RuntimeError(f"Failed to load photo image as volume: {path}")
    if node_name:
        volumeNode.SetName(node_name)
    return volumeNode


def create_envelopes(lh_pialNode, rh_pialNode, surf_dir):
    """Create envelope models from pial surface models and save as STL. Uses external MATLAB script."""

    # First, write pial model to STL (in LPS)
    write_stl_surface(lh_pialNode, os.path.join(surf_dir, "lh_pial.stl"))
    write_stl_surface(rh_pialNode, os.path.join(surf_dir, "rh_pial.stl"))

    # Now run the MATLAB script to create envelope from STL
    # Find MATLAB directory relative to parent package
    matlab_scripts_root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "MATLAB")
    result = subprocess.run(
        [
            "matlab", "-batch",
            f"restoredefaultpath; addpath('{matlab_scripts_root}', '{os.path.join(matlab_scripts_root, 'Functions')}'); create_envelopes('{surf_dir}')"
        ],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print("Error running create_envelope.m", result.stderr)
        raise RuntimeError("Envelope creation failed.")
    print(f"Created STL envelopes in: {surf_dir}")

    # Load the created envelope STLs into model nodes
    lh_envelopeNode = load_stl_surface(os.path.join(surf_dir, "lh_envelope.stl"), "lh_envelope")
    rh_envelopeNode = load_stl_surface(os.path.join(surf_dir, "rh_envelope.stl"), "rh_envelope")
    brain_envelopeNode = load_stl_surface(os.path.join(surf_dir, "brain_envelope.stl"), "brain_envelope")

    return lh_envelopeNode, rh_envelopeNode, brain_envelopeNode


def save_scene_to_directory(directory_path, Nodes, photo_states=None):
    """Save the current Slicer scene to a specified directory as a Slicer Data Bundle.

    `photo_states`, if given, is a dict of photo_id -> PhotoProjectionState. The
    scene keeps every materialised photo plane/projected envelope so the saved
    scene reloads with all photo projections available for visibility toggling;
    only genuinely non-essential temporary nodes (unused hemisphere envelopes,
    interactive transform handles) are removed or hidden.
    """

    # Import here to avoid circular dependency
    from .interaction import setup_interactive_transform, center_camera_on_projection

    print(f"Saving scene to directory: {directory_path}")

    # Remove non-essential elements from scene before saving and setup display for saving.
    if Nodes.get('lh_envelopeNode') is not None:
        slicer.mrmlScene.RemoveNode(Nodes['lh_envelopeNode'])
    if Nodes.get('rh_envelopeNode') is not None:
        slicer.mrmlScene.RemoveNode(Nodes['rh_envelopeNode'])
    setup_interactive_transform(Nodes['transformNode'], visibility=False, limit_to_surf_aligned=False)
    Nodes['lh_pialNode'].GetDisplayNode().SetOpacity(0.7)
    Nodes['rh_pialNode'].GetDisplayNode().SetOpacity(0.7)
    if Nodes.get('brain_envelopeNode') is not None and Nodes['brain_envelopeNode'].GetDisplayNode() is not None:
        Nodes['brain_envelopeNode'].GetDisplayNode().SetOpacity(0.6)
    center_camera_on_projection(Nodes)

    if photo_states:
        print(f"Scene includes {len(photo_states)} photo projection(s): {', '.join(sorted(photo_states.keys()))}")

    # Make directory if needed
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    # Process events to ensure scene is up to date
    slicer.app.processEvents()

    # Save scene
    success = slicer.app.applicationLogic().SaveSceneToSlicerDataBundleDirectory(directory_path)
    if success:
        print(f"Scene saved.")
    else:
        print(f"Failed to save scene!")

    # Add screenshot of current 3D view
    image = slicer.app.layoutManager().threeDWidget(0).threeDView().grab()
    screenshot_path = os.path.join(directory_path, "scene_screenshot.png")
    image.save(screenshot_path)
    print(f"Saved scene screenshot to: {screenshot_path}")

    return success
