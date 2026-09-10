"""
Geometry and mathematical utilities for 3D Slicer including matrix conversions,
rotation utilities, and VTK polydata operations.
"""

import os
import numpy as np
import vtk  # type: ignore
from vtkmodules.util import numpy_support  # type: ignore
import slicer  # type: ignore


def vtkMatrixToNumpy(vtkMat):
    """Convert VTK 4x4 matrix to numpy array."""
    a = np.zeros((4, 4))
    for r in range(4):
        for c in range(4):
            a[r, c] = vtkMat.GetElement(r, c)
    return a


def numpyToVtkMatrix(npMat):
    """Convert numpy array to VTK 4x4 matrix."""
    vtkMat = vtk.vtkMatrix4x4()
    for r in range(4):
        for c in range(4):
            vtkMat.SetElement(r, c, float(npMat[r, c]))
    return vtkMat


def extractRotationScale(mat3):
    """Extract rotation matrix and scale from a 3x3 matrix using SVD."""
    U, S, Vt = np.linalg.svd(mat3)
    R = U.dot(Vt)
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U.dot(Vt)
    scale = np.mean(S)
    return R, scale


def rotationFromVectors(v_from, v_to):
    """Compute rotation matrix that rotates v_from to v_to."""
    v_from = v_from / np.linalg.norm(v_from)
    v_to = v_to / np.linalg.norm(v_to)
    cross = np.cross(v_from, v_to)
    dot = np.dot(v_from, v_to)
    if np.allclose(cross, 0) and dot > 0.9999:
        return np.eye(3)
    if np.allclose(cross, 0) and dot < -0.9999:
        axis = np.array([1, 0, 0])
        if abs(v_from[0]) > 0.9:
            axis = np.array([0, 1, 0])
        axis = axis - np.dot(axis, v_from) * v_from
        axis = axis / np.linalg.norm(axis)
        K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        R = np.eye(3) + 2 * K.dot(K)
        return R
    K = np.array([[0, -cross[2], cross[1]], [cross[2], 0, -cross[0]], [-cross[1], cross[0], 0]])
    s = np.linalg.norm(cross)
    c = dot
    R = np.eye(3) + K + K.dot(K) * ((1 - c) / (s * s))
    return R


def get_poly_normals(polyData):
    """Ensure polydata has point normals and return them."""
    # Get normals
    normals = polyData.GetPointData().GetNormals()
    # Compute if missing
    if normals is None:
        normalsFilter = vtk.vtkPolyDataNormals()
        normalsFilter.SetInputData(polyData)
        normalsFilter.ComputePointNormalsOn()
        normalsFilter.AutoOrientNormalsOn()
        normalsFilter.SplittingOff()
        normalsFilter.ConsistencyOn()
        normalsFilter.Update()
        polyData = normalsFilter.GetOutput()
        normals = polyData.GetPointData().GetNormals()
    return normals


def subdivide_model(modelNode, iterations=1):
    """Subdivide a model node's polydata using loop subdivision."""
    poly = modelNode.GetPolyData()
    subdiv = vtk.vtkLoopSubdivisionFilter()
    subdiv.SetNumberOfSubdivisions(iterations)
    subdiv.SetInputData(poly)
    subdiv.Update()
    modelNode.SetAndObservePolyData(subdiv.GetOutput())


def clone_model_node(sourceNode, new_name, color=None, opacity=1.0, visibility=False):
    """Create a new model node with a deep copy of `sourceNode`'s polydata.

    Used so that per-photo projected envelopes each own independent polydata:
    projecting/subdividing one photo's copy can never mutate another photo's
    copy or the pristine source/template node.
    """
    sourcePoly = sourceNode.GetPolyData()
    if sourcePoly is None:
        raise RuntimeError(f"Source model '{sourceNode.GetName()}' has no polydata to clone.")

    polyCopy = vtk.vtkPolyData()
    polyCopy.DeepCopy(sourcePoly)

    newNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", new_name)
    newNode.SetAndObservePolyData(polyCopy)

    displayNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelDisplayNode")
    if color is not None:
        displayNode.SetColor(*color)
    displayNode.SetOpacity(opacity)
    displayNode.SetVisibility(visibility)
    newNode.SetAndObserveDisplayNodeID(displayNode.GetID())

    return newNode


def sample_scalar_along_normals(envelopeNode, pialNode, scalarName="curv", rayLength=25.0, attachToEnvelope=True):
    """
    For each point on the envelope surface, cast a ray along the negative normal
    and sample the scalar value from the pial surface beneath it.
    
    Parameters
    ----------
    envelopeNode : vtkMRMLModelNode
        The outer envelope surface node.
    pialNode : vtkMRMLModelNode
        The pial surface node containing the scalar array.
    scalarName : str
        Name of the scalar array on the pial surface to sample.
    rayLength : float
        Maximum length along the negative normal to search for intersection (mm).
    attachToEnvelope : bool
        If True, attach the sampled scalar as a new array on the envelope surface.
    
    Returns
    -------
    sampledScalars : numpy.ndarray
        Array of sampled scalar values for each envelope point (NaN if no intersection).
    """

    print(f"Sampling scalar '{scalarName}' from pial surface onto envelope surface...", flush=True)
    slicer.app.processEvents()

    # Check scalar
    pialScalars = pialNode.GetPolyData().GetPointData().GetArray(scalarName)
    if not pialScalars:
        raise RuntimeError(f"Scalar '{scalarName}' not found on pial surface")
    
    # Get polydata
    pialPoly = pialNode.GetPolyData()
    envPoly = envelopeNode.GetPolyData()
    
    # Compute envelope normals if missing
    envNormals = envPoly.GetPointData().GetNormals()
    if not envNormals:
        normalGenerator = vtk.vtkPolyDataNormals()
        normalGenerator.SetInputData(envPoly)
        normalGenerator.ComputePointNormalsOn()
        normalGenerator.ComputeCellNormalsOff()
        normalGenerator.Update()
        envPoly.DeepCopy(normalGenerator.GetOutput())
        envNormals = envPoly.GetPointData().GetNormals()
    
    # Build OBB tree for intersection
    obbTree = vtk.vtkOBBTree()
    obbTree.SetDataSet(pialPoly)
    obbTree.BuildLocator()
    
    # Prepare array
    nPoints = envPoly.GetNumberOfPoints()
    sampledScalars = np.zeros(nPoints, dtype=float)
    
    # Point locator for nearest vertex on pial
    locator = vtk.vtkPointLocator()
    locator.SetDataSet(pialPoly)
    locator.BuildLocator()
    
    intersectionPoints = vtk.vtkPoints()
    
    # Loop over envelope points
    for i in range(nPoints):
        pt = np.array(envPoly.GetPoint(i))
        normal = np.array(envNormals.GetTuple(i))
        rayStart = pt
        rayEnd = pt + normal * rayLength
        
        intersectionPoints.Reset()
        code = obbTree.IntersectWithLine(rayStart, rayEnd, intersectionPoints, None)
        
        if intersectionPoints.GetNumberOfPoints() > 0:
            intersectPt = np.array(intersectionPoints.GetPoint(0))
            closestId = locator.FindClosestPoint(intersectPt)
            sampledScalars[i] = pialScalars.GetTuple1(closestId)
        else:
            sampledScalars[i] = np.nan
    
    if attachToEnvelope:
        vtkArray = numpy_support.numpy_to_vtk(sampledScalars.astype("f4"), deep=True)
        vtkArray.SetName(scalarName)
        if envPoly.GetPointData().HasArray(scalarName):
            envPoly.GetPointData().RemoveArray(scalarName)
        envPoly.GetPointData().AddArray(vtkArray)
        envPoly.GetPointData().SetActiveScalars(scalarName)
        envPoly.Modified()
        if envelopeNode.GetDisplayNode():
            envelopeNode.GetDisplayNode().SetActiveScalarName("curv")
            envelopeNode.GetDisplayNode().Modified()

    print(f"Completed sampling scalar '{scalarName}'. {np.nansum(np.isnan(sampledScalars))}/{nPoints} points had no intersection.", flush=True)

    return sampledScalars


def ras_to_lps_polydata(polyData):
    """Convert polydata from RAS to LPS coordinate system."""
    transform = vtk.vtkTransform()
    transform.Scale(-1, -1, 1)
    transformFilter = vtk.vtkTransformPolyDataFilter()
    transformFilter.SetInputData(polyData)
    transformFilter.SetTransform(transform)
    transformFilter.Update()
    return transformFilter.GetOutput()


def load_photo_masks(mask_path, photoVolumeNode,
                     scalar_value_resection=(1, 128, 1),
                     scalar_value_outside=(0, 0, 0)):
    """
    Apply user-drawn resection and outside masks to a photo texture volume.

    Replaces pixel values in masked areas with specified RGB or scalar values.
    This is used to highlight or hide specific regions in the photograph.
    
    Parameters
    ----------
    mask_path : str
        Path to .npz file containing 'resection_mask' and 'outside_mask' boolean arrays
    photoVolumeNode : vtkMRMLScalarVolumeNode
        Slicer volume node containing the photograph (RGB or grayscale)
    scalar_value_resection : tuple or scalar, optional
        RGB values or scalar to use for resection mask pixels. Defaults to (1, 128, 1)
    scalar_value_outside : tuple or scalar, optional
        RGB values or scalar to use for outside mask pixels. Defaults to (0, 0, 0)
    """

    if not os.path.exists(mask_path):
        print(f"[Mask] File not found: {mask_path}")
        return

    # Load masks
    masks = np.load(mask_path)
    resection_mask = masks.get("resection_mask")
    outside_mask = masks.get("outside_mask")
    if resection_mask is None or outside_mask is None:
        print("[Mask] Missing mask arrays in file.")
        return

    imageData = photoVolumeNode.GetImageData()
    if imageData is None:
        print("[Mask] No image data in photoVolumeNode.")
        return

    dims = imageData.GetDimensions()  # (x, y, z)
    comps = imageData.GetNumberOfScalarComponents()

    vtk_array = imageData.GetPointData().GetScalars()
    np_array = numpy_support.vtk_to_numpy(vtk_array)

    # Reshape properly: (z, y, x, comps)
    arr = np_array.reshape((dims[2], dims[1], dims[0], comps))
    if dims[2] > 1:
        print(f"[Mask] Multi-slice photo (z={dims[2]}), applying to first slice only.")
    img = arr[0]  # (y, x, comps)

    if resection_mask.shape != (dims[1], dims[0]):
        raise ValueError(f"[Mask] Mask shape {resection_mask.shape} != image shape {(dims[1], dims[0])}")

    # Convert scalar → per-component tuple if needed
    def to_vec(val):
        val = np.atleast_1d(val)
        if val.size == 1:
            return np.repeat(val, comps)
        if comps == 4 and val.size == 3:
            # Keep alpha unchanged
            return np.concatenate([val, [img[..., 3].mean()]])
        return np.resize(val, (comps,))

    resec_vals = to_vec(scalar_value_resection)
    outside_vals = to_vec(scalar_value_outside)

    # Apply masks
    modified = img.copy()
    for c in range(comps):
        modified[..., c][resection_mask] = resec_vals[c]
        modified[..., c][outside_mask] = outside_vals[c]

    arr[0] = modified

    # Convert back to VTK
    new_vtk = numpy_support.numpy_to_vtk(num_array=arr.ravel(order='C'), deep=True)
    new_vtk.SetNumberOfComponents(comps)
    imageData.GetPointData().SetScalars(new_vtk)
    imageData.Modified()
    photoVolumeNode.Modified()

    print(f"[Mask] Applied masks to photo volume ({comps} components).")
