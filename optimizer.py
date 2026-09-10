import math
import numpy as np
import vtk  # type: ignore
from vtkmodules.util import numpy_support  # type: ignore
import slicer  # type: ignore
import util
from scipy import ndimage as ndi
from scipy import optimize

try:
    import cv2
except ImportError:
    slicer.util.pip_install("opencv-python")
    import cv2
import pprint

import os
import time
try:
    import matplotlib.pyplot as plt
except Exception:
    slicer.util.pip_install("matplotlib")
    import matplotlib.pyplot as plt

def plot_debug_images(proj_img, curv_img, mask, sdf_p, sdf_q, nmi_map,
                      debug_dir="L:/her_knf_golf/Wetenschap/newtransport/Sjors/data/tmp"):
    """
    Plot debug images for visual inspection of alignment optimization.
    
    Creates a diagnostic figure showing:
    1. Projected photo (p)
    2. Curvature map (q)  
    3. Overlay (R=p, G=q)
    4. SDF of p
    5. SDF of q
    6. NMI map
    
    Parameters
    ----------
    proj_img : np.ndarray
        Projected photo soft mask image
    curv_img : np.ndarray
        Curvature soft mask image
    mask : np.ndarray
        Boolean mask indicating valid regions
    sdf_p : np.ndarray
        Signed distance field for projected photo
    sdf_q : np.ndarray
        Signed distance field for curvature
    nmi_map : np.ndarray
        Normalized mutual information map
    debug_dir : str, optional
        Directory to save debug images
    """
    # Get images
    photo_proc = proj_img
    curv_proc = curv_img

    # Timestamp
    timestamp = int(time.time() * 1000)

    plt.figure()  # wide layout for 3 columns x 2 rows

    # Ensure full-image sdf arrays (expand if masked/1D)
    try:
        sdf_p_img = np.asarray(-sdf_p, dtype=np.float32)
        sdf_q_img = np.asarray(sdf_q, dtype=np.float32)

        if sdf_p_img.shape != photo_proc.shape:
            temp_p = np.zeros_like(photo_proc, dtype=np.float32)
            temp_q = np.zeros_like(curv_proc, dtype=np.float32)
            if mask is not None:
                temp_p[mask] = sdf_p_img.ravel()
                temp_q[mask] = sdf_q_img.ravel()
            else:
                try:
                    temp_p = sdf_p_img.reshape(photo_proc.shape)
                    temp_q = sdf_q_img.reshape(curv_proc.shape)
                except Exception:
                    pass
            sdf_p_img = temp_p
            sdf_q_img = temp_q
    except Exception:
        sdf_p_img = np.zeros_like(photo_proc, dtype=np.float32)
        sdf_q_img = np.zeros_like(curv_proc, dtype=np.float32)

    # compute difference
    sdf_diff_abs = np.abs(sdf_p_img - sdf_q_img)
    maxdiff = float(np.max(sdf_diff_abs))
    maxabs = max(float(np.max(np.abs(sdf_p_img))), float(np.max(np.abs(sdf_q_img))), 1e-9)

    # Plot 1: photo_proc (projected photo soft mask)
    ax1 = plt.subplot(2, 3, 1)
    ax1.imshow(photo_proc, cmap='gray', interpolation='nearest')
    ax1.set_title("1: Projected photo (p)")
    ax1.axis('off')

    # Plot 2: curv_proc (curvature soft mask)
    ax2 = plt.subplot(2, 3, 2)
    ax2.imshow(curv_proc, cmap='gray', interpolation='nearest')
    ax2.set_title("2: Curvature (q)")
    ax2.axis('off')

    # Plot 3: overlay photo_proc (red) + curv_proc (green)
    ax3 = plt.subplot(2, 3, 3)
    p_float = np.clip(photo_proc.astype(np.float32), 0.0, 1.0)
    q_float = np.clip(curv_proc.astype(np.float32), 0.0, 1.0)
    blue = np.minimum(p_float, q_float) * 0.5
    overlay_rgb = np.dstack([p_float, q_float, blue])
    ax3.imshow(overlay_rgb, interpolation='nearest')
    ax3.set_title("3: Overlay (R=p, G=q)")
    ax3.axis('off')

    # Plot 4: sdf_p
    ax4 = plt.subplot(2, 3, 4)
    ax4.imshow(sdf_p_img, cmap='seismic', vmin=-maxabs, vmax=maxabs, interpolation='nearest')
    ax4.set_title("4: SDF p")
    ax4.axis('off')

    # Plot 5: sdf_q
    ax5 = plt.subplot(2, 3, 5)
    ax5.imshow(sdf_q_img, cmap='seismic', vmin=-maxabs, vmax=maxabs, interpolation='nearest')
    ax5.set_title("5: SDF q")
    ax5.axis('off')

    # Plot 6: NMI map
    ax6 = plt.subplot(2, 3, 6)
    nmi_map[~mask] = (np.max(nmi_map) + np.min(nmi_map)) / 2.0
    ax6.imshow(nmi_map, cmap='seismic', vmin=np.min(nmi_map), vmax=np.max(nmi_map), interpolation='nearest')
    ax6.set_title("6: NMI map")
    ax6.axis('off')

    plt.tight_layout()

    file_path = os.path.join(debug_dir, f"autoAlign_debug_img_{timestamp}.png")
    plt.savefig(file_path, bbox_inches='tight')
    plt.close()

def preprocess_photo(
    img,
    denoise_strength=5,
    clip_limit=2.0,
    tile_grid=(8, 8),
):
    """
    Enhance sulci and large vessels in cortical intraoperative photographs.
    
    Ignores masked regions (value == 0.0).
    
    Parameters
    ----------
    img : np.ndarray
        Input photograph image (grayscale or RGB converted to grayscale)
    denoise_strength : int, optional
        Bilateral filter diameter for denoising. Use 0 to disable. Defaults to 5
    clip_limit : float, optional
        CLAHE clip limit for local contrast enhancement. Defaults to 2.0
    tile_grid : tuple, optional
        CLAHE tile grid size. Defaults to (8, 8)
    
    Returns
    -------
    tuple
        (enhanced_smoothed, valid_mask) - Enhanced image and boolean mask of valid regions
    """

    # ---- Step 0. Extract mask ----
    img = np.asarray(img, dtype=np.float32)
    valid_mask = img > 1e-6  # mask out resection (== 0.0)
    if not np.any(valid_mask):
        raise ValueError("Photo mask covers entire image.")

    # Replace invalids temporarily with local mean for filtering
    mean_val = np.mean(img[valid_mask])
    img[~valid_mask] = mean_val

    # ---- Step 1. Normalize input ----
    if img.max() > 1.5:
        img = (img - img.min()) / (img.max() - img.min() + 1e-6)
    else:
        img = np.clip(img, 0, 1)

    # ---- Step 2. Denoising (optional) ----
    if denoise_strength > 0:
        img_denoised = cv2.bilateralFilter(
            (img * 255).astype(np.uint8),
            d=denoise_strength,
            sigmaColor=50,
            sigmaSpace=50,
        )
        img = img_denoised.astype(np.float32) / 255.0

    # ---- Step 3. Local contrast normalization (CLAHE) ----
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    img_clahe = clahe.apply((img * 255).astype(np.uint8)).astype(np.float32) / 255.0

    # ---- Step 4. Ridge / sulcus enhancement ----
    blur = cv2.GaussianBlur(img_clahe, (3, 3), 0)
    lap = cv2.Laplacian(blur, ddepth=-1, ksize=3)
    enhanced = np.clip(img_clahe - 0.5 * lap, 0, 1)

    # ---- Step 5. Morphological cleanup ----
    morph = cv2.morphologyEx(
        (enhanced * 255).astype(np.uint8),
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    )
    enhanced = morph.astype(np.float32) / 255.0

    # ---- Step 6. Normalize output to [0,1] ----
    enhanced = np.clip(
        (enhanced - enhanced.min()) / (enhanced.max() - enhanced.min() + 1e-6),
        0, 1
    )

    # Re-apply mask (set invalids back to 0)
    enhanced[~valid_mask] = 0.0

    # Binarize
    # enhanced_thresh = enhanced > 0.3 * np.max(enhanced)

    # enhanced_morph = (enhanced_thresh.astype(np.uint8) * 255)
    # # Convert back to float [0,1] and apply a tiny Gaussian blur
    # enhanced_smoothed = cv2.GaussianBlur(enhanced_morph.astype(np.float32) / 255.0, (3, 3), sigmaX=1.0, sigmaY=1.0)
    # enhanced_smoothed = 1. - enhanced_smoothed  # Invert so sulci are bright

    enhanced_smoothed = enhanced # cv2.GaussianBlur(enhanced.astype(np.float32), (3, 3), sigmaX=1.0, sigmaY=1.0)

    # Erode valid mask to avoid edge artifacts at the mask border
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    _valid_u8 = (valid_mask.astype(np.uint8) * 255)
    _eroded_u8 = cv2.erode(_valid_u8, se, iterations=1)
    eroded_mask = _eroded_u8.astype(bool)

    # Apply eroded mask to remove border effects and return it
    enhanced_smoothed[~eroded_mask] = 0.0
    valid_mask = eroded_mask

    return enhanced_smoothed.astype(np.float32), valid_mask

def preprocess_curvature(curv, sigma_blur=1.0):
    """
    Preprocess curvature map to match photo appearance.
    
    Applies thresholding, morphological operations, and blurring to highlight
    gyri and sulci patterns.
    
    Parameters
    ----------
    curv : np.ndarray
        Input curvature map from cortical surface
    sigma_blur : float, optional
        Gaussian blur sigma for smoothing. Defaults to 1.0
    
    Returns
    -------
    np.ndarray
        Float32 image with sulci bright, normalized to [0, 1]
    """

    # Threshold to get gyrus/sulcus mask
    curv = np.asarray(curv, dtype=np.float32)
    curv_thresh = curv < 0.5 * np.max(curv)

    # Small morphological smoothing + slight Gaussian blur to remove tiny speckles
    curv_bin = (curv_thresh.astype(np.uint8) * 255)

    # Very small structuring element (5x5 ellipse)
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    # Open then close to remove noise but keep small features
    curv_morph = cv2.morphologyEx(curv_bin, cv2.MORPH_OPEN, se)
    # curv_morph = cv2.morphologyEx(curv_morph, cv2.MORPH_CLOSE, se)

    # Convert back to float [0,1] and apply a tiny Gaussian blur
    curv_smoothed = cv2.GaussianBlur(curv_morph.astype(np.float32) / 255.0, (3, 3), sigmaX=sigma_blur)

    # curv_smoothed = cv2.GaussianBlur(curv_thresh.astype(np.float32), (3, 3), sigmaX=sigma_blur, sigmaY=sigma_blur)
    curv_smoothed = 1. - curv_smoothed  # Invert so sulci are bright
    return curv_smoothed.astype(np.float32)

def sdf_similarity(photo_mask, curv_mask, mask=None):
    """
    Compute smooth similarity between two binary gyral masks.
    
    Uses the mean squared difference of signed distance maps to measure
    shape similarity between photo and curvature masks.
    
    Parameters
    ----------
    photo_mask : np.ndarray
        Binary mask from projected photo
    curv_mask : np.ndarray
        Binary mask from curvature map
    mask : np.ndarray, optional
        Optional restriction mask for the comparison
    
    Returns
    -------
    tuple
        (similarity, sdf_p, sdf_q) - Similarity score (higher = more similar),
        normalized signed distance field of photo mask, 
        normalized signed distance field of curvature mask
    """

    # Ensure binary float arrays
    p = (photo_mask > 0.5).astype(np.float32)
    q = (curv_mask > 0.5).astype(np.float32)

    # Compute signed distance fields (positive inside gyri)
    def sdf(x):
        pos = ndi.distance_transform_edt(x)
        neg = ndi.distance_transform_edt(1 - x)
        sdf = pos - neg
        return sdf / (np.max(np.abs(sdf)) + 1e-9)

    # Calculate SDFs and smooth via Gaussian filter
    sdf_p = ndi.gaussian_filter(sdf(p), sigma=1.0)
    sdf_q = ndi.gaussian_filter(sdf(q), sigma=1.0)

    # Optional mask restriction
    if mask is not None:
        m = mask.astype(bool)
        sdf_p = sdf_p[m]
        sdf_q = sdf_q[m]
    
    # Normalize
    sdf_p = sdf_p / (np.max(np.abs(sdf_p)) + 1e-9)
    sdf_q = sdf_q / (np.max(np.abs(sdf_q)) + 1e-9)

    # Convert squared difference to a similarity (negative cost)
    mse = np.mean((sdf_p - sdf_q) ** 2)
    sim = 1.0 / (1.0 + mse)  # smoother, bounded [0,1]

    return float(sim), sdf_p, sdf_q

def masked_normalized_mutual_information(img1, img2, mask, bins=64, window=32, eps=1e-10, return_map=False):
    """
    Compute masked normalized mutual information (NMI) between two grayscale images (float32 [0,1]),
    and return a local NMI map.
    
    Args:
        img1, img2 : np.ndarray (H, W), float32, range [0,1]
        mask       : np.ndarray (H, W), boolean or float, same size
        bins       : int, number of histogram bins
        window     : int, local window size for NMI map (sliding box)
        eps        : small constant to avoid log(0)
        return_map : bool, whether to return local NMI map
    
    Returns:
        global_nmi : float, normalized mutual information over masked pixels
        nmi_map    : np.ndarray (H, W), float32, local normalized MI map
    """
    # --- Input checks ---
    assert img1.shape == img2.shape == mask.shape, "Images and mask must have same shape"
    H, W = img1.shape

    # --- Flatten masked values ---
    m = mask > 0
    x = img1[m].ravel()
    y = img2[m].ravel()

    # --- Discretize into bins ---
    x_bin = np.clip((x * (bins - 1)).astype(np.int32), 0, bins - 1)
    y_bin = np.clip((y * (bins - 1)).astype(np.int32), 0, bins - 1)

    # --- Joint histogram ---
    hist_2d, _, _ = np.histogram2d(x_bin, y_bin, bins=bins, range=[[0, bins], [0, bins]])
    pxy = hist_2d / np.sum(hist_2d)
    px = np.sum(pxy, axis=1)
    py = np.sum(pxy, axis=0)

    # --- Entropies ---
    Hx = -np.sum(px[px > 0] * np.log(px[px > 0]))
    Hy = -np.sum(py[py > 0] * np.log(py[py > 0]))
    Hxy = -np.sum(pxy[pxy > 0] * np.log(pxy[pxy > 0]))

    # --- Normalized Mutual Information ---
    global_nmi = (Hx + Hy) / (Hxy + eps)

    # --- Local NMI map ---
    if return_map:
        # Compute local means using uniform filters (sliding box)
        def local_entropy(im):
            # Estimate local entropy as local variance proxy (simplified)
            p = im * mask
            mean = ndi.uniform_filter(p, window)
            mean2 = ndi.uniform_filter(p * p, window)
            var = np.maximum(mean2 - mean**2, 0)
            # Normalize to 0-1 entropy range
            return var / (var.max() + eps)

        ex = local_entropy(img1)
        ey = local_entropy(img2)
        e_joint = local_entropy(0.5 * (img1 + img2))
        
        nmi_map = (ex + ey) / (e_joint + eps)
        nmi_map *= mask.astype(np.float32)
        nmi_map = nmi_map.astype(np.float32)
    else:
        nmi_map = None

    return float(global_nmi), nmi_map

def vtk_image_to_gray_np(vtk_img):
    """
    Convert vtkImageData (RGB) to normalized grayscale numpy array (H,W).
    """
    # Get dimensions
    dims = vtk_img.GetDimensions()  # (x, y, z)
    w, h, z = dims

    # Get scalars
    scalars = vtk_img.GetPointData().GetScalars()
    comps = scalars.GetNumberOfComponents()
    flat = numpy_support.vtk_to_numpy(scalars)

    # Reshape correctly: VTK stores as (x-fastest, y, z)
    if z > 1:
        arr = flat.reshape(z, h, w, comps)
        arr = arr[0]  # take first slice
    else:
        arr = flat.reshape(h, w, comps)

    rgb = arr[..., :3].astype(np.float32)

    # Convert to grayscale
    gray = 0.2989*rgb[...,0] + 0.5870*rgb[...,1] + 0.1140*rgb[...,2]

    # Flip vertically
    gray = np.flipud(gray)

    # Normalize to [0., 1.]
    gray_min, gray_max = gray.min(), gray.max()
    gray = (gray - gray_min) / (gray_max - gray_min + 1e-9)

    return gray

def render_offscreen(renderer, renWin):
    """Render vtkRenderer offscreen and return vtkImageData."""
    renWin.Render()
    w2i = vtk.vtkWindowToImageFilter()
    w2i.SetInput(renWin)
    w2i.SetInputBufferTypeToRGBA()
    w2i.ReadFrontBufferOff()
    w2i.Update()
    return w2i.GetOutput()

def make_curv_actor_from_model(modelNode, scalarName="curv"):
    """
    Create a VTK actor and mapper that renders a model's point scalars
    as a grayscale curvature map.
    """
    poly = modelNode.GetPolyData()
    if poly is None:
        raise RuntimeError(f"Model {modelNode.GetName()} has no polydata.")

    pd = poly.GetPointData()
    scalar_array = pd.GetArray(scalarName)
    if scalar_array is None:
        raise RuntimeError(f"Scalar array '{scalarName}' not found on model {modelNode.GetName()}.")

    # Mapper
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(poly)
    mapper.SetScalarModeToUsePointFieldData()
    mapper.SelectColorArray(scalarName)
    mapper.SetScalarRange(scalar_array.GetRange())

    # Setup lookup table for grayscale mapping
    lut = vtk.vtkLookupTable()
    lut.SetNumberOfTableValues(256)
    lut.SetRange(mapper.GetScalarRange())  # match scalar min/max
    lut.SetSaturationRange(0.0, 0.0)       # zero saturation → grayscale
    lut.SetValueRange(0.0, 1.0)            # full brightness range
    lut.SetRampToLinear()
    lut.Build()

    # Apply lookup table to mapper
    mapper.SetLookupTable(lut)
    mapper.SetColorModeToMapScalars()
    mapper.SetScalarVisibility(True)

    # Actor
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    # Use ambient only to prevent shading from modifying scalar intensity
    actor.GetProperty().SetAmbient(1.0)
    actor.GetProperty().SetDiffuse(0.0)
    actor.GetProperty().SetSpecular(0.0)

    return actor, mapper

def make_scalar_actor_from_model(modelNode, scalarName="PhotoProjection"):
    """
    Create a VTK actor for rendering scalar data on a model as grayscale.
    
    Parameters
    ----------
    modelNode : vtkMRMLModelNode
        Model node containing the scalar array
    scalarName : str, optional
        Name of the scalar array to visualize. Defaults to "PhotoProjection"
    
    Returns
    -------
    tuple
        (actor, mapper) - VTK actor and mapper for the scalar visualization
    """

    # Create an actor for the envelope model
    actor = vtk.vtkActor()
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(modelNode.GetPolyData())
    mapper.SelectColorArray(scalarName)
    mapper.SetScalarModeToUsePointData()
    mapper.SetScalarRange(0, 255)

    # Setup lookup table for grayscale mapping
    lut = vtk.vtkLookupTable()
    lut.SetNumberOfTableValues(256)
    lut.SetRange(mapper.GetScalarRange())  # match scalar min/max
    lut.SetSaturationRange(0.0, 0.0)       # zero saturation → grayscale
    lut.SetValueRange(0.0, 1.0)            # full brightness range
    lut.SetRampToLinear()
    lut.Build()

    # Apply lookup table to mapper
    mapper.SetLookupTable(lut)
    mapper.SetColorModeToMapScalars()
    mapper.SetScalarVisibility(True)

    # Configure actor
    actor.SetMapper(mapper)
    actor.GetProperty().SetAmbient(1.0)
    actor.GetProperty().SetDiffuse(0.0)
    actor.GetProperty().SetSpecular(0.0)

    return actor, mapper

def auto_align_photo_to_brain(
    photoVolumeNode, transformNode, pialModelNode, envelopeModelNode,
    plane_dims, mask_path, MainProjection,
    scalarName="curv",
    # Coarse search parameters
    coarse_search_tu_mm=2.5,
    coarse_search_tv_mm=2.5,
    coarse_search_rotation_deg=5.0,
    coarse_search_cam_dist_ratio=0.5,
    coarse_search_scale=0.20,
    coarse_search_n_steps=5,
    # Fine search parameters
    fine_search_tu_mm=1.0,
    fine_search_tv_mm=1.0,
    fine_search_rotation_deg=1.5,
    fine_search_cam_dist_ratio=0.25,
    fine_search_scale=0.1,
    # Other parameters
    photo_sample_size=(128, 128),
    ren_camera_distance_mm=10.0
):
    """
    Two-stage automatic alignment of photo to brain surface.
    
    This is an EXPERIMENTAL feature that performs coarse-to-fine optimization
    to align a projected photograph with cortical curvature patterns. The alignment
    uses normalized mutual information (NMI) and signed distance field (SDF) 
    similarity metrics.
    
    WARNING: This feature is not yet stable and requires additional development
    before production use.
    
    Parameters
    ----------
    photoVolumeNode : vtkMRMLScalarVolumeNode
        Volume node containing the photograph
    transformNode : vtkMRMLLinearTransformNode
        Transform node to optimize
    pialModelNode : vtkMRMLModelNode
        Pial surface model with curvature scalars
    envelopeModelNode : vtkMRMLModelNode
        Brain envelope model for projection
    plane_dims : tuple
        (width, height) dimensions of photo plane in mm
    mask_path : str
        Path to .npz file containing photo masks
    MainProjection : Projection
        Main projection object to update
    scalarName : str, optional
        Name of curvature scalar on pial surface. Defaults to "curv"
    coarse_search_tu_mm : float, optional
        Coarse search range in U direction (mm). Defaults to 2.5
    coarse_search_tv_mm : float, optional
        Coarse search range in V direction (mm). Defaults to 2.5
    coarse_search_rotation_deg : float, optional
        Coarse search range for rotation (degrees). Defaults to 5.0
    coarse_search_cam_dist_ratio : float, optional
        Coarse search range for camera distance ratio. Defaults to 0.5
    coarse_search_scale : float, optional
        Coarse search range for scale. Defaults to 0.20
    coarse_search_n_steps : int, optional
        Number of steps per dimension in coarse grid. Defaults to 5
    fine_search_tu_mm : float, optional
        Fine search range in U direction (mm). Defaults to 1.0
    fine_search_tv_mm : float, optional
        Fine search range in V direction (mm). Defaults to 1.0
    fine_search_rotation_deg : float, optional
        Fine search range for rotation (degrees). Defaults to 1.5
    fine_search_cam_dist_ratio : float, optional
        Fine search range for camera distance ratio. Defaults to 0.25
    fine_search_scale : float, optional
        Fine search range for scale. Defaults to 0.1
    photo_sample_size : tuple, optional
        (width, height) size for rendering comparisons. Defaults to (128, 128)
    ren_camera_distance_mm : float, optional
        Rendering camera distance in mm. Defaults to 10.0
    
    Returns
    -------
    None or dict
        None if optimization cancelled or failed, otherwise optimization result dict
    """
    # Get base transform matrix
    vtkMat_base = vtk.vtkMatrix4x4()
    transformNode.GetMatrixTransformToWorld(vtkMat_base)
    base_M = np.array([[vtkMat_base.GetElement(i, j) for j in range(4)] for i in range(4)], dtype=np.float64)

    # Get base projection camera distance
    base_proj_cam_dist_mm = MainProjection.cam_dist_mm

    # Extract in-plane axes (tu, tv) and plane normal (tn)
    tu = base_M[:3, 0].copy()
    tv = base_M[:3, 1].copy()
    tn = base_M[:3, 2].copy()

    # Renderer (shared)
    ren = vtk.vtkRenderer()
    renWin = vtk.vtkRenderWindow()
    renWin.OffScreenRenderingOn()
    renWin.AddRenderer(ren)
    ren.SetBackground(0, 0, 0)
    renWin.SetSize(photo_sample_size)

    # Masked photo copy
    if slicer.mrmlScene.GetNodeByID("photo_masked_copy") is None:
        volumesLogic = slicer.modules.volumes.logic()
        photoVolumeNode_copy = volumesLogic.CloneVolume(slicer.mrmlScene, photoVolumeNode, "photo_masked_copy")
        util.load_photo_masks(mask_path, photoVolumeNode_copy, (0., 0., 0.), (0., 0., 0.))
    else:
        photoVolumeNode_copy = slicer.mrmlScene.GetNodeByID("photo_masked_copy")
    # Transform node copy
    if slicer.mrmlScene.GetNodeByID("plane_transform_copy") is None:
        transformNode_copy = slicer.mrmlScene.CopyNode(transformNode)
        transformNode_copy.SetName("plane_transform_copy")
    else:
        transformNode_copy = slicer.mrmlScene.GetNodeByID("plane_transform_copy")
    # Model node copy
    if slicer.mrmlScene.GetNodeByID("envelope_model_copy") is None:
        envelopeModelNode_copy = slicer.mrmlScene.CopyNode(envelopeModelNode)
        envelopeModelNode_copy.SetName("envelope_model_copy")
    else:
        envelopeModelNode_copy = slicer.mrmlScene.GetNodeByID("envelope_model_copy")
    # Pial model node copy
    if slicer.mrmlScene.GetNodeByID("pial_model_copy") is None:
        pialModelNode_copy = slicer.mrmlScene.CopyNode(pialModelNode)
        pialModelNode_copy.SetName("pial_model_copy")
    else:
        pialModelNode_copy = slicer.mrmlScene.GetNodeByID("pial_model_copy")

    # Projection setup (using proj_state as mutable storage for cam distance)
    Projection = util.Projection(
        envelopeModelNode_copy, photoVolumeNode_copy, transformNode_copy, plane_dims,
        cam_dist_mm=base_proj_cam_dist_mm, rgb=False, upsample_iterations=None,
        plane_to_cortex_distance=0., visualize_camera=False
    )

    # Curvature scalar setup
    # Only sample if 'curv' does not already exist
    # if envelopeModelNode_copy.GetPolyData().GetPointData().GetArray("curv") is None:
    #     util.sample_scalar_along_normals(
    #         envelopeModelNode_copy, pialModelNode,
    #         scalarName="curv", rayLength=25.0, attachToEnvelope=True
    #     )
    # else:
    #     print(f"[AutoAlign] 'curv' scalar already present on '{envelopeModelNode_copy.GetName()}'; skipping sampling.")

    # Actors
    curvActor, _ = make_curv_actor_from_model(pialModelNode_copy, scalarName)
    projActor, _ = make_scalar_actor_from_model(envelopeModelNode_copy)
    ren.AddActor(curvActor)
    ren.AddActor(projActor)

    def apply_M_to_transformNode(
            M, transform_node=transformNode_copy, model_node=envelopeModelNode_copy,
            trigger_update=True):
        # Apply candidate transform to the temporary transform node used for offscreen projection
        vtkM = vtk.vtkMatrix4x4()
        for i in range(4):
            for j in range(4):
                vtkM.SetElement(i, j, float(M[i, j]))
        transform_node.SetMatrixTransformToParent(vtkM)
        # ensure envelope model is marked modified so render/projection uses new transform
        if trigger_update:
            model_node.GetPolyData().Modified()

    # Helper to render gray images
    def render_gray(model_visibility):
        # Set visibilities
        curvActor.SetVisibility(model_visibility == "curv")
        projActor.SetVisibility(model_visibility == "proj")

        # Render and convert to gray numpy
        img = vtk_image_to_gray_np(render_offscreen(ren, renWin))
        return img

    def run_grid_search(
        start_M, start_proj_cam_dist_mm,
        ren, renWin, projection,
        ren_camera_distance_mm,
        photo_sample_size,
        progress,
        start_val=0,
        tu_range_mm=5.0, tv_range_mm=5.0,
        rot_range_deg=5.0, cam_dist_ratio_range=0.5, scale_range=0.2,
        n_steps=5,
        n_basins=3,
        min_param_distance=1.0,
        stage_name="GridSearch"
    ):
        """
        Exhaustive but coarse grid search around current plane pose.
        Evaluates NMI similarity for sampled (du, dv, rot, cam_ratio, scale)
        and returns up to n_basins best (non-adjacent) basins.
        """

        print(f"[{stage_name}] coarse grid search ({n_steps} steps per dim)...")
        progress.setLabelText(f"[{stage_name}]: evaluating grid...")
        slicer.app.processEvents()

        tu = start_M[:3, 0]
        tv = start_M[:3, 1]
        tn = start_M[:3, 2]

        du_vals = np.linspace(-tu_range_mm, tu_range_mm, n_steps)
        dv_vals = np.linspace(-tv_range_mm, tv_range_mm, n_steps)
        rot_vals = np.linspace(-rot_range_deg, rot_range_deg, n_steps)
        cam_vals = np.linspace(1.0 - cam_dist_ratio_range, 1.0 + cam_dist_ratio_range, n_steps)
        scale_vals = np.linspace(1.0 - scale_range, 1.0 + scale_range, n_steps)

        results = []
        count = 0

        for du in du_vals:
            for dv in dv_vals:
                for rot_deg in rot_vals:
                    for cam_ratio in cam_vals:
                        for scale in scale_vals:
                            count += 1
                            if progress.wasCanceled:
                                raise RuntimeError("User cancelled grid search.")

                            # Compose transform
                            rot_rad = math.radians(rot_deg)
                            k = tn / np.linalg.norm(tn)
                            K = np.array([[0, -k[2], k[1]],
                                          [k[2], 0, -k[0]],
                                          [-k[1], k[0], 0]])
                            dR = np.eye(3) + math.sin(rot_rad)*K + (1 - math.cos(rot_rad))*(K @ K)
                            dt = du*tu + dv*tv

                            M = np.eye(4)
                            M[:3, :3] = start_M[:3, :3] @ (dR * scale)
                            M[:3, 3] = start_M[:3, 3] + dt

                            projection.set_cam_distance(float(start_proj_cam_dist_mm) * cam_ratio, update=False)
                            apply_M_to_transformNode(M, trigger_update=False)

                            proj = render_gray("proj")
                            curv = render_gray("curv")
                            proj_proc, mask = preprocess_photo(proj)
                            curv_proc = preprocess_curvature(curv)

                            nmi, _ = masked_normalized_mutual_information(
                                proj_proc, curv_proc, mask, bins=64, window=32, return_map=False
                            )

                            results.append(((du, dv, rot_deg, cam_ratio, scale), nmi))

                            if count % 10 == 0:
                                progress.setValue(start_val + count)
                                slicer.app.processEvents()

        # Sort by score
        results.sort(key=lambda x: x[1], reverse=True)

        # Select top N non-adjacent basins
        basins = []
        for params, score in results:
            too_close = False
            for prev_params, _ in basins:
                dist = np.linalg.norm(np.array(params[:3]) - np.array(prev_params[:3]))
                if dist < min_param_distance:
                    too_close = True
                    break
            if not too_close:
                basins.append((params, score))
            if len(basins) >= n_basins:
                break

        print(f"[{stage_name}] found {len(basins)} candidate basins (best NMI={basins[0][1]:.4f})")

        basin_models = []
        for params, score in basins:
            du, dv, rot_deg, cam_ratio, scale = params
            rot_rad = math.radians(rot_deg)
            k = tn / np.linalg.norm(tn)
            K = np.array([[0, -k[2], k[1]],
                          [k[2], 0, -k[0]],
                          [-k[1], k[0], 0]])
            dR = np.eye(3) + math.sin(rot_rad)*K + (1 - math.cos(rot_rad))*(K @ K)
            dt = du*tu + dv*tv

            M = np.eye(4)
            M[:3, :3] = start_M[:3, :3] @ (dR * scale)
            M[:3, 3] = start_M[:3, 3] + dt
            cam_mm = float(start_proj_cam_dist_mm) * cam_ratio

            basin_models.append((score, M, cam_mm, params))

        return basin_models

    def run_gradient_opt(
        start_M, start_proj_cam_dist_mm,
        tu_range_mm, tv_range_mm, rot_range_deg, cam_dist_ratio_range, scale_range,
        ren, renWin, projection,
        ren_camera_distance_mm, photo_sample_size,
        progress,
        start_val=0,
        maxiter=150,
        sdf_smooth_sigma=1.0,
        param_scales=None,
        callback_step=1,
        stage_name="GradOpt"
    ):
        """
        Gradient-based optimization replacing the grid-search.
        Minimizes negative SDF-sim (i.e. maximizes SDF similarity).

        Parameters:
        - ranges: tuples like (half_range) specifying +/- search window around start_M settings
        - ren, renWin: VTK renderer/window (reused)
        - proj_state: mutable state used by your projection setup (so cam distance can be adjusted)
        - progress: slicer progress dialog
        - param_scales: array/list to scale parameters for optimizer (recommended)
        - sdf_smooth_sigma: gaussian sigma (in pixels) applied to sdf difference to smooth the loss (helps gradients)
        - maxiter: maximum optimizer iterations
        Returns:
        (best_params, best_score) where best_params = (du_mm, dv_mm, rot_deg, cam_dist_ratio, scale)
        """

        print(f"[{stage_name}] starting gradient-based optimization (maxiter={maxiter})...")
        progress.setLabelText(f"[{stage_name}]: gradient-based optimization...")

        # Parameterization: x = [du_mm, dv_mm, rot_deg, cam_ratio_mult, scale_mult]
        # start values (du,dv,rot are relative to start_M which is the current plane pose)
        x0 = np.array([0.0, 0.0, 0.0, 1.0, 1.0], dtype=np.float64)

        # bounds: restrict each parameter to the supplied ranges
        bounds = [
            (-tu_range_mm, tu_range_mm),
            (-tv_range_mm, tv_range_mm),
            (-rot_range_deg, rot_range_deg),
            (1.0 - cam_dist_ratio_range, 1.0 + cam_dist_ratio_range),
            (1.0 - scale_range, 1.0 + scale_range)
        ]

        # Default parameter scales if none provided (used only for optimizer scaling via initial guess / callbacks)
        if param_scales is None:
            # bring rotation to same order as mm: scales used only for display / sanity
            param_scales = np.array([1.0, 1.0, 1.0, 0.2, 0.1], dtype=np.float64)

        # Helper: build transform matrix from start_M and x
        def build_M_from_x(x):
            du, dv, rot_deg, cam_ratio, scale = float(x[0]), float(x[1]), float(x[2]), float(x[3]), float(x[4])
            tu = start_M[:3, 0]; tv = start_M[:3, 1]; tn = start_M[:3, 2]
            rot_rad = math.radians(rot_deg)
            k = tn / np.linalg.norm(tn)
            K = np.array([[0, -k[2], k[1]],
                        [k[2], 0, -k[0]],
                        [-k[1], k[0], 0]])
            dR = np.eye(3) + math.sin(rot_rad) * K + (1 - math.cos(rot_rad)) * (K @ K)
            dt = du * tu + dv * tv
            M = np.eye(4)
            M[:3, :3] = start_M[:3, :3] @ (dR * scale)
            M[:3, 3] = start_M[:3, 3] + dt
            return M, cam_ratio

        # The objective function to *minimize*: negative similarity (so minimizer maximizes similarity)
        eval_count = {"n": 0}
        last_reported = {"n": -1}

        def objective(x):
            if progress.wasCanceled:
                # raise an exception so minimize stops early and we can handle cancellation in caller
                raise RuntimeError("User cancelled optimization.")

            eval_count["n"] += 1

            # build M and place rendering camera
            M, _ = build_M_from_x(x)
            cam = vtk.vtkCamera()
            plane_origin = M[:3, 3]
            plane_normal = M[:3, 2]
            cam_pos = plane_origin + plane_normal * ren_camera_distance_mm
            cam.SetPosition(*cam_pos)
            cam.SetFocalPoint(*plane_origin)
            cam.SetViewUp(*M[:3, 1])
            cam.SetParallelProjection(True)
            cam.SetParallelScale(photo_sample_size[1] / 4)
            ren.SetActiveCamera(cam)

            # update projection camera distance
            cam_ratio = float(x[3])
            Projection.set_cam_distance(float(start_proj_cam_dist_mm) * cam_ratio, update=False)

            # Apply new transform
            apply_M_to_transformNode(
                M, transform_node=transformNode_copy, model_node=envelopeModelNode_copy,
                trigger_update=False
            )

            # Update projection (already triggered by transform modified above)
            # Projection.update()

            # Render and preprocess
            proj = render_gray("proj")
            curv = render_gray("curv")

            proj_proc, mask = preprocess_photo(proj)
            curv_proc = preprocess_curvature(curv)

            # # Binary masks for sdf similarity
            # bin_proj = proj_proc > 0.5
            # bin_curv = curv_proc > 0.5

            # # compute SDF similarity
            # sdf_sim, sdf_p, sdf_q = sdf_similarity(bin_proj, bin_curv, mask=mask)
            # Convert similarity to loss
            # mse = float((1.0 - sdf_sim) / sdf_sim)  # inverse of similarity
            # Compute NMI
            nmi, _ = masked_normalized_mutual_information(
                proj_proc, curv_proc, mask, bins=64, window=32, return_map=False
            )
            # Convert NMI to loss
            loss = 1.0 - nmi

            # Penalize deviation from 1.0 for scale lightly
            scale_dev = (x[4] - 1.0)
            reg = 1e-4 * (scale_dev**2)
            # Update loss
            loss += reg

            # Update progress display occasionally
            if eval_count["n"] % callback_step == 0:
                try:
                    progress.setValue(start_val + int(eval_count["n"]))
                    slicer.app.processEvents()
                except Exception:
                    pass

            return float(loss)

        # Callback used by optimizer at each iteration (xk is current parameter estimate)
        iter_count = {"n": 0}
        def callback(xk):
            iter_count["n"] += 1
            # compute similarity to report best-so-far (cheap approximate)
            try:
                # compute sim quickly (re-render) but don't write files
                cam_ratio = float(xk[3])
                projection.cam_dist_mm = float(start_proj_cam_dist_mm) * cam_ratio
                M, _ = build_M_from_x(xk)
                cam = vtk.vtkCamera()
                plane_origin = M[:3, 3]
                plane_normal = M[:3, 2]
                cam_pos = plane_origin + plane_normal * ren_camera_distance_mm
                cam.SetPosition(*cam_pos)
                cam.SetFocalPoint(*plane_origin)
                cam.SetViewUp(*M[:3, 1])
                cam.SetParallelProjection(True)
                cam.SetParallelScale(photo_sample_size[1] / 4)
                ren.SetActiveCamera(cam)

                proj = render_gray("proj")
                curv = render_gray("curv")
                proj_proc, mask = preprocess_photo(proj)
                curv_proc = preprocess_curvature(curv)
                sim_val, sdf_p, sdf_q = sdf_similarity(proj_proc > 0.3, curv_proc > 0.5, mask=mask)
                nmi, nmi_map = masked_normalized_mutual_information(
                    proj_proc, curv_proc, mask, bins=64, window=32, return_map=True
                )

                # Plot debug imgs
                plot_debug_images(proj_proc, curv_proc, mask, sdf_p, sdf_q, nmi_map)

                # optional print / log
                print(f"[{stage_name}] iter {iter_count['n']} NMI={nmi:.4f} SIM={sim_val:.4f} params={xk}")
            except Exception as e:
                # ignore callback errors (e.g., cancellation)
                print(f"[{stage_name}] callback error: {e}")

        # Run optimizer
        try:
            print(f"[{stage_name}] starting optimization...")
            res = optimize.minimize(
                objective,
                x0,
                method='Powell',
                bounds=bounds,
                options={'maxiter': maxiter, 'disp': True},  # 'ftol':1e-6, 'xtol':1e-6, 
                callback=callback
            )
            # # TODO: ------------------ DEBUG OUTPUT ------------------
            # print(f"[{stage_name}] optimization finished: {res.success}")
            # print(f"[{stage_name}] optimizer result object (res):")
            # pprint.pprint(res)
            # # --------------------------------------------------
        except RuntimeError as e:
            # Likely cancellation
            print(f"[{stage_name}] optimization stopped: {e}")
            return None, None

        # res.x gives best parameters found (relative to start_M)
        best_x = res.x
        best_loss = float(res.fun)

        # compute final similarity and return nicer score (SDF-sim)
        # set proj cam dist to best
        Projection.set_cam_distance(float(start_proj_cam_dist_mm) * float(best_x[3]))

        # final render to compute SDF-sim
        M_final, _ = build_M_from_x(best_x)
        cam = vtk.vtkCamera()
        plane_origin = M_final[:3, 3]
        plane_normal = M_final[:3, 2]
        cam_pos = plane_origin + plane_normal * ren_camera_distance_mm
        cam.SetPosition(*cam_pos)
        cam.SetFocalPoint(*plane_origin)
        cam.SetViewUp(*M_final[:3, 1])
        cam.SetParallelProjection(True)
        cam.SetParallelScale(photo_sample_size[1] / 4)
        ren.SetActiveCamera(cam)

        proj = render_gray("proj")
        curv = render_gray("curv")
        proj_proc, mask = preprocess_photo(proj)
        curv_proc = preprocess_curvature(curv)
        sdf_sim, sdf_p, sdf_q = sdf_similarity(proj_proc > 0.5, curv_proc > 0.5, mask=mask)
        nmi, nmi_map = masked_normalized_mutual_information(
            proj_proc, curv_proc, mask, bins=64, window=32, return_map=True
        )

        plot_debug_images(proj_proc, curv_proc, mask, sdf_p, sdf_q, nmi_map)

        # Translate best_x into human-readable tuple (du,dv,rot_deg, cam_ratio, scale)
        best_params = (float(best_x[0]), float(best_x[1]), float(best_x[2]), float(best_x[3]), float(best_x[4]))
        print(f"[{stage_name}] finished: NMI={nmi:.6f}, SDF-sim={sdf_sim:.6f}, loss={best_loss:.6e}, params={best_params}")

        return best_params, float(nmi)

    # Progress dialog setup
    total_iters_approx = coarse_search_n_steps**5 + 500

    progress = slicer.util.createProgressDialog(
        labelText="Aligning photo to cortex...",
        windowTitle="AutoAlign",
        minimum=0,
        maximum=total_iters_approx,
        value=0,
        cancelButtonText="Cancel"
    )

    # Stage 1: Coarse grid search
    coarse_search_ranges = dict(
        tu_range_mm=coarse_search_tu_mm,
        tv_range_mm=coarse_search_tv_mm,
        rot_range_deg=coarse_search_rotation_deg,
        cam_dist_ratio_range=coarse_search_cam_dist_ratio,
        scale_range=coarse_search_scale,
        n_steps=coarse_search_n_steps
    )

    basin_models = run_grid_search(
        base_M, base_proj_cam_dist_mm,
        ren=ren, renWin=renWin, projection=Projection,
        ren_camera_distance_mm=ren_camera_distance_mm,
        photo_sample_size=photo_sample_size,
        progress=progress,
        start_val=0,
        n_basins=5,  # number of top basins
        min_param_distance=1.0,
        **coarse_search_ranges
    )

    # Stage 2: Fine optimization for each basin
    fine_search_ranges = dict(
        tu_range_mm=fine_search_tu_mm,
        tv_range_mm=fine_search_tv_mm,
        rot_range_deg=fine_search_rotation_deg,
        cam_dist_ratio_range=fine_search_cam_dist_ratio,
        scale_range=fine_search_scale
    )

    best_overall_score = -np.inf
    best_overall_params = None
    best_overall_M = None
    best_overall_cam_mm = None

    for i, (score, basin_M, basin_cam_mm, params) in enumerate(basin_models):
        print(f"[AutoAlign] Optimizing basin {i+1}/{len(basin_models)} (coarse NMI={score:.4f})...")
        best_params, sim_score = run_gradient_opt(
            basin_M, basin_cam_mm,
            ren=ren, renWin=renWin, projection=Projection,
            ren_camera_distance_mm=ren_camera_distance_mm,
            photo_sample_size=photo_sample_size,
            progress=progress,
            start_val=coarse_search_n_steps**5 + i * 50,
            maxiter=10,
            sdf_smooth_sigma=1.0,
            stage_name=f"FineOpt_{i+1}",
            **fine_search_ranges
        )

        if best_params is None:
            continue

        if sim_score > best_overall_score:
            best_overall_score = sim_score
            best_overall_params = best_params
            best_overall_M = basin_M
            best_overall_cam_mm = basin_cam_mm

    progress.close()

    if best_params is None:
        return None

    # TODO: Debug. Idk whether this is correct. 
    tn = base_M[:3, 2]

    # Get final params
    du_f, dv_f, rot_f, d_cam_dist_f, scale_f = best_overall_params

    # Calculate final transform
    rot_rad = math.radians(rot_f)
    k = tn / np.linalg.norm(tn)
    K = np.array([[0, -k[2], k[1]],
                  [k[2], 0, -k[0]],
                  [-k[1], k[0], 0]])
    dR_f = np.eye(3) + math.sin(rot_rad) * K + (1 - math.cos(rot_rad)) * (K @ K)
    dt_f = du_f * base_M[:3, 0] + dv_f * base_M[:3, 1]

    final_M = np.eye(4)
    final_M[:3, :3] = best_overall_M[:3, :3] @ (dR_f * scale_f)
    final_M[:3, 3] = best_overall_M[:3, 3] + dt_f

    vtkM = vtk.vtkMatrix4x4()
    for i in range(4):
        for j in range(4):
            vtkM.SetElement(i, j, final_M[i, j])

    # Apply final transform to the real transform node
    transformNode.SetMatrixTransformToParent(vtkM)
    transformNode.InvokeEvent(slicer.vtkMRMLTransformNode.TransformModifiedEvent)
    # Apply final cam distance to MainProjection
    MainProjection.visualize_camera = True
    MainProjection.set_cam_distance(float(best_overall_cam_mm) * float(d_cam_dist_f), update=True)

    # Exit message
    print(f"\n[AutoAlign] done. NMI={best_overall_score:.4f}")
