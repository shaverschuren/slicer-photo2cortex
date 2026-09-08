"""
UI, camera, and interaction handling for 3D Slicer.
"""

import sys
import numpy as np
import vtk
import qt  # type: ignore
import ctk  # type: ignore
import slicer
import os
import surf2vol
import optimizer
from .geometry import get_poly_normals, vtkMatrixToNumpy, numpyToVtkMatrix, extractRotationScale, rotationFromVectors
from .io import save_scene_to_directory


def center_camera_on_projection(Nodes):
    """Center the 3D view camera on the projection plane normal."""

    threeDView = slicer.app.layoutManager().threeDWidget(0).threeDView()
    # Get plane normal in world coordinates
    vtkMat = vtk.vtkMatrix4x4()
    Nodes['transformNode'].GetMatrixTransformToWorld(vtkMat)
    R = np.array([[vtkMat.GetElement(i, j) for j in range(3)] for i in range(3)])
    normal = R[:, 2] / np.linalg.norm(R[:, 2])

    # Get plane origin in world coordinates
    origin = np.array([vtkMat.GetElement(i, 3) for i in range(3)])

    # Set camera position and focal point
    camera = threeDView.cameraNode()
    focal_point = origin
    # Move camera to same distance from focal point, but along the plane normal (arc to normal)
    current_position = np.array(camera.GetPosition())
    distance = np.linalg.norm(current_position - focal_point)
    camera_position = focal_point + normal * distance

    camera.SetFocalPoint(*focal_point)
    camera.SetPosition(*camera_position)
    camera.SetViewUp(0, 0, 1)
    threeDView.renderWindow().Render()

class PhotoTransformObserver:
    """
    Enables interactive dragging of a plane along an envelope surface.
    Keeps the plane offset along the normal, allows rotation and scaling, and
    updates smoothly when the transform node changes.

    Usage:
        observer = PhotoTransformObserver(transformNode, planeNode, envelopeModelNode, offset_mm=2.0)
        observer.start()
        ...
        observer.stop()
    """

    def __init__(self, transformNode, planeNode, envelopeModelNode, offset_mm=0.0, initial_scale=0.8):
        self.transformNode = transformNode
        self.planeNode = planeNode
        self.envelopeModelNode = envelopeModelNode
        self.offset_mm = offset_mm
        self.initial_scale = initial_scale

        self._observerTag = None
        self._isUpdating = False
        self._state = {
            'start_point': None,
            'start_translation': None,
            'start_matrix': None,
            'start_normal': None
        }

        # Precompute envelope geometry
        self.envelopePoly = self.envelopeModelNode.GetPolyData()
        if self.envelopePoly is None:
            raise RuntimeError("Envelope polydata is empty")

        self.normals = get_poly_normals(self.envelopePoly)
        self.pointLocator = vtk.vtkPointLocator()
        self.pointLocator.SetDataSet(self.envelopePoly)
        self.pointLocator.BuildLocator()

        # Initialize transform
        self._initialize_transform()

    # ------------------------------------------------------------------
    def _initialize_transform(self):
        bounds = self.envelopePoly.GetBounds()
        lateral_x = bounds[0]
        center_y = (bounds[2] + bounds[3]) / 2
        center_z = (bounds[4] + bounds[5]) / 2
        lateral_edge = np.array([lateral_x, center_y, center_z])
        pid = self.pointLocator.FindClosestPoint(lateral_edge)
        closestPoint = np.array(self.envelopePoly.GetPoint(pid))
        env_normal = np.array(self.normals.GetTuple(pid))
        env_normal /= np.linalg.norm(env_normal)

        # Setup orientation
        z_axis = env_normal
        arbitrary = np.array([1, 0, 0]) if abs(z_axis[0]) < 0.99 else np.array([0, 1, 0])
        x_axis = np.cross(arbitrary, z_axis)
        x_axis /= np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        y_axis /= np.linalg.norm(y_axis)

        initialMat = vtk.vtkMatrix4x4()
        initialMat.Identity()
        for i in range(3):
            initialMat.SetElement(i, 0, x_axis[i] * self.initial_scale)
            initialMat.SetElement(i, 1, y_axis[i] * self.initial_scale)
            initialMat.SetElement(i, 2, z_axis[i] * self.initial_scale)
            initialMat.SetElement(i, 3, closestPoint[i] + env_normal[i] * self.offset_mm)

        self.transformNode.SetMatrixTransformToParent(initialMat)

    # ------------------------------------------------------------------
    def _onModified(self, caller=None, event=None):
        if self._isUpdating:
            return
        self._isUpdating = True
        try:
            vtkMat = vtk.vtkMatrix4x4()
            self.transformNode.GetMatrixTransformToParent(vtkMat)
            mat = vtkMatrixToNumpy(vtkMat)

            trans = mat[:3, 3]
            R_user, scale_user = extractRotationScale(mat[:3, :3])

            # Detect pure scaling
            if (
                self._state["start_translation"] is not None
                and np.allclose(trans, self._state["start_translation"])
                and np.allclose(R_user, extractRotationScale(self._state["start_matrix"][:3, :3])[0])
            ):
                scale_matrix = mat[:3, :3]
                s = np.linalg.norm(scale_matrix[:, 0])
                if not np.allclose(np.linalg.norm(scale_matrix[:, 1]), s) or not np.allclose(np.linalg.norm(scale_matrix[:, 2]), s):
                    # Reset to uniform scale
                    R_user = extractRotationScale(self._state["start_matrix"][:3, :3])[0]
                    new_mat = np.eye(4)
                    new_mat[:3, :3] = R_user * s
                    new_mat[:3, 3] = self._state["start_translation"]
                    vtkNewMat = numpyToVtkMatrix(new_mat)
                    self.transformNode.SetMatrixTransformToParent(vtkNewMat)
                return

            # Envelope lookup
            pid = self.pointLocator.FindClosestPoint(trans)
            closestPoint = np.asarray(self.envelopePoly.GetPoint(pid))
            env_normal = np.asarray(self.normals.GetTuple(pid))
            env_normal /= np.linalg.norm(env_normal)

            # Initialize state
            if self._state["start_point"] is None:
                self._state.update({
                    "start_point": closestPoint,
                    "start_translation": trans.copy(),
                    "start_matrix": mat.copy(),
                    "start_normal": env_normal.copy(),
                })
                return

            new_translation = closestPoint + env_normal * self.offset_mm
            plane_local_z = R_user[:, 2]
            R_align = rotationFromVectors(plane_local_z, env_normal)
            R_final = R_align @ R_user

            new_mat = np.eye(4)
            new_mat[:3, :3] = R_final * scale_user
            new_mat[:3, 3] = new_translation

            self._state.update({
                "start_point": closestPoint,
                "start_translation": new_translation.copy(),
                "start_matrix": new_mat.copy(),
                "start_normal": env_normal.copy(),
            })

            vtkNewMat = numpyToVtkMatrix(new_mat)
            self.transformNode.SetMatrixTransformToParent(vtkNewMat)

        finally:
            self._isUpdating = False

    # ------------------------------------------------------------------
    def start(self):
        """Attach the observer."""
        if self._observerTag is None:
            self._observerTag = self.transformNode.AddObserver(
                slicer.vtkMRMLLinearTransformNode.TransformModifiedEvent, self._onModified
            )
            print(f"PhotoTransformDragger: observing '{self.transformNode.GetName()}'")

    def stop(self):
        """Detach the observer."""
        if self._observerTag is not None:
            self.transformNode.RemoveObserver(self._observerTag)
            print(f"PhotoTransformDragger: stopped observing '{self.transformNode.GetName()}'")
            self._observerTag = None

def setup_interactive_transform(transformNode, visibility=True, limit_to_surf_aligned=True):
    """
    Set up transform node for interactive manipulation with handle controls.
    
    Parameters
    ----------
    transformNode : vtkMRMLLinearTransformNode
        Transform node to make interactive
    visibility : bool, optional
        Whether to show interactive handles. Defaults to True
    limit_to_surf_aligned : bool, optional
        If True, limits rotation to Z-axis only and translation to XY plane.
        If False, allows full 3D rotation and translation. Defaults to True
    """

    # Ensure transform node has a display node
    if not transformNode.GetDisplayNode():
        displayNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTransformDisplayNode")
        transformNode.SetAndObserveDisplayNodeID(displayNode.GetID())
    else:
        displayNode = transformNode.GetDisplayNode()

    # Enable interactive handles
    if visibility:
        displayNode.SetEditorVisibility(True)
        displayNode.EditorTranslationEnabledOn()
        displayNode.EditorRotationEnabledOn()
        displayNode.EditorScalingEnabledOn()
        displayNode.Visibility3DOn()
        displayNode.Visibility2DOff()

        if limit_to_surf_aligned:
            displayNode.SetRotationHandleComponentVisibility3D(False, False, True, False)       # Z rotation only
            displayNode.SetTranslationHandleComponentVisibility3D(True, True, False, True)      # X/Y/viewing plane translation only
        else:
            displayNode.SetRotationHandleComponentVisibility3D(True, True, True, False)         # XYZ rotations
            displayNode.SetTranslationHandleComponentVisibility3D(True, True, True, True)      # All translations

        displayNode.SetScaleHandleComponentVisibility3D(True, False, False, False)              # X/Y scaling only
        displayNode.SetTranslationHandleComponentVisibilitySlice(False, False, False, False)    # Not in 2D
        displayNode.SetScaleHandleComponentVisibilitySlice(True, False, False, False)           # Not in 2D
        displayNode.SetRotationHandleComponentVisibilitySlice(False, False, False, False)       # Not in 2D
    else:
        displayNode.SetEditorVisibility(False)
        displayNode.Visibility3DOff()
        displayNode.Visibility2DOff()

def setup_ui_widgets(MainProjection, transformObserver, transformNode,
                     min_mm=1.0, max_mm=300.0, initial_mm=150.0):
    """
    Create a 'Projector Control' dock widget for controlling projection settings.
    
    Creates a docked panel with camera distance slider and snap-to-surface toggle.
    The widget is safely attached to the main window and survives layout changes.
    
    Parameters
    ----------
    MainProjection : Projection
        The projection object to control
    transformObserver : PhotoTransformObserver
        The transform observer that handles dragging constraints
    transformNode : vtkMRMLLinearTransformNode
        The transform node being controlled
    min_mm : float, optional
        Minimum camera distance in mm. Defaults to 1.0
    max_mm : float, optional
        Maximum camera distance in mm. Defaults to 300.0
    initial_mm : float, optional
        Initial camera distance in mm. Defaults to 150.0
    
    Returns
    -------
    tuple
        (slider, toggle, dockWidget) - The camera distance slider, snap toggle checkbox, 
        and the dock widget container
    """

    mainWindow = slicer.util.mainWindow()

    # --- Create a collapsible section as the inner content ---
    collapsibleButton = ctk.ctkCollapsibleButton()
    collapsibleButton.text = "Projector Control"
    collapsibleButton.collapsed = False

    formLayout = qt.QFormLayout(collapsibleButton)
    formLayout.setContentsMargins(8, 4, 8, 8)

    # --- Slider ---
    slider = ctk.ctkSliderWidget()
    slider.minimum = min_mm
    slider.maximum = max_mm
    slider.value = initial_mm
    slider.singleStep = 1
    slider.decimals = 1
    slider.setToolTip("Adjust projection camera distance (mm)")
    slider.setFixedWidth(300)
    formLayout.addRow("Camera distance (mm):", slider)

    # --- Toggle ---
    toggle = ctk.ctkCheckBox()
    toggle.text = "Snap to surface"
    toggle.setChecked(True)
    toggle.setToolTip("Enable/disable dragging constrained to surface")
    formLayout.addRow("Transform mode:", toggle)

    # --- Make a dock widget ---
    dockWidget = qt.QDockWidget("Projector Control")
    dockWidget.setObjectName("ProjectorControlDock")
    dockWidget.setWidget(collapsibleButton)
    dockWidget.setFeatures(qt.QDockWidget.DockWidgetFloatable | qt.QDockWidget.DockWidgetMovable)

    # --- Add dock widget below the 3D view ---
    mainWindow.addDockWidget(qt.Qt.BottomDockWidgetArea, dockWidget)

    # --- Slider connection ---
    def onValueChanged(value):
        MainProjection.set_cam_distance(value)
    slider.connect('valueChanged(double)', onValueChanged)

    # --- External update timer ---
    timer = qt.QTimer()
    timer.setInterval(200)
    timer.start()

    last_value = MainProjection.cam_dist_mm

    def update_slider():
        nonlocal last_value
        current = MainProjection.cam_dist_mm
        if current != last_value:
            slider.blockSignals(True)
            slider.value = current
            slider.blockSignals(False)
            last_value = current

    timer.timeout.connect(update_slider)

    # --- Toggle observer ---
    def onToggle(state):
        if state:
            transformObserver.start()
            setup_interactive_transform(transformNode, visibility=True, limit_to_surf_aligned=True)
        else:
            transformObserver.stop()
            setup_interactive_transform(transformNode, visibility=True, limit_to_surf_aligned=False)

    toggle.connect('toggled(bool)', onToggle)

    # --- Projection depth slider ---
    depthSlider = ctk.ctkSliderWidget()
    depthSlider.minimum = 1
    depthSlider.maximum = 50
    depthSlider.value = 20
    depthSlider.singleStep = 1
    depthSlider.decimals = 0
    depthSlider.setFixedWidth(300)
    depthSlider.setToolTip("Projection depth for surface-to-volume mask (mm)")
    formLayout.addRow("Surf2vol depth (mm):", depthSlider)

    # Store in global so surf2vol can read it later
    globals()["surf2vol_depth_mm"] = 20.

    def onDepthChanged(value):
        globals()["surf2vol_depth_mm"] = float(value)

    depthSlider.connect('valueChanged(double)', onDepthChanged)

    # --- Include-white-matter toggle ---
    includeWM = qt.QCheckBox()
    includeWM.setChecked(False)
    includeWM.text = "Include white matter"
    includeWM.setToolTip(
        "If enabled, projection includes white matter; otherwise only gray matter "
        "with morphological closing."
    )
    formLayout.addRow("Surf2vol mode:", includeWM)

    # Store the setting
    globals()["surf2vol_include_wm"] = False
    def onIncludeWM(state):
        globals()["surf2vol_include_wm"] = bool(state)

    includeWM.stateChanged.connect(onIncludeWM)

    print("[UI] Added 'Projector Control' dock below 3D view")

    return slider, toggle, depthSlider, includeWM, dockWidget

def setup_interactor(Nodes, plane_dims, photo_mask_path, MainProjection, transformObserver, output_dir):
    """
    Install keypress handlers on all 3D and slice view interactors.

    This prepares the application to respond to keyboard shortcuts that control
    layout, camera centering, projection alignment, curve drawing, saving, and quitting.
    
    Keyboard shortcuts:
    - '1': Switch to 3D view only
    - '2': Switch to red slice view only
    - 'space': Center camera on projection plane
    - 'Return': Align projection plane to current camera view
    - 'a': Auto-align projection plane (experimental)
    - 'v': Create volumetric resection mask from aligned surfaces
    - 's': Save scene and resection mask
    - 'q': Quit Slicer
    - 'x': Mark as atlas-based and quit Slicer
    - 'Escape': Quit Slicer and break outside loop
    
    Parameters
    ----------
    Nodes : dict
        Dictionary containing Slicer nodes (must include 'transformNode', 
        'brain_envelopeNode', and other scene nodes)
    plane_dims : tuple
        (width, height) dimensions of the photo plane in mm
    photo_mask_path : str
        Path to the .npz file containing photo masks
    MainProjection : Projection
        The projection object handling photo projection
    transformObserver : PhotoTransformObserver
        The transform observer managing dragging constraints
    output_dir : str
        Directory for saving output files
    """

    # Get app
    app = slicer.app

    # Initialize quit flag
    quit_once = False

    # Set up keypress observer
    def onKeyPress(interactor):

        # Get key
        key = interactor.GetKeySym()

        # Switch views with 1, 2 keys
        if key == "1":
            slicer.app.layoutManager().setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutOneUp3DView)
        elif key == "2":
            slicer.app.layoutManager().setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutOneUpRedSliceView)

        # Center camera on plane with spacebar
        elif key == "space":
            center_camera_on_projection(Nodes)
        
        # Align projection plane to camera view and make projection visible with Enter key
        elif key == "Return":
            # Get camera position
            threeDView = slicer.app.layoutManager().threeDWidget(0).threeDView()
            camera = threeDView.cameraNode()
            cam_pos = np.array(camera.GetPosition())

            # find closest point on model to camera position
            poly = Nodes['brain_envelopeNode'].GetPolyData()
            if poly is None:
                raise RuntimeError("Model polydata is empty")

            locator = vtk.vtkPointLocator()
            locator.SetDataSet(poly)
            locator.BuildLocator()
            pid = locator.FindClosestPoint(tuple(cam_pos))
            closest_pt = np.array(poly.GetPoint(pid))

            # Move transformNode translation to the closest point (preserve rotation/scale)
            vtkMatParent = vtk.vtkMatrix4x4()
            Nodes['transformNode'].GetMatrixTransformToParent(vtkMatParent)
            mat_np = np.array([[vtkMatParent.GetElement(i, j) for j in range(4)] for i in range(4)], dtype=float)
            mat_np[:3, 3] = closest_pt
            vtkNew = numpyToVtkMatrix(mat_np)
            Nodes['transformNode'].SetMatrixTransformToParent(vtkNew)
            # Trigger transform observer once
            transformObserver._onModified()

            # Make projection and transformation interaction visible
            Nodes['brain_envelopeNode'].GetDisplayNode().SetVisibility(True)
            setup_interactive_transform(Nodes['transformNode'], visibility=True, limit_to_surf_aligned=True)

        # Auto-align projection plane with "a" key
        # TODO: Perform check if lh or rh based on coords
        elif key == "a":
            optimizer.auto_align_photo_to_brain(
                Nodes['photoVolumeNode'], Nodes['transformNode'], Nodes['rh_pialNode'],
                Nodes['rh_envelopeNode'], plane_dims, photo_mask_path, MainProjection
                )

        # Make volumetric resection mask from aligned surfaces.
        elif key == "v":
            # Generate segmentation from ribbon and brain envelope
            depth = globals().get('surf2vol_depth_mm', 20)
            include_wm = globals().get('surf2vol_include_wm', True)
            segmentationNode = surf2vol.project_surface_to_volume_mask(
                Nodes["brain_envelopeNode"], Nodes["ribbonNode"], max_depth_mm=depth, include_wm=include_wm
            )
            # Store globally for access
            globals()['segmentationNode'] = segmentationNode
            Nodes['segmentationNode'] = segmentationNode
            # Set opacity for visualization
            Nodes['lh_pialNode'].GetDisplayNode().SetOpacity(0.3)
            Nodes['rh_pialNode'].GetDisplayNode().SetOpacity(0.3)
            Nodes['brain_envelopeNode'].GetDisplayNode().SetOpacity(0.4)
        # Save resection curve with "s" key
        elif key == "s":
            segmentationNode = Nodes.get('segmentationNode', None)
            if segmentationNode:
                # Save resection mask to output directory
                output_nifti_path = os.path.join(output_dir, "photo2cortex_resection_mask.nii.gz")
                surf2vol.save_resection_mask(segmentationNode, output_nifti_path)
                # Save scene
                scene_path = os.path.join(output_dir, "scene")
                save_scene_to_directory(scene_path, Nodes)
            else:
                print("No segmentationNode found to save.")
        # Exit program and label as atlas-based mask with "x" key
        elif key == "x":
            # Label subject as atlas-based to come back to later (for lobectomies etc.)
            with open(os.path.join(output_dir, "atlas_based.txt"), "w") as f:
                f.write("This subject uses an atlas-based resection mask.\n")
            # Quit app
            sys.exit(0)
        # Exit program with "q" key
        elif key == "q":
            # Check whether to save before quitting
            segmentationNode = Nodes.get('segmentationNode', None)
            if segmentationNode and not (
                os.path.exists(os.path.join(output_dir, "photo2cortex_resection_mask.nii.gz")) \
                and os.path.exists(os.path.join(output_dir, "scene"))
            ):
                nonlocal quit_once
                if not quit_once:
                    print("Don't forget to save your resection mask and scene before quitting! (press 's')")
                    print("Press 'q' again to quit without saving.")
                    quit_once = True
                else:
                    print("Exiting application without saving.")
                    sys.exit(1)
            elif not segmentationNode:
                print("Exiting without results to save.")
                sys.exit(1)
            else:
                print("Exiting application.")
                sys.exit(0)
        # Exit program with return code 2 with "escape" key to break loop
        elif key == "Escape":
            # Quit application
            sys.exit(2)
        # Ignore other keys
        else:
            return

    # Also install filter on all 3D and slice view interactors
    lm = app.layoutManager()
    for viewIndex in range(lm.threeDViewCount):
        interactor = lm.threeDWidget(viewIndex).threeDView().interactor()
        interactor.AddObserver("KeyPressEvent", lambda obj, evt: onKeyPress(obj))

    for name in lm.sliceViewNames():
        interactor = lm.sliceWidget(name).sliceView().interactor()
