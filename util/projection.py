"""
Photo projection onto 3D surfaces for 3D Slicer.
"""

import numpy as np
import vtk
from vtkmodules.util import numpy_support
import slicer
from .geometry import subdivide_model


class Projection:
    """
    Handles projection of a 2D photo volume onto a 3D model surface in 3D Slicer.
    Supports orthographic and perspective projection and automatically updates when
    the associated transform changes.

    Projective geometry is owned by the reference-photo coordinate system: the
    plane, transform, camera model, slab thickness, and cortex mapping define the
    shared reference-grid geometry. Secondary photographs can later be warped into
    the same reference grid and sampled through this same projection without
    creating independent 3D transforms.

    Usage:
        proj = Projection(
            modelNode, photoVolumeNode, transformNode,
            plane_dims=(100, 100), cam_dist_mm=100.0,
            rgb=True, visualize_camera=True
        )
        proj.update()  # manually force reproject
        proj.set_cam_distance(120.0)
        proj.remove()  # cleanup observers and visualization
    """

    def __init__(self, modelNode, photoVolumeNode, transformNode, plane_dims,
                 proj_slab_thickness_mm=15.0,
                 rgb=True,
                 upsample_iterations=2,
                 cam_dist_mm=None,
                 plane_to_cortex_distance=0.0,
                 visualize_camera=False):

        self.modelNode = modelNode
        self.photoVolumeNode = photoVolumeNode
        self.transformNode = transformNode
        self.plane_dims = plane_dims
        self.proj_slab_thickness_mm = proj_slab_thickness_mm
        self.rgb = rgb
        self.cam_dist_mm = cam_dist_mm
        self.plane_to_cortex_distance = plane_to_cortex_distance
        self.visualize_camera = visualize_camera
        self.upsample_iterations = upsample_iterations

        # Internal handles
        self._camera_actor = None
        self._rays_actor = None
        self._observer_tag = None

        # Setup
        self._prepare_data()
        self.update()  # initial projection

        # Setup display params
        self.displayNode.SetActiveScalarName("ProjectedPhoto")
        self.displayNode.SetScalarVisibility(True)
        self.displayNode.SetThresholdEnabled(True)
        self.displayNode.SetThresholdRange(1, 255)
        if rgb:
            self.displayNode.SetAutoScalarRange(True)
            self.displayNode.SetScalarRangeFlag(4)
        else:
            self.displayNode.SetAutoScalarRange(False)
            self.displayNode.SetScalarRange(30, 200)
            self.displayNode.SetAndObserveColorNodeID("vtkMRMLColorTableNodeGrey")

        self.displayNode.SetVisibility(False)

        # attach update observer
        self._observer_tag = self.transformNode.AddObserver(
            slicer.vtkMRMLTransformNode.TransformModifiedEvent,
            self._update_projection
        )

        print(f"[Projection] {'Perspective' if self.cam_dist_mm else 'Orthographic'} "
              f"projection set up on '{self.modelNode.GetName()}'.")

    # -------------------------------------------------------------------------
    # Setup helpers
    # -------------------------------------------------------------------------
    def _prepare_data(self):
        """Extract numpy data and precompute model points."""

        # Upsample model if requested
        if self.upsample_iterations is not None:
            subdivide_model(self.modelNode, iterations=self.upsample_iterations)
            print(f"[Projection] Upsampled model '{self.modelNode.GetName()}' "
                  f"with {self.upsample_iterations} iterations.")

        # Get model polydata and display node
        self.displayNode = self.modelNode.GetDisplayNode()
        self.envelopePoly = self.modelNode.GetPolyData()
        if self.envelopePoly is None:
            raise RuntimeError("Envelope polydata is empty.")

        # Cache model points
        npts = self.envelopePoly.GetNumberOfPoints()
        self.points_np = np.array(
            [self.envelopePoly.GetPoint(i) for i in range(npts)], dtype=np.float32
        )

        # Load image volume
        arr = slicer.util.arrayFromVolume(self.photoVolumeNode)
        if arr.ndim == 4:
            arr = arr[0]
        if arr.ndim == 3 and arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)

        if not self.rgb:
            if arr.shape[-1] >= 3:
                arr = np.dot(arr[..., :3], [0.2989, 0.5870, 0.1140])
            arr = arr.astype(np.uint8)
        else:
            if arr.shape[-1] == 4:
                arr = arr[..., :3]
            arr = np.clip(arr, 0, 255).astype(np.uint8)

        self.photo_array = arr
        self.img_h, self.img_w = arr.shape[:2]

    # -------------------------------------------------------------------------
    # Core projection logic
    # -------------------------------------------------------------------------
    def set_reference_image(self, photoVolumeNode):
        """Bind a different reference-grid image to the existing geometry.

        This keeps the same plane transform and cortex projection geometry while
        allowing another image already expressed in the reference-photo coordinate
        system to be sampled. The actual math remains the same as the original
        photo→cortex projection.
        """
        self.photoVolumeNode = photoVolumeNode
        self._prepare_data()
        self.update()

    def _update_projection(self, caller=None, event=None):
        """Observer callback."""
        self.update()

    def update(self):
        """Compute and apply projection to the model."""

        # --- Get transform ---
        vtkMat = vtk.vtkMatrix4x4()
        self.transformNode.GetMatrixTransformToWorld(vtkMat)
        M = np.array([[vtkMat.GetElement(i, j) for j in range(4)] for i in range(4)], dtype=np.float32)
        R = M[:3, :3]
        t = M[:3, 3]
        plane_normal = R[:, 2] / np.linalg.norm(R[:, 2])
        u_axis, v_axis = R[:, 0], R[:, 1]
        plane_origin = t + plane_normal * self.plane_to_cortex_distance

        # --- Camera setup ---
        camera_position = None
        if self.cam_dist_mm is not None:
            camera_position = plane_origin + plane_normal * self.cam_dist_mm

        # --- Projection geometry ---
        pts = self.points_np
        rel = pts - plane_origin
        dist = np.dot(rel, plane_normal)
        mask = np.abs(dist) <= self.proj_slab_thickness_mm
        if not np.any(mask):
            print("\x1b[38;5;208m[Projection] No model points within projection slab thickness.\x1b[0m")
            return

        if camera_position is None:
            proj = pts[mask] - np.outer(dist[mask], plane_normal)
            u = np.dot(proj - plane_origin, u_axis)
            v = np.dot(proj - plane_origin, v_axis)
            valid_points = np.where(mask)[0]
        else:
            mask_points = np.where(mask)[0]
            rays = pts[mask_points] - camera_position
            ray_norms = np.linalg.norm(rays, axis=1)
            rays /= ray_norms[:, None]
            denom = np.dot(rays, plane_normal)
            valid = np.abs(denom) > 1e-6
            valid_points = mask_points[valid]
            t_vals = np.dot(plane_origin - camera_position, plane_normal) / denom[valid]
            hit_points = camera_position + rays[valid] * t_vals[:, None]
            proj = hit_points
            u = np.dot(proj - plane_origin, u_axis)
            v = np.dot(proj - plane_origin, v_axis)

        # --- Convert to image indices ---
        v = -v  # invert vertical axis
        spacing_u = self.plane_dims[0] / self.img_w
        spacing_v = self.plane_dims[1] / self.img_h

        scale_u = np.linalg.norm(R[:, 0]) ** 2
        scale_v = np.linalg.norm(R[:, 1]) ** 2
        photo_size_x_mm = self.img_w * spacing_u * scale_u
        photo_size_y_mm = self.img_h * spacing_v * scale_v

        u_norm = 0.5 + (u / photo_size_x_mm)
        v_norm = 0.5 + (v / photo_size_y_mm)

        valid_mask = (u_norm >= 0) & (u_norm < 1) & (v_norm >= 0) & (v_norm < 1)
        if not np.any(valid_mask):
            raise RuntimeError("No model points project inside the photo image bounds.")
            return

        u_idx = (u_norm[valid_mask] * (self.img_w - 1)).astype(int)
        v_idx = (v_norm[valid_mask] * (self.img_h - 1)).astype(int)
        valid_points = valid_points[valid_mask]

        # --- Sample colors and assign ---
        npts = self.points_np.shape[0]
        if self.rgb:
            sampled = self.photo_array[v_idx, u_idx, :]
            colors = np.zeros((npts, 3), dtype=np.uint8)
            colors[valid_points] = sampled
            vtk_colors = numpy_support.numpy_to_vtk(colors, deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
            vtk_colors.SetNumberOfComponents(3)
            vtk_colors.SetName("ProjectedPhoto")
            if self.envelopePoly.GetPointData().HasArray("ProjectedPhoto"):
                self.envelopePoly.GetPointData().RemoveArray("ProjectedPhoto")
            self.envelopePoly.GetPointData().AddArray(vtk_colors)
            self.envelopePoly.GetPointData().SetActiveScalars("ProjectedPhoto")
        else:
            sampled = self.photo_array[v_idx, u_idx]
            grays = np.zeros(npts, dtype=np.uint8)
            grays[valid_points] = sampled
            vtk_gray = numpy_support.numpy_to_vtk(grays, deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
            vtk_gray.SetNumberOfComponents(1)
            vtk_gray.SetName("ProjectedPhoto")
            if self.envelopePoly.GetPointData().HasArray("ProjectedPhoto"):
                self.envelopePoly.GetPointData().RemoveArray("ProjectedPhoto")
            self.envelopePoly.GetPointData().AddArray(vtk_gray)
            self.envelopePoly.GetPointData().SetActiveScalars("ProjectedPhoto")

        self.envelopePoly.Modified()
        self.displayNode.Modified()
        self.displayNode.SetActiveScalarName("ProjectedPhoto")
        self.displayNode.SetScalarVisibility(True)

        # Optional visualization
        if self.visualize_camera and camera_position is not None:
            self._show_camera_geometry(plane_origin, u_axis, v_axis, plane_normal, camera_position)

    # -------------------------------------------------------------------------
    # Visualization helpers
    # -------------------------------------------------------------------------
    def _show_camera_geometry(self, plane_origin, u_axis, v_axis, plane_normal, camera_position):
        """Draw camera sphere + rays for debugging."""
        self.hide_camera()
        sphere = vtk.vtkSphereSource()
        sphere.SetCenter(camera_position)
        sphere.SetRadius(self.cam_dist_mm * 0.02)
        sphere.Update()

        self._camera_actor = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", "CameraPosition")
        self._camera_actor.SetAndObservePolyData(sphere.GetOutput())
        disp = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelDisplayNode")
        disp.SetColor(1, 0, 0)
        disp.SetVisibility(True)
        self._camera_actor.SetAndObserveDisplayNodeID(disp.GetID())

        half_w, half_h = self.plane_dims[0] / 2, self.plane_dims[1] / 2
        corners = [
            plane_origin + u_axis * half_w + v_axis * half_h,
            plane_origin - u_axis * half_w + v_axis * half_h,
            plane_origin - u_axis * half_w - v_axis * half_h,
            plane_origin + u_axis * half_w - v_axis * half_h
        ]

        linesPoly = vtk.vtkPolyData()
        points = vtk.vtkPoints()
        lines = vtk.vtkCellArray()
        pid = 0
        for corner in corners:
            points.InsertNextPoint(camera_position)
            points.InsertNextPoint(corner)
            line = vtk.vtkLine()
            line.GetPointIds().SetId(0, pid)
            line.GetPointIds().SetId(1, pid + 1)
            lines.InsertNextCell(line)
            pid += 2
        linesPoly.SetPoints(points)
        linesPoly.SetLines(lines)

        self._rays_actor = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", "CameraRays")
        self._rays_actor.SetAndObservePolyData(linesPoly)
        disp2 = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelDisplayNode")
        disp2.SetColor(0, 1, 0)
        disp2.SetVisibility(True)
        self._rays_actor.SetAndObserveDisplayNodeID(disp2.GetID())

    def hide_camera(self):
        if self._camera_actor:
            slicer.mrmlScene.RemoveNode(self._camera_actor)
            self._camera_actor = None
        if self._rays_actor:
            slicer.mrmlScene.RemoveNode(self._rays_actor)
            self._rays_actor = None

    # -------------------------------------------------------------------------
    # Parameter control
    # -------------------------------------------------------------------------
    def set_cam_distance(self, dist_mm, update=True):
        self.cam_dist_mm = dist_mm
        if update:
            self.update()

    def set_plane_offset(self, offset_mm, update=True):
        self.plane_to_cortex_distance = offset_mm
        if update:
            self.update()

    # -------------------------------------------------------------------------
    # Cleanup
    # -------------------------------------------------------------------------
    def remove(self):
        """Remove observers and visualization nodes."""
        if self._observer_tag is not None:
            self.transformNode.RemoveObserver(self._observer_tag)
            self._observer_tag = None
        self.hide_camera()
        print("[Projection] Removed observers and visualization.")
