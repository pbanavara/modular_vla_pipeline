import numpy as np
import mujoco
from PIL import Image
from log.setup_logger import setup_logger

class VisionFrame:
    def __init__(
        self,
        mujoco_model_path: str,
        mujoco_camera_name: str,
        rgb_image: np.ndarray,
        image_shape: tuple,
        object_name: str,
    ):
        self.rgb = rgb_image
        self.model = mujoco.MjModel.from_xml_path(mujoco_model_path)
        self.cam_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_CAMERA, mujoco_camera_name
        )
        if self.cam_id == -1:
            raise ValueError(f"Camera {mujoco_camera_name} not found in model")

        self.width, self.height = image_shape
        self.logger = setup_logger("vision_frame")
        self.fx, self.fy, self.cx, self.cy = self._compute_intrinsics()
        self.object_dimensions = self._extract_object_dimensions(object_name)

    def _compute_intrinsics(self):
        fovy_rad = np.deg2rad(self.model.cam_fovy[self.cam_id])
        fy = 0.5 * self.height / np.tan(fovy_rad / 2)
        fx = fy  # square pixels assumed
        cx = self.width / 2
        cy = self.height / 2
        return fx, fy, cx, cy

    def _extract_object_dimensions(self, name: str) -> dict:
        dims = {}
        try:
            geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if geom_id == -1:
                raise ValueError(f"Geometry with name '{name}' not found in model.")

            geom_type = self.model.geom_type[geom_id]
            size = self.model.geom_size[geom_id]

            if geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
                dims[name] = size[0] * 2  # diameter
            elif geom_type == mujoco.mjtGeom.mjGEOM_BOX:
                dims[name] = size[0] * 2  # width
            else:
                self.logger.info(f"[VisionFrame] Warning: Unknown geom type for {name}")
        except KeyError:
            self.logger.info(f"[VisionFrame] Warning: {name} not found in model")
        return dims

    def estimate_depth_from_mask(self, 
                                 mask: np.ndarray, 
                                 object_name: str) -> float:
        """
        Estimate depth using known object width and its pixel width in the mask.

        Args:
            mask (np.ndarray): binary mask for the object
            object_name (str): name of the object (e.g., "plate")

        Returns:
            float: estimated Z in meters
        """
        if object_name not in self.object_dimensions:
            raise ValueError(f"Unknown object dimensions for: {object_name}")

        known_width = self.object_dimensions[object_name]

        y_indices, x_indices = np.where(mask > 0)
        if len(x_indices) == 0:
            raise ValueError("Mask is empty.")

        pixel_width = x_indices.max() - x_indices.min()
        if pixel_width == 0:
            raise ValueError("Degenerate mask width.")

        Z = (self.fx * known_width) / pixel_width
        return Z
    
    def project_centroid_to_3d(self, 
                               cx_px: int, 
                               cy_px: int, 
                               Z: float) -> np.ndarray:
        """
        Project 2D pixel centroid + estimated depth into 3D camera-space coordinates.
        Returns [X, Y, Z] in meters.
        """
        X = (cx_px - self.cx) * Z / self.fx
        Y = (cy_px - self.cy) * Z / self.fy
        return np.array([X, Y, Z])



