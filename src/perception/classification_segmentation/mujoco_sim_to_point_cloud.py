
import numpy as np
import mujoco
from mujoco import viewer


model = mujoco.MjModel.from_xml_path("/Users/pbanavara/dev/mujoco_menagerie/aloha/aloha.xml")
data = mujoco.MjData(model)

def get_intrinsics_from_mujoco(model, cam_id, width, height):
    """
    Computes fx, fy, cx, cy from MuJoCo model and camera info.

    Args:
        model: MjModel
        cam_id: int (index of camera)
        width, height: resolution of rendered image in pixels

    Returns:
        fx, fy, cx, cy: float values for intrinsics
    """
    fovy_deg = model.cam_fovy[cam_id]
    fovy_rad = np.deg2rad(fovy_deg)

    fy = 0.5 * height / np.tan(fovy_rad / 2)
    fx = fy  # Assume square pixels (no aspect ratio distortion)

    cx = width / 2
    cy = height / 2

    return fx, fy, cx, cy

cam_id = model.camera_name2id("teleoperator_pov")
fx, fy, cx, cy = get_intrinsics_from_mujoco(model, cam_id, 640, 480)