import mujoco
import mujoco.viewer
import numpy as np
import time
from scipy.optimize import minimize
import argparse
from scipy.spatial.transform import Rotation as R
import torch
from log.setup_logger import setup_logger

def rotation_error(rot_vec, target_vec):
    R1 = R.from_euler("xyz", rot_vec).as_matrix()
    R2 = R.from_euler("xyz", target_vec).as_matrix()

    R_err = R1.T @ R2
    angle = np.arccos(np.clip((np.trace(R_err) - 1) / 2, -1.0, 1.0))
    return angle


class MuJoCoExecutor:
    def __init__(self, model_path, action_json):
        self.logger = setup_logger("MuJoCoExecutor")
        print(f"Loading model from: {model_path}")
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.joint_names = []
        for joint_id in range(self.model.njnt):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            self.joint_names.append(name)

        # --- Joint groupings ---
        self.left_arm_joints = [name for name in self.joint_names if name.startswith("left/")]
        self.right_arm_joints = [name for name in self.joint_names if name.startswith("right/")]

        self.left_joint_ids = [self.model.joint(name).qposadr 
                               for name in self.left_arm_joints]
        self.eft_joint_ids = [int(i) for i in self.left_joint_ids]
        self.right_joint_ids = [self.model.joint(name).qposadr 
                                for name in self.right_arm_joints]
        self.right_joint_ids = [int(i) for i in self.right_joint_ids]

        self.plan = action_json
    
    def interpolate_trajectory(self, traj, num_points=20):
        smooth_traj = []
        for i in range(len(traj) - 1):
            p1, r1 = np.array(traj[i]["position"]), np.array(traj[i]["rotation"])
            p2, r2 = np.array(traj[i + 1]["position"]), np.array(traj[i + 1]["rotation"])

            for alpha in np.linspace(0, 1, num_points):
                pos = (1 - alpha) * p1 + alpha * p2
                rot = (1 - alpha) * r1 + alpha * r2
                smooth_traj.append({"position": pos.tolist(), "rotation": rot.tolist()})

        return smooth_traj


    def move_gripper_to(self, gripper: str, 
                        arm_joints: list, 
                        trajectory: list,
                        viewer):
        trajectory = self.interpolate_trajectory(trajectory, num_points=50)
        self.logger.info(f"left_arm_joints = {self.left_arm_joints}")

        for step in trajectory:
            pos = np.array(step["position"])
            rot = np.array(step["rotation"])
            # Your IK solver here
            original_qpos = self.data.qpos.copy()

            result = self.solve_ik(gripper, arm_joints, pos, rot)
            #self.data.qpos[: len(qpos)] = qpos
            for i, joint_name in enumerate(arm_joints):
                joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                qpos_index = self.model.jnt_qposadr[joint_id]
                self.data.qpos[qpos_index] = result[i]
            # 👇 Freeze the rest (unchanged)
            for j in range(len(self.data.qpos)):
                if j not in [self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)] for name in arm_joints]:
                    self.data.qpos[j] = original_qpos[j]
            mujoco.mj_forward(self.model, self.data)
        
        for _ in range(100):  # or 200 for slower, smoother movement
            mujoco.mj_step(self.model, self.data)
            viewer.sync()
            time.sleep(0.01)


    def solve_ik(self, gripper_site, gripper_joints, target_pos, target_rot):
        """
        Inverse kinematics solver using L-BFGS-B to minimize position and optional rotation error.

        Args:
            gripper_site (str): Name of the end-effector site in the MuJoCo model.
            gripper_joints (list): List of joint names controlling the arm.
            target_pos (np.ndarray): Desired XYZ position in world coordinates.
            target_rot (np.ndarray or None): Desired Euler XYZ rotation in radians (or None for position-only IK).

        Returns:
            np.ndarray: Optimal joint angles for the given target.
        """
        # TODO: REmove this target_rot = None
        target_rot = None
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, gripper_site)
        joint_ids = [int(self.model.joint(name).qposadr) for name in gripper_joints]
        def fk(qpos):
            # Create an isolated copy of data
            temp_data = mujoco.MjData(self.model)
            temp_data.qpos[:] = self.data.qpos[:]  # start with current sim state

            for i, j in enumerate(joint_ids):
                temp_data.qpos[j] = qpos[i]

            mujoco.mj_forward(self.model, temp_data)

            pos = temp_data.site_xpos[site_id].copy()
            rot_euler = None
            if target_rot is not None:
                rot_mat = temp_data.site_xmat[site_id].reshape(3, 3).copy()
                rot_euler = R.from_matrix(rot_mat).as_euler("xyz")
            return pos, rot_euler

        def rotation_error(rot_vec, target_vec):
            R1 = R.from_euler('xyz', rot_vec).as_matrix()
            R2 = R.from_euler('xyz', target_vec).as_matrix()
            R_err = R1.T @ R2
            angle = np.arccos(np.clip((np.trace(R_err) - 1) / 2, -1.0, 1.0))
            return angle

        def cost_fn(qpos):
            pos, rot = fk(qpos)
            pos_error = np.linalg.norm(pos - target_pos)
            if target_rot is not None and rot is not None:
                rot_error = rotation_error(rot, target_rot)
                return float(pos_error + 0.5 * rot_error)
            else:
                return float(pos_error)

        q_init = np.array([self.data.qpos[j] for j in joint_ids], 
                          dtype=float).flatten()
        bounds = [
            tuple(self.model.jnt_range[self.model.joint(name).id])
            for name in gripper_joints
        ]
        self.logger.info(f"IK target position: {target_pos}")
        self.logger.info(f"IK target rotation: {target_rot}")
        self.logger.info(f"Initial guess (q_init): {q_init}")
        self.logger.info(f"Joint bounds: {bounds}")
        result = minimize(cost_fn, q_init, method="L-BFGS-B", bounds=bounds)

        if not result.success:
            raise RuntimeError(f"IK solver failed: {result.message}")
        return result.x


    def project_to_point_cloud(
        self, contour: np.ndarray, Z: float, fx: float, fy: float, cx: float, cy: float
        ):
        u = contour[:, 0]
        v = contour[:, 1]
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy
        Z_arr = np.full_like(X, Z)
        return np.stack([X, Y, Z_arr], axis=1)
    
    def get_intrinsics_from_mujoco(self, cam_id, width, height):
        """
        Computes fx, fy, cx, cy from MuJoCo model and camera info.

        Args:
            model: MjModel
            cam_id: int (index of camera)
            width, height: resolution of rendered image in pixels

        Returns:
            fx, fy, cx, cy: float values for intrinsics
        """
        fovy_deg = self.model.cam_fovy[cam_id]
        fovy_rad = np.deg2rad(fovy_deg)

        fy = 0.5 * height / np.tan(fovy_rad / 2)
        fx = fy  # Assume square pixels (no aspect ratio distortion)

        cx = width / 2
        cy = height / 2

        return fx, fy, cx, cy
    
    def extract_markers_from_plan(self):
        """Extracts XYZ positions from the plan into a marker list."""
        self.markers = []
        for action in self.plan:
            for step in action.get("trajectory", []):
                pos = step["position"]
                self.markers.append(np.array(pos, dtype=float))

    def render_callback(self, viewer):
        """Custom callback that adds markers to the MuJoCo viewer."""
        viewer.user_overlay.clear()
        self.logger.info(f"Markers in render: {self.markers}")
        for i, pos in enumerate(self.markers):
            viewer.add_marker(
                pos=pos,
                size=[0.1, 0.1, 0.1],     # small sphere
                rgba=[1, 0, 0, 1],           # red color
                label=f"waypoint {i}",
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
            )

    def run(self):
        self.extract_markers_from_plan()
        self.logger.info(f"Markers outside render: {self.markers}")
        with mujoco.viewer.launch_passive(
                self.model, self.data 
        ) as viewer:
            viewer.user_render_callback = self.render_callback

            for action in self.plan:
                arm = action.get("arm", "left")
                if action["trajectory"]:
                    if action["action"] == "move_to_pose": #Robot specific definition
                        if arm == "right": #Robot specific definition
                            self.move_gripper_to("right/gripper", 
                                             self.right_arm_joints,
                                             action["trajectory"], viewer)
                        else:
                            self.move_gripper_to("left/gripper", 
                                             self.left_arm_joints,
                                             action["trajectory"], viewer)
                    
                else:
                    self.logger.info("No trajectory — skipping")
            while viewer.is_running():
                viewer.sync()
