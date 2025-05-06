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
        self.left_arm_joints = [name for name in self.joint_names if "left" in name]
        self.right_arm_joints = [name for name in self.joint_names if "right" in name]

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


    def move_gripper_to(self, gripper: str, arm_joints: list, trajectory: list):
        #trajectory = self.interpolate_trajectory(trajectory, num_points=50)
        for step in trajectory:
            pos = np.array(step["position"])
            rot = np.array(step["rotation"])
            # Your IK solver here
            qpos = self.solve_ik(gripper, arm_joints, pos, rot)
            self.data.qpos[: len(qpos)] = qpos
            mujoco.mj_forward(self.model, self.data)

            for _ in range(20):  # render multiple frames to simulate motion
                mujoco.mj_step(self.model, self.data)

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
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, gripper_site)
        joint_ids = [self.model.joint(name).qposadr for name in gripper_joints]

        def fk(qpos):
            original_qpos = self.data.qpos.copy()
            for i, j in enumerate(joint_ids):
                self.data.qpos[j] = qpos[i]
            mujoco.mj_forward(self.model, self.data)

            pos = self.data.site_xpos[site_id].copy()
            if target_rot is not None:
                rot_mat = self.data.site_xmat[site_id].reshape(3, 3).copy()
                rot_euler = R.from_matrix(rot_mat).as_euler('xyz')
            else:
                rot_euler = None

            self.data.qpos[:] = original_qpos  # restore original state
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
                self.logger.info(f"Cost function pos_error: {pos_error:.4f}, rot_error: {rot_error:.4f}")
                return float(pos_error + 0.5 * rot_error)
            else:
                self.logger.info(f"Cost function pos_error: {pos_error:.4f} (rotation not used)")
                return float(pos_error)

        q_init = np.array([self.data.qpos[j] for j in joint_ids], 
                          dtype=float).flatten()
        self.logger.info(f"Initial guess: {q_init.shape}")
        bounds = [
            tuple(self.model.jnt_range[self.model.joint(name).id])
            for name in gripper_joints
        ]

        result = minimize(cost_fn, q_init, method="L-BFGS-B", bounds=bounds)

        if not result.success:
            raise RuntimeError(f"IK solver failed: {result.message}")
        self.logger.info(f"IK solver succeeded: {result.message}")
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

    def run(self):
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            for action in self.plan:
                arm = action.get("arm", "right")
                if action["trajectory"]:
                    if action["action"] == "move_to_pose": #Robot specific definition
                        if arm == "right": #Robot specific definition
                            self.move_gripper_to("right_gripper_site", 
                                             self.right_arm_joints,
                                             action["trajectory"])
                        else:
                            self.move_gripper_to("left_gripper_site", 
                                             self.left_arm_joints,
                                             action["trajectory"])
                    
                else:
                    self.logger.info("No trajectory — skipping")
                mujoco.mj_step(self.model, self.data)
                viewer.sync()
                # Optional: pause for interaction or review
                time.sleep(0.2)
            while viewer.is_running():
                viewer.sync()