import mujoco
import mujoco.viewer
import numpy as np
import json
import time
from scipy.optimize import minimize
import argparse
from scipy.spatial.transform import Rotation as R


def rotation_error(rot_vec, target_vec):
    R1 = R.from_euler("xyz", rot_vec).as_matrix()
    R2 = R.from_euler("xyz", target_vec).as_matrix()

    R_err = R1.T @ R2
    angle = np.arccos(np.clip((np.trace(R_err) - 1) / 2, -1.0, 1.0))
    return angle


# Globals
class MuJoCoExecutor:
    def __init__(self, model_path, action_file):
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

        with open(action_file, "r") as f:
            self.plan = json.load(f)
    
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
        """A simple inverse kinematics solver which minimizes the error between the current and target position.

        Args:
            model (_type_): _description_
            site_name (_type_): _description_
            data (_type_): _description_
            gripper (_type_): _description_
            target_pos (_type_): _description_
            target_rot (_type_): _description_

        Returns:
            _type_: _description_
        """
        site_id = mujoco.mj_name2id(self.model, 
                                    mujoco.mjtObj.mjOBJ_SITE, gripper_site)
        
        # Get the joint indices for the right arm; assume right_arm_joints is defined externally
        joint_ids = [self.model.joint(name).qposadr for name in gripper_joints]

        def fk(qpos):
            # Save original qpos to avoid side effects in each evaluation
            original_qpos = self.data.qpos.copy()
            # Update the joint positions for our right arm
            for i, j in enumerate(joint_ids):
                self.data.qpos[j] = qpos[i]
            mujoco.mj_forward(self.model, self.data)
            # Copy the computed end-effector position (to avoid later changes)
            # Restore the original qpos
            self.data.qpos[:] = original_qpos
            pos = self.data.site_xpos[site_id].copy()
            rot = (
                self.data.site_xmat[site_id].reshape(3, 3).copy()
                if target_rot is not None
                else None
            )
            return pos, rot

        def rotation_error(rot_vec, target_vec):
            R1 = R.from_euler('xyz', rot_vec).as_matrix()
            R2 = R.from_euler('xyz', target_vec).as_matrix()
            R_err = R1.T @ R2  # relative rotation
            # Compute angle error (there are various equivalent formulations)
            angle = np.arccos(np.clip((np.trace(R_err) - 1) / 2, -1.0, 1.0))
            return angle

        def cost_fn(qpos):
            pos, rot = fk(qpos)
            pos_error = np.linalg.norm(pos - target_pos)
            if target_rot is not None:
                rot_error = rotation_error(rot, target_rot)
                print((f"pos_error: {pos_error}, rot_error: {rot_error}"))
                return float(pos_error + 0.5 * np.sum(rot_error)/len(rot_error))

            return float(pos_error)
            

        # Prepare initial guess as a flat 1D array of floats
        q_init = np.array([float(self.data.qpos[j]) for j in joint_ids]).flatten()
        bounds = []
        for i, j in enumerate(joint_ids):
            bounds.append(
                (
                    self.model.jnt_range[self.model.joint(gripper_joints[i]).id, 0],
                    self.model.jnt_range[self.model.joint(gripper_joints[i]).id, 1],
                )
            )
        result = minimize(cost_fn, q_init, method="L-BFGS-B", bounds=bounds)
        return result.x  # optimal joint angles

    def run(self):
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            for action in self.plan:
                print(f">>> {action['action']} with {action['gripper']}")
                if action["trajectory"]:
                    if action["gripper"] == "right": #Robot specific definition
                        self.move_gripper_to("right_gripper_site", 
                                             self.right_arm_joints,
                                             action["trajectory"])
                    elif action["gripper"] == "left": #Robot specific definition
                        self.move_gripper_to("left_gripper_site", 
                                             self.left_arm_joints,
                                             action["trajectory"])
                    
                else:
                    print("No trajectory — skipping")
                mujoco.mj_step(self.model, self.data)
                viewer.sync()
                # Optional: pause for interaction or review
                time.sleep(0.5)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--action_file", type=str, required=True)
    parser.add_argument("--model_file", type=str, required=True)
    args = parser.parse_args()
    executor = MuJoCoExecutor(args.model_file, args.action_file)
    executor.run()
