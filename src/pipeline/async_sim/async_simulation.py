import asyncio
import time
import mujoco
from mujoco import viewer
import numpy as np
from utils.utilities import get_resolved_path
from log import setup_logger 
import os
from pathlib import Path
import torch
import numpy as np
from log.setup_logger import setup_logger
from perception.capture import camera_capture
from perception.classification_segmentation.segmentation_image import SAMSegmentation
from planning.planner_llm import PlannerLLM
from action.mujoco_executor  import MuJoCoExecutor
from pipeline.vision_frame import VisionFrame
from pipeline import cache_helper
import json
import matplotlib.pyplot as plt
from perception.classification_segmentation.owl_vit import OwlVitDetector
from PIL import Image, ImageOps
from scipy.spatial.transform import Rotation as R
from scipy.optimize import minimize


CAMERA_NAME = "teleoperator_pov"

class MujocoRealtimeExecutor:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.queue = asyncio.Queue()
        self.joint_names = []
        for joint_id in range(self.model.njnt):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            self.joint_names.append(name)
        self.last_action_state = None
        self.left_arm_joints = [name for name in self.joint_names if name.startswith("left/")]
        self.right_arm_joints = [name for name in self.joint_names if name.startswith("right/")]
        self.logger = setup_logger("MujocoRealtimeExecutor")

    def enqueue_action(self, action):
        """Enqueue a single control/qpos action (list or np.array)."""
        return self.queue.put_nowait(action)

    def enqueue_plan(self, plan):
        """Enqueue a full plan (list of actions)."""
        for action in plan:
            self.enqueue_action(action)

    def capture_image(self):
        self.logger.info("Step 1: Capturing image")
        capture = camera_capture.CameraCapture(model_path=self.model_path)
        image = capture.capture_image(CAMERA_NAME)
        image = np.array(image).astype(np.float32) / 255.0
        image = Image.fromarray((image * 255).astype(np.uint8)).convert("RGB")
        image = ImageOps.expand(image, border=5, fill="white")
        image = image.resize((495, 374))
        # plt.imshow(image)
        # plt.show()
        self.logger.info(f"Image captured, {image.size}")
        return image

    def build_sam_segmentation(self):
        SAM_CHECKPOINT = "/Users/pbanavara/Downloads/sam_vit_b_01ec64.pth"
        MODEL_TYPE = "vit_b"

        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        self.logger.info(f"Using device: {device}")
        segmentation = SAMSegmentation(SAM_CHECKPOINT, MODEL_TYPE, device)
        return segmentation

    async def segment_and_retrieve_depth(self):
        self.logger.info("Getting bounding boxes")
        input("Press Enter to detect objects...")
        detector = OwlVitDetector()
        prompts = [
            "a white ceramic plate",
            "a kitchen dish",
            "a plate inside a sink",
            "a shallow white bowl",
            "a porcelain plate with a rim",
        ]
        image = self.capture_image()
        image, results = detector.detect_objects(image, prompts)
        self.logger.info(f"Detected objects: {results} for prompts: {prompts}")
        box_tensor = results["boxes"][0]  # tensor([x0, y0, x1, y1])
        box_np = box_tensor.cpu().numpy().astype(np.float32).reshape(1, 4)
        # Add bounding box to image and show the same
        self.logger.info(f"Bounding box: {box_np}")
        image_np = np.array(image)
        self.logger.info("Step 2: Classifying and segmenting image")
        input("Press Enter to segment image...")
        segmentation = self.build_sam_segmentation()
        masks, scores = segmentation.predict(image_np, box_np=box_np)

        objects_with_contours = segmentation.classify_masks(
            masks, image_np, self.get_text_prompts()
        )
        self.logger.info("Image segmented")

        input("Press Enter to process segmented objects...")
        for object in objects_with_contours:
            self.logger.info(f"Object: {object}")
            cx_px, cy_py = object["center"]
            if not os.path.exists(self.model_path):
                self.logger.info(f"Model file does not exist: {self.model_path}")
                break
            mapped_object_name = self.map_model_detections(object["name"])
            frame = VisionFrame(
                str(self.model_path), CAMERA_NAME, image, (640, 480), mapped_object_name
            )
            Z = frame.estimate_depth_from_mask(object["mask"], mapped_object_name)

            self.logger.info(f"Estimated depth of the object : {object}: {Z}")
            self.logger.info(f"Centroid: {cx_px}, {cy_py}")
            world_coords = frame.project_pixel_to_world(cx_px, cy_py, Z)
            self.logger.info(f"World coordinates: {world_coords}")
            perception_output = [
                {"name": mapped_object_name, "labels": ["plate"]},
            ]
            known_positions = {mapped_object_name: world_coords}
            return perception_output, known_positions


    async def planner_task(self):
        """Async planner that generates a plan."""
        input("Press Enter to generate plan...")
        task = input("Enter the dishwashing task:: ")
        self.logger.info("Step 3: Generating plan")
        perception_output, known_positions = self.segment_and_retrieve_depth()
        plan_json_path = str(get_resolved_path("../../plan.json"))
        if not os.path.exists(plan_json_path):
            self.logger.info(f"Plan file does not exist: {plan_json_path}")
            aloha_yaml_path = str(get_resolved_path("../planning/aloha.yaml"))
            if not os.path.exists(aloha_yaml_path):
                self.logger.info(f"aloha yaml file does not exist: {aloha_yaml_path}")
                break
            self.logger.info(f"ALOHA YAML path: {aloha_yaml_path}")
            planner = PlannerLLM(robot_yaml_path=aloha_yaml_path)
            plan = planner.build_action_plan(task, perception_output, known_positions)
            plan = json.loads(plan)
            self.logger.info(f"Generated plan: {plan}")
        else:
            with open(plan_json_path, "r") as f:
                plan = json.load(f)
        self.enqueue_plan(plan)

    def get_text_prompts(self):
        TEXT_PROMPTS = [
            "a transparent cylindrical 8 oz drinking glass",
            "a transparent cylindrical 12 oz drinking glass",
            "a transparent conical 12 oz glass",
            "a plate",
            "a transparent 16 oz cylindrical glass",
            "a ceramic coffee mug with a handle",
            "a stainless steel saucepan",
            "a pressure cooker with black handles",
            "a kitchen sink",
        ]
        return TEXT_PROMPTS

    def map_model_detections(self, detected_object_name: str):
        detection_to_model_name = {
            "a plate": "plate_geom",
            "a bowl": "bowl_geom",
            "a cup": "cup_geom",
            "a fork": "fork_geom",
            # Add more as needed
        }
        if detected_object_name in detection_to_model_name:
            return detection_to_model_name[detected_object_name]
        else:
            raise  ValueError(f"No mapping for detected object: {detected_object_name}")

    def move_through_trajectory(self, gripper, arm_joints, trajectory, viewer):

        joint_ids = [mujoco.mj_name2id(self.model, 
                                       mujoco.mjtObj.mjOBJ_JOINT, name)
                    for name in arm_joints]
        if self.last_action_state is not None:
            self.logger.info("Overriding sim with the last action state")
            for i, joint_name in enumerate(arm_joints):
                joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                qpos_index = self.model.jnt_qposadr[joint_id]
                self.data.qpos[qpos_index] = self.last_action_state[i]
            # Setting the velocisity and acceleration to zero explicitly
            self.data.qvel[:] = 0
            self.data.qacc[:] = 0
            mujoco.mj_forward(self.model, self.data)

        q_guess = np.array([
            self.data.qpos[self.model.jnt_qposadr[jid]]
            for jid in joint_ids
            ], dtype=float)
        self.logger.info(f"Solving IK for initial state: {q_guess} ")
        # Temporary hack to get the arm to move to the plate
        plate_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "plate")
        plate_pos = self.data.site_xpos[plate_site_id].copy()
        print("🍽 Plate position:", plate_pos)
        for step in trajectory:
            pos = np.array(step["position"])
            #rot = np.array(step["rotation"])
            rot = None 
            self.logger.info(f"Updated persistent joint state: {self.last_action_state}")
            # Solve from current joint state
            self.logger.info(f"Solving IK for remaining state: {q_guess} ")
            current_sim_qpos = np.array([
                self.data.qpos[self.model.jnt_qposadr[jid]]
                    for jid in joint_ids
                ])
            self.logger.info(f"Sim qpos before solve: {current_sim_qpos} and last action state {self.last_action_state}")
            mujoco.mj_forward(self.model, self.data)
            q_solution = self.solve_ik(gripper, 
                                       arm_joints, pos, rot, q_init=q_guess)

            # Apply solution to joints
            for i, joint_name in enumerate(arm_joints):
                joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                qpos_index = self.model.jnt_qposadr[joint_id]
                self.data.qpos[qpos_index] = q_solution[i]
                actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, joint_name)
                self.data.ctrl[actuator_id] = q_solution[i]

            print("Actuator ctrl before stepping:", self.data.ctrl[:])
            mujoco.mj_forward(self.model, self.data)

            # Simulate to see movement
            for _ in range(100):  # could be based on trajectory delta
                mujoco.mj_step(self.model, self.data)
                viewer.sync()
                time.sleep(0.01)
            
            for alpha in np.linspace(0, 1, 20):
                ctrl_interp = (1 - alpha) * q_guess + alpha * q_solution
                for i, joint_name in enumerate(arm_joints):
                    actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, joint_name)
                    self.data.ctrl[actuator_id] = ctrl_interp[i]

                mujoco.mj_step(self.model, self.data)
                viewer.sync()
                time.sleep(0.01)
            #Persist joint state
            q_guess = q_solution.copy()
            self.last_action_state = q_solution.copy()
            viewer.sync()

    def solve_ik(self, gripper_site, 
                 gripper_joints, 
                 target_pos, 
                 target_rot,
                 q_init=None):
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
        # TODO: REmove this target_rot = None. This is a hack until we can figure out why rotations cause the arm not to move.
        self.logger.info(f"Solving IK for target pos: {target_pos} and rot {target_rot}")   
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

        bounds = [
            tuple(self.model.jnt_range[self.model.joint(name).id])
            for name in gripper_joints
        ]
        result = minimize(cost_fn, q_init, method="L-BFGS-B", bounds=bounds)
        if not result.success:
            raise RuntimeError(f"IK solver failed: {result.message}")
        return result.x

    async def run_simulation(self, step_hz: float = 100.0):
        """Main sim loop: dequeues actions and steps MuJoCo."""
        viewer_passive = viewer.launch_passive(self.model, self.data)
        step_delay = 1.0 / step_hz

        while True:
            if not self.queue.empty():
                action = await self.queue.get()
                arm = action.get("arm", "left")
                if action["trajectory"]:
                    if action["action"] == "move_to_pose": 
                        if arm == "right": #Robot specific definition
                            qpos_snapshot = np.array([
                                        self.data.qpos[self.model.jnt_qposadr[
                                                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                                                ]] for name in self.right_arm_joints
                                            ])
                            print(f"[🚨] QPOS before new action: {qpos_snapshot}")
                            self.move_through_trajectory("right/gripper", 
                                             self.right_arm_joints,
                                             action["trajectory"], viewer_passive)
                        else:
                            qpos_snapshot = np.array([
                                        self.data.qpos[self.model.jnt_qposadr[
                                                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                                                ]] for name in self.left_arm_joints
                                            ])
                            print(f"[🚨] QPOS before new action: {qpos_snapshot}")
                            self.move_through_trajectory("left/gripper", 
                                             self.left_arm_joints,
                                             action["trajectory"], viewer_passive)
                else:
                    self.logger.info("No trajectory — skipping")
                self.logger.info(f"Completed Action: {action}")
                # Or update qpos and forward if that's your control mode:
                # self.data.qpos[:len(action)] = action
                # mujoco.mj_forward(self.model, self.data)
                mujoco.mj_step(self.model, self.data)
            await asyncio.sleep(step_delay)

    async def start(self):
        """Starts both planner and sim concurrently."""
        try:
            await asyncio.gather(self.planner_task(), self.run_simulation())
        except asyncio.CancelledError:
            self.logger.info("Shutting down cleanly...")

if __name__ == "__main__":
    mujoco_model_path = str(get_resolved_path("../simulated_sink/aloha/aloha.xml"))
    executor = MujocoRealtimeExecutor(mujoco_model_path)
    asyncio.run(executor.start())