import asyncio
import time
import mujoco
from mujoco import viewer
import numpy as np
from utils.utilities import get_resolved_path, get_text_prompts
from utils.image_utils import save_bbox_overlay 
from log import setup_logger 
import os
from pathlib import Path
import torch
import numpy as np
from log.setup_logger import setup_logger
from perception.capture import camera_capture
from perception.classification_segmentation.segmentation_image import SAMSegmentation
from planning.planner_llm import PlannerLLM
from planning.llama_planner import LlamaPlanner
from pipeline.vision_frame import VisionFrame
import json
from perception.classification_segmentation.owl_vit import OwlVitDetector
from PIL import Image, ImageOps
from scipy.spatial.transform import Rotation as R
from scipy.optimize import minimize
import time


CAMERA_NAME = "teleoperator_pov"

class MujocoRealtimeExecutor:
    def __init__(self, model_path: str):
        start = time.time()
        self.logger = setup_logger("MujocoRealtimeExecutor")
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
        self.text_prompts = get_text_prompts()
        end = time.time()
        self.logger.info(f"MujocoRealtimeExecutor initialization completed in {end - start:.3f}s")

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
        self.logger.info("Capturing image and obtaining bounding boxes")
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
        box_list = box_np[0].tolist()
        # Add bounding box to image and show the same
        self.logger.info(f"Bounding box: {box_np}")
        image_np = np.array(image)
        save_bbox_overlay(image_np, box_list, "/tmp/output.jpg")
        self.logger.info("Step 2: Classifying and segmenting image")
        input("Press Enter to segment image...")
        segmentation = self.build_sam_segmentation()
        masks, scores = segmentation.predict(image_np, box_np=box_np)

        objects_with_contours = segmentation.classify_masks(
            masks, image_np, self.text_prompts
        )
        self.logger.info("Image segmented")

        input("Press Enter to process segmented objects...")
        for object in objects_with_contours:
            cx_px, cy_py = object["center"]
            if not os.path.exists(self.model_path):
                self.logger.info(f"Model file does not exist: {self.model_path}")
                break
            mapped_object_name = self.map_model_detections(object["name"])
            frame = VisionFrame(
                str(self.model_path), CAMERA_NAME, image, (640, 480), mapped_object_name
            )
            Z = frame.estimate_depth_from_mask(object["mask"], mapped_object_name)
            self.logger.info(f"Estimated depth of the object : {Z}")
            self.logger.info(f"Centroid: {cx_px}, {cy_py}")
            world_coords = frame.project_pixel_to_world(cx_px, cy_py, Z)
            self.logger.info(f"World coordinates: {world_coords}")
            plate_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "plate")
            sink_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "sink")
            self.logger.info(f"Sink world position:, {self.data.xpos[sink_id]}")
            self.logger.info(f"Plate center (data.xpos):, {self.data.xpos[plate_body_id]}")
            
            perception_output = [
                {"name": mapped_object_name, "labels": [object["name"]]},
            ]
            known_positions = {mapped_object_name: world_coords}
            return perception_output, known_positions


    async def planner_task(self):
        """Async planner that generates a plan."""
        plan_json_path = str(get_resolved_path("../../plans/plan.json"))
        response = input("Continue with the last generated plan y/n or q to quit: ").strip().lower()
        if response == "q":
            exit(0)
        elif response == "y": 
            with open(plan_json_path, "r") as f:
                plan = json.load(f) 
        else:
            self.logger.info(f"Creating new plan for task")
            task = input("Enter the dishwashing task:: ")
            perception_output, known_positions = await self.segment_and_retrieve_depth()
            
            # Optimized LlamaPlanner initialization with performance logging
            self.logger.info("🚀 Initializing LlamaPlanner (optimized)...")
            init_start_time = time.time()
            
            llama_yaml_path = str(get_resolved_path("../planning/llama.yaml"))
            planner = LlamaPlanner(
                robot_yaml_path=llama_yaml_path,
                model="llama-4-maverick-17b-128e-instruct-fp8",
                enable_caching=True,
                cache_size=1000,
                max_workers=4,
                timeout=30.0,
                max_retries=3
            )
            
            init_time = time.time() - init_start_time
            self.logger.info(f"✅ LlamaPlanner initialized in {init_time:.3f}s (optimized)")
            
            # Generate plan with performance logging
            self.logger.info("🤖 Generating action plan...")
            plan_start_time = time.time()
            
            plan = planner.build_action_plan(task, perception_output, known_positions)
            
            plan_time = time.time() - plan_start_time
            self.logger.info(f"✅ Plan generated in {plan_time:.3f}s")
            
            # Save plan and get performance stats
            planner.save_plan(plan, plan_json_path)
            stats = planner.get_performance_stats()
            self.logger.info(f"📊 Performance: {stats['total_requests']} requests, "
                           f"avg {stats['avg_response_time']:.3f}s, "
                           f"cache hit rate {stats['cache_hit_rate']:.1%}")
            
            self.logger.info(f"Generated plan: {plan}")
            plan = json.loads(plan)
        self.enqueue_plan(plan)


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
        # Temporary hack to get the arm to move to the plate
        for step in trajectory:
            pos = np.array(step["position"])
            # TODO These hardcoded placeholders are a reminder of the painstaking iteration of getting IK to work 
            #pos = np.array([-0.14999904, - 0.2285476, - 0.3785524]) 
            #pos = np.array([-1.2363652, -0.27473684, 0.96247765])
            #pos = np.array([-0.49454608, -0.16989474, -0.86499106])
            self.logger.info(
                f"Current EE pos: {self.data.site_xpos[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, gripper)]}"
                f"Current Plate pos: {self.data.site_xpos[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, 'plate')]}")
            self.logger.info(f"Gripper positioon {pos}")
            #rot = np.array(step["rotation"])
            rot = None
            self.logger.info(f"Updated persistent joint state: {self.last_action_state}")
            # Solve from current joint state
            self.logger.info(f"Solving IK for remaining state: {q_guess} ")
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
            self.logger.info(f"IK solved for position {pos}")
            self.logger.info(
                f"Current EE pos after solve: {self.data.site_xpos[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, gripper)]}"
            )

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
        #result = minimize(cost_fn, q_init, method="L-BFGS-B", bounds=bounds)
        result = minimize(
            cost_fn,
            q_init,
            method="L-BFGS-B",
            bounds=bounds,
            tol=1e-6,
            options={"maxiter": 1000, "disp": True},
        )
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
                last_action_time = time.time()
                mujoco.mj_step(self.model, self.data)
            elif time.time() - last_action_time > 10:
                self.logger.info("No new actions — exiting simulation loop.")
                return
            await asyncio.sleep(step_delay)

    async def start(self):
        """Starts both planner and sim concurrently."""
        try:
             while True:
                self.logger.info("Running planner...")
                await self.planner_task()
                self.logger.info("Running simulation...")
                await self.run_simulation()
        except asyncio.CancelledError:
            self.logger.info("Shutting down cleanly...")
