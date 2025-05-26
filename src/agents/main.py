import threading
import asyncio
import time
import mujoco
from mujoco import viewer
from utils.utilities import get_resolved_path
from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt
import torch
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
import cv2
from multiprocessing import Queue, Process

# ---------- Shared Context for Viewer ----------
shared_viewer = {}
viewer_ready = threading.Event()
CAMERA_NAME = "teleoperator_pov"

# ---------- Launch Viewer in Thread ----------
def launch_viewer(model, data):
    v = viewer.launch_passive(model, data)
    shared_viewer["instance"] = v
    viewer_ready.set()  # Signal viewer is ready

class PerceptionAgent:
    def __init__(self, model, data, model_path):
        self.model = model
        self.data = data
        self.viewer = None
        self.model_path = model_path
        self.logger = setup_logger("PerceptionAgent")
        self.image_queue = Queue()
        self.p = Process(target=self.image_display_worker, args=(self.image_queue,))
        self.p.start()

    def image_display_worker(self, q):
        while True:
            img = q.get()
            if img is None:
                break
            cv2.imshow("Debug", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()

    async def capture_image(self):
        self.logger.info("Step 1: Capturing image")

        capture = camera_capture.CameraCapture(model_path=self.model_path)
        image = capture.capture_image(CAMERA_NAME)
        image = np.array(image).astype(np.float32) / 255.0
        self.image_queue.put(image)
        image = Image.fromarray((image * 255).astype(np.uint8)).convert("RGB")
        image = ImageOps.expand(image, border=5, fill="white")
        image = image.resize((495, 374))

        self.logger.info(f"Image captured, {image.size}")
        return image
    
    async def build_sam_segmentation(self):
        """A factory like method for instantiating segmentation module

        Returns:
            SAMSegmentation : Instance of SAMSegmentation class
        """
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
        """A function to retrieve depth from the segmentation mask and object priors.

        Returns:
            Tuple: depth, position
        """
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

    async def run(self):
        """Main Async method to run the agent
        """
        print("Perception agent waiting for viewer")
        while not viewer_ready.is_set():
            await asyncio.sleep(0.1)
        self.viewer = shared_viewer["instance"]
        await self.capture_image()
        self.viewer.sync()


# ---------- Dummy PlannerAgent ----------
class PlannerAgent:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.viewer = None

    async def run(self):
        # Wait for the viewer to be ready
        print("[Planner] Waiting for viewer...")
        while not viewer_ready.is_set():
            await asyncio.sleep(0.1)

        self.viewer = shared_viewer["instance"]
        print("[Planner] Viewer ready, starting loop.")

        # Example loop — just call viewer.sync() repeatedly
        while True:
            # (In real case, modify qpos/ctrl, plan actions, etc.)
            self.viewer.sync()
            await asyncio.sleep(0.1)


# ---------- Main Async Entry ----------
async def main():
    mujoco_model_path = str(get_resolved_path("../simulated_sink/aloha/aloha.xml"))
    model = mujoco.MjModel.from_xml_path(mujoco_model_path)
    data = mujoco.MjData(model)

    # Start the viewer in a background thread
    threading.Thread(target=launch_viewer, args=(model, data), daemon=True).start()

    # Start async agents
    perception = PerceptionAgent(model, data, mujoco_model_path)
    await asyncio.gather(
        perception.run()
        # You can add more agents here
    )


# ---------- Run it ----------
if __name__ == "__main__":
    asyncio.run(main())
