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

logger = setup_logger()
CAMERA_NAME = "teleoperator_pov"

def build_sam_segmentation():
    SAM_CHECKPOINT = "/Users/pbanavara/Downloads/sam_vit_b_01ec64.pth"
    MODEL_TYPE = "vit_b"

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    logger.info(f"Using device: {device}")
    segmentation = SAMSegmentation(SAM_CHECKPOINT, MODEL_TYPE, device)
    return segmentation

def get_text_prompts():
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

def map_model_detections(detected_object_name: str):
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

def get_resolved_path(given_path: Path) -> str:
    # Get path to current file's directory
    current_dir = Path(__file__).parent
    # Resolve relative path from current file
    model_path = (current_dir / given_path).resolve()
    return model_path


def run_pipeline():
    mujoco_model_path = get_resolved_path("../simulated_sink/aloha/aloha.xml")
    executor = MuJoCoExecutor(str(mujoco_model_path))
    if os.path.exists(mujoco_model_path):
        logger.info(f"MuJoCo model path exists: {mujoco_model_path}")
    else:
        logger.error(f"MuJoCo model path does not exist: {mujoco_model_path}")
        exit(1)
    logger.info("Starting pipeline run")
    input("Press Enter to start the pipeline...")
    executor.view_model()

    logger.info("Step 1: Capturing image")
    capture = camera_capture.CameraCapture(model_path=str(mujoco_model_path))
    image = capture.capture_image(CAMERA_NAME)
    image = np.array(image).astype(np.float32) / 255.0
    image = Image.fromarray((image * 255).astype(np.uint8)).convert("RGB")
    image = ImageOps.expand(image, border=5, fill='white')
    image = image.resize((495, 374))
    #plt.imshow(image)
    #plt.show()
    logger.info(f"Image captured, {image.size}")

    logger.info("Getting bounding boxes")
    input("Press Enter to detect objects...")
    detector = OwlVitDetector()
    prompts = [
        "a white ceramic plate", "a kitchen dish", "a plate inside a sink", "a shallow white bowl", "a porcelain plate with a rim"
    ]
    image, results = detector.detect_objects(
        image,prompts
    )
    logger.info(f"Detected objects: {results} for prompts: {prompts}")
    box_tensor = results["boxes"][0]  # tensor([x0, y0, x1, y1])
    box_np = box_tensor.cpu().numpy().astype(np.float32).reshape(1, 4)
    # Add bounding box to image and show the same
    logger.info(f"Bounding box: {box_np}")
    image_np = np.array(image)
    logger.info("Step 2: Classifying and segmenting image")
    input("Press Enter to segment image...")
    segmentation = build_sam_segmentation()
    masks, scores = segmentation.predict(image_np, box_np=box_np)

    objects_with_contours = segmentation.classify_masks(masks,
                                                        image_np,
                                                        get_text_prompts())
    logger.info("Image segmented")

    input("Press Enter to process segmented objects...")
    for object in objects_with_contours:
        logger.info(f"Object: {object}")
        cx_px, cy_py = object["center"]
        if not os.path.exists(mujoco_model_path):
            logger.info(f"Model file does not exist: {mujoco_model_path}")
            break
        mapped_object_name = map_model_detections(object["name"])
        frame = VisionFrame(str(mujoco_model_path),
                            CAMERA_NAME, image, (640, 480), mapped_object_name)
        Z = frame.estimate_depth_from_mask(object["mask"], mapped_object_name)

        logger.info(f"Estimated depth of the object : {object}: {Z}")
        logger.info(f"Centroid: {cx_px}, {cy_py}")
        world_coords = frame.project_pixel_to_world(cx_px, cy_py, Z)
        logger.info(f"World coordinates: {world_coords}")
        input("Press Enter to generate plan...")
        logger.info("Step 3: Generating plan")
        task = "Move the left arm to grab the plate in the sink"
        perception_output = [
            {"name": mapped_object_name, "labels": ["plate"]},
        ]
        known_positions = {
            mapped_object_name: world_coords
        }

        plan_json_path = str(get_resolved_path("../../plan.json"))
        if not os.path.exists(plan_json_path):
            logger.info(f"Plan file does not exist: {plan_json_path}")
            aloha_yaml_path = str(get_resolved_path("../planning/aloha.yaml"))
            if not os.path.exists(aloha_yaml_path):
                logger.info(f"aloha yaml file does not exist: {aloha_yaml_path}")
                break
            logger.info(f"ALOHA YAML path: {aloha_yaml_path}")
            planner = PlannerLLM(robot_yaml_path=aloha_yaml_path)
            plan = planner.build_action_plan(task,
                                          perception_output, known_positions)
            plan = json.loads(plan)
            logger.info(f"Generated plan: {plan}")
            planner.save_plan(json.dumps(plan), "plan.json")
        else:
            with open(plan_json_path, "r") as f:
                plan = json.load(f)
        logger.info("Step 4: Executing actions")
        input("Press Enter to execute actions...")
        executor.run(plan=plan)

    logger.info("Pipeline completed successfully")
