import torch
import clip
import numpy as np
import cv2
from PIL import Image
from segment_anything import sam_model_registry, SamPredictor
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from log.setup_logger import setup_logger


class SAMSegmentation:
    def __init__(self, sam_checkpoint, model_type, device):
        self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        self.sam.to(device=device)
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=device)
        self.device = device
        self.logger = setup_logger("SAMSegmentation")

    def predict(self, image_rgb, box_np = None):
        predictor = SamPredictor(self.sam)
        predictor.set_image(image_rgb)

        if box_np is not None:
            input_boxes = box_np  # must be (N, 4), float32
            input_boxes = input_boxes.astype(np.float32)

            masks, scores, _ = predictor.predict(
                box=input_boxes,
                multimask_output=False
            )
        else:
        # fallback to default grid-based point sampling
            height, width = image_rgb.shape[:2]
            step = 150
            grid_points = [
                [x, y] for y in range(0, height, step) for x in range(0, width, step)
            ]
            input_points = np.array(grid_points)
            input_labels = np.ones(len(grid_points))

            masks, scores, _ = predictor.predict(
                point_coords=input_points,
                point_labels=input_labels,
                multimask_output=False
            )
        return masks, scores

    def classify_masks(self, masks, image_bgr, text_prompts):
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        contours = []
        for i, mask in enumerate(masks):
            self.logger.info(f"\n--- Detected Object {i + 1} ---")
            mask_resized = cv2.resize(
                mask.astype(np.uint8),
                (image_rgb.shape[1], image_rgb.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
            masked = cv2.bitwise_and(image_rgb, image_rgb, mask=mask_resized)
            coords = cv2.findNonZero(mask_resized)
            if coords is None:
                continue
            x, y, w, h = cv2.boundingRect(coords)
            crop = masked[y : y + h, x : x + w]

            try:
                new_image, _ = self.crop_largest_mask_region(
                    crop, mask_resized[y : y + h, x : x + w]
                )
            except ValueError:
                continue

            new_image = Image.fromarray(new_image)
            image_input = self.clip_preprocess(new_image).unsqueeze(0).to(self.device)
            text_tokens = clip.tokenize(text_prompts).to(self.device)

            with torch.no_grad():
                image_features = self.clip_model.encode_image(image_input)
                text_features = self.clip_model.encode_text(text_tokens)
                logits_per_image = image_features @ text_features.T
                probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]

            for j, prob in enumerate(probs):
                self.logger.info(f"{text_prompts[j]}: {prob:.2f}")

            top_idx = int(np.argmax(probs))
            self.logger.info(f"Prediction: {text_prompts[top_idx]} ({probs[top_idx]:.2f})")
            # Extract the mask geometry after the prediction has been made
            (cx, cy), contour = self.extract_mask_geometry(mask_resized)
            contours.append({"name": text_prompts[top_idx], 
                             "probability": probs[top_idx],
                             "contour": contour,
                             "center": (cx, cy),
                             "mask": mask_resized})
            self.logger.info(f"Mask Geometry: {cx, cy}, {contour}")
        return contours


    def show_point_cloud(self, points_3d):
        """
        Visualize 3D point cloud using matplotlib.

        Args:
            points_3d (np.ndarray): (N, 3) array of [x, y, z] points
        """
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], s=1, c="r")

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("Projected 3D Point Cloud")
        plt.show()

    def crop_largest_mask_region(self, image: np.ndarray, mask: np.ndarray):
        mask_uint8 = mask.astype(np.uint8)
        contours, _ = cv2.findContours(
            mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            raise ValueError("No contours found in the mask.")
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        cropped_img = image[y : y + h, x : x + w]
        return cropped_img, (x, y, w, h)

    def extract_mask_geometry(self, mask: np.ndarray):
        mask_uint8 = mask.astype(np.uint8)
        contours, _ = cv2.findContours(
            mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            raise ValueError("No contours found in mask.")
        largest_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)
        if M["m00"] == 0:
            raise ValueError("Degenerate contour with zero area.")
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        contour = largest_contour[:, 0, :]
        return (cx, cy), contour

   
    def estimate_depth_from_mask(self, mask: np.ndarray, 
                                 fx: float, fy: float, known_width_m: float) -> float:
        # Bounding box of mask
        y_indices, x_indices = np.where(mask > 0)
        if len(x_indices) == 0 or len(y_indices) == 0:
            raise ValueError("Empty mask.")
        
        width_px = x_indices.max() - x_indices.min()
        # Assume width across x-axis; use fx
        Z = fx * known_width_m / width_px
        return Z


def main():
    IMAGE_PATH = (
        "/Users/pbanavara/dev/inference/yolo_seg_clip/test_image/teleoperator_view.png"
    )
    SAM_CHECKPOINT = "/Users/pbanavara/Downloads/sam_vit_b_01ec64.pth"
    MODEL_TYPE = "vit_b"
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

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    segmentation = SAMSegmentation(SAM_CHECKPOINT, MODEL_TYPE, device)

    image_bgr = cv2.imread(IMAGE_PATH)
    masks, _ = segmentation.predict(image_bgr, TEXT_PROMPTS)
    contours = segmentation.classify_masks(masks, image_bgr, TEXT_PROMPTS)


if __name__ == "__main__":
    main()
