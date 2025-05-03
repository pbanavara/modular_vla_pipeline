import torch
import clip
import numpy as np
import cv2
from PIL import Image
from segment_anything import sam_model_registry, SamPredictor
import matplotlib.pyplot as plt

# -------- CONFIG --------
class SAMSegmentation:
    def __init__(self, sam_checkpoint, model_type, device):
        self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        self.sam.to(device=device)
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=device)
        self.device = device

    def predict(self, image_path, text_prompts):
        image_rgb = cv2.cvtColor(image_path, cv2.COLOR_BGR2RGB)
        predictor = SamPredictor(self.sam)
        predictor.set_image(image_rgb)
        # -------- SAM SEGMENTATION --------
        # Get grid points across the image to guide mask prediction
        grid_points = []
        height, width = image.shape[:2]
        step = 150
        for y in range(0, height, step):
            for x in range(0, width, step):
                grid_points.append([x, y])

        input_points = np.array(grid_points)
        input_labels = np.ones(len(grid_points))

        masks, scores, _ = predictor.predict(
            point_coords=input_points, point_labels=input_labels, multimask_output=False
        )
        return masks, scores

    def predict_and_visualize(self, image_path, text_prompts):
        masks, scores = self.predict(image_path, text_prompts)
        # -------- VISUALIZE --------
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(12, 8))
        plt.imshow(image)
        plt.axis("off")
        for mask, score in zip(masks, scores):
            mask_bool = mask.astype(bool)
            # Make red overlay where mask is active
            red_overlay = np.zeros_like(image)
            red_overlay[mask_bool] = [255, 0, 0]
            image = cv2.addWeighted(image, 0.7, red_overlay, 0.3, 0)
            plt.imshow(image, alpha=0.5)
        plt.show()


    def crop_largest_mask_region(self, 
                                 image: np.ndarray, 
                                 mask: np.ndarray) -> np.ndarray:
        """
        Given an image and a binary mask, this function finds the largest contour in the mask
        and returns a tightly cropped region of the image around that contour.

        Args:
            image (np.ndarray): Original image (H x W x 3).
            mask (np.ndarray): Binary mask (H x W), values 0 or 1 (or bool).

        Returns:
            cropped_img (np.ndarray): Cropped image containing the largest mask region.
            bbox (tuple): (x, y, w, h) of the bounding box used to crop.
        """
        # Ensure mask is uint8
        mask_uint8 = mask.astype(np.uint8)

        # Find contours
        contours, _ = cv2.findContours(
            mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            raise ValueError("No contours found in the mask.")

        # Find the largest contour
        largest_contour = max(contours, key=cv2.contourArea)

        # Get bounding box around the largest contour
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Crop and return
        cropped_img = image[y : y + h, x : x + w]
        return cropped_img, (x, y, w, h)


    def extract_mask_geometry(self, mask: np.ndarray):
        """
        Given a binary mask (cropped or full), returns:
        - Centroid (u, v)
        - Contour points
        - Optional: a pseudo point cloud using depth prior

        Args:
            mask: Binary (bool or 0/1) mask, 2D

        Returns:
            centroid_uv: (u, v) tuple (image coordinates)
            contour: np.ndarray of shape (N, 2) for largest contour
        """
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

        contour = largest_contour[:, 0, :]  # (N, 2)
        return (cx, cy), contour

    def project_to_point_cloud(self,
            contour: np.ndarray, 
            Z: float, 
            fx: float, 
            fy: float, 
            cx: float, 
            cy: float
        ):
        """
        Backprojects 2D pixel contour points to 3D point cloud using depth and camera intrinsics.

        Args:
            contour: (N, 2) array of pixel points (u, v)
            Z: fixed depth value in meters
            fx, fy: focal lengths
            cx, cy: principal point (image center)

        Returns:
            points_3d: (N, 3) numpy array of 3D points
        """
        u = contour[:, 0]
        v = contour[:, 1]

        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy
        Z_arr = np.full_like(X, Z)

        return np.stack([X, Y, Z_arr], axis=1)


    # -------- CLIP CLASSIFICATION PER MASK --------
    def classify_masks(self, masks, image_path, text_prompts):
        """
        Classify each mask using CLIP.
        """
        for i, mask in enumerate(masks):
            print(f"\n--- Object {i + 1} ---")
            mask_resized = cv2.resize(
                mask.astype(np.uint8),
                (image.shape[1], image.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
            masked = cv2.bitwise_and(image_rgb, image_rgb, mask=mask_resized)
            coords = cv2.findNonZero(mask_resized)
            if coords is None:
                continue
            x, y, w, h = cv2.boundingRect(coords)
            crop = masked[y : y + h, x : x + w]

            pil_crop = Image.fromarray(crop)
            new_image, _ = self.crop_largest_mask_region(crop, mask_resized)
            new_image = Image.fromarray(new_image)
            image_input = self.clip_preprocess(new_image).unsqueeze(0).to(device)
            text_tokens = clip.tokenize(text_prompts).to(self.device)

            with torch.no_grad():
                image_features = clip_model.encode_image(image_input)
                text_features = clip_model.encode_text(text_tokens)
                logits_per_image = image_features @ text_features.T
                probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]

            for j, prob in enumerate(probs):
                print(f"{text_prompts[j]}: {prob:.2f}")

            top_idx = int(np.argmax(probs))
            print(f"Prediction: {text_prompts[top_idx]} ({probs[top_idx]:.2f})")

def main():
    IMAGE_PATH = "/Users/pbanavara/dev/inference/yolo_seg_clip/test_image/teleoperator_view.png"  # Replace with actual image path
    SAM_CHECKPOINT = "/Users/pbanavara/Downloads/sam_vit_b_01ec64.pth"  # Download from Meta's SAM release
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
        "a kitchen sink"
    ]

    # -------- SETUP --------
    device = "cuda" if torch.cuda.is_available() else "cpu"