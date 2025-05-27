import numpy as np
import cv2

def save_bbox_overlay(image: np.ndarray, bbox: list, output_path: str):
    """
    Draws bounding box on the RGB image and saves it to disk.

    Args:
        image (np.ndarray): RGB image (H, W, 3)
        bbox (list): [x_min, y_min, x_max, y_max]
        output_path (str): Path to save the image
    """
    # Convert RGB to BGR for OpenCV
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Extract coordinates
    x_min, y_min, x_max, y_max = map(int, bbox)

    # Draw bounding box
    cv2.rectangle(image_bgr, (x_min, y_min), (x_max, y_max), color=(0, 0, 255), thickness=2)

    # Draw center point
    cx, cy = (x_min + x_max) // 2, (y_min + y_max) // 2
    cv2.circle(image_bgr, (cx, cy), radius=4, color=(0, 255, 0), thickness=-1)

    # Save to file
    cv2.imwrite(output_path, image_bgr)
    print(f"Saved overlay to: {output_path}")