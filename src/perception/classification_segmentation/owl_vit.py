from transformers import OwlViTProcessor, OwlViTForObjectDetection
from PIL import Image
import cv2
import torch
import argparse
import os
import numpy as np
from log import setup_logger


class OwlVitDetector:
    """
    A class for object detection using the OwlViT model.
    """

    def __init__(self, model_name="google/owlvit-base-patch32"):
        """
        Initialize the OwlViT detector.

        Args:
            model_name (str): The name of the pre-trained model to use.
        """
        # Load model
        self.processor = OwlViTProcessor.from_pretrained(model_name)
        self.model = OwlViTForObjectDetection.from_pretrained(model_name)

        # Set device
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.model.to(self.device)
        self.logger = setup_logger.setup_logger("OwlViTDetector")
        print(f"Using device: {self.device}")

    def detect_objects(self, image, prompts, threshold=0.1):
        """
        Detect objects in an image based on text prompts.

        Args:
            image_path (str): Path to the input image.
            prompts (list): List of text prompts for object detection.
            threshold (float): Confidence threshold for detection.

        Returns:
            tuple: (image, results) where image is the PIL Image and results contains detection data.
        """
        # Load image
        #image = Image.open(image_path)
        self.logger.info(f"Prompts : {prompts}")

        # Image hash
        self.logger.info(f"Image hash:, {np.array(image).sum()}")

        # Prepare text prompts
        texts = prompts

        # Resize the image for the model
        self.logger.info(f"Resized image {np.array(image).shape}")

        # Forward pass
        inputs = self.processor(text=texts, images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        for k, v in inputs.items():
            self.logger.info(f"{k}: shape={v.shape}, device={v.device}, dtype={v.dtype}")
        #self.logger.info("Model device:", next(self.model.parameters()).device)
        # Post-process
        target_sizes = torch.tensor([image.size[::-1]]).to(
            self.device
        )  # (height, width)
        results = self.processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=threshold
        )
        self.logger.info(f"Raw Results: {results}")

        if results[0]["scores"].numel() == 0:
            self.logger.info("No objects detected")
            return image, None

        # Print detection results
        for score, label, box in zip(
            results[0]["scores"], results[0]["labels"], results[0]["boxes"]
        ):
            box_coords = box.tolist()
            self.logger.info(
                f"Detected {texts[label]} with confidence {round(score.item(), 3)} at {box_coords}"
            )

        return image, results[0]

    def visualize_detections(self, image, results, output_path=None):
        """
        Visualize detection results by drawing bounding boxes on the image.

        Args:
            image_path (str): Path to the input image.
            results (dict): Detection results from detect_objects.
            output_path (str, optional): Path to save the output image. If None,
                                         a default name will be used.

        Returns:
            str: Path to the saved output image.
        """
        if results is None:
            print("No results to visualize")
            return None

        # Load image with OpenCV for visualization
        # Load the PIL Image into OpenCV format
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Get the prompts from the results
        texts = [[results["labels"][i].item()] for i in range(len(results["labels"]))]

        # Draw bounding boxes and labels
        for score, label, box in zip(
            results["scores"], results["labels"], results["boxes"]
        ):
            x1, y1, x2, y2 = map(int, box.tolist())

            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)

            # Add label and confidence
            text = f"{label.item()} ({score.item():.2f})"
            cv2.putText(
                image,
                text,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

        # Determine output path
        if output_path is None:
            base_name = os.path.basename("file_name")
            output_path = "output_" + base_name + ".jpg"

        # Save the image
        cv2.imwrite(output_path, image)
        print(f"Visualization saved to {output_path}")

        return output_path


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Detect objects in an image using OwlViT"
    )
    parser.add_argument("file_name", help="Name of the file to process")
    parser.add_argument("prompts", help="Comma-separated list of prompts")
    parser.add_argument(
        "--threshold", type=float, default=0.1, help="Detection confidence threshold"
    )
    parser.add_argument("--output", help="Output image path")

    args = parser.parse_args()

    # Process prompts
    prompts = args.prompts.split(",")
    prompts = [prompt.strip() for prompt in prompts if prompt.strip()]

    if not prompts:
        print("Error: No valid prompts provided")
        return

    # Initialize detector
    detector = OwlVitDetector()

    # Convert image path to PIL Image 
    image = Image.open(args.file_name)

    # Detect objects
    res_image, results = detector.detect_objects(
        image=image, prompts=prompts, threshold=args.threshold
    )

    # Visualize results
    if results is not None:
        detector.visualize_detections(
            image=image, results=results, output_path=args.output
        )


if __name__ == "__main__":
    main()
