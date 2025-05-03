import torch
import clip
import numpy as np
import logging
import cv2
from PIL import Image
from ultralytics import YOLO
import os


# -------- CONFIG --------
YOLO_MODEL_PATH = "yolov8x-seg.pt"
IMAGE_PATH = "glass_tumbler_sink.jpeg"  # replace with your test image
TEXT_PROMPTS = [
    "a faceted 8 oz glass",
    "a cylindrical 12 oz drinking glass",
    "a ceramic mug",
    "a stainless steel saucepan",
    "a pressure cooker with black handles",
]

# -------- SETUP --------
device = "cuda" if torch.cuda.is_available() else "cpu"
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
yolo_model = YOLO(YOLO_MODEL_PATH)


# -------- HELPERS --------
def apply_mask_and_crop(image, mask):
    mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
    mask = mask.astype(np.uint8) * 255
    masked = cv2.bitwise_and(image, image, mask=mask)

    # Find bounding box
    coords = cv2.findNonZero(mask)
    x, y, w, h = cv2.boundingRect(coords)
    crop = masked[y : y + h, x : x + w]
    return crop

import matplotlib.pyplot as plt


def visualize_masks(image_path, results):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(12, 8))
    plt.imshow(img)
    plt.axis("off")

    for i, seg in enumerate(results.masks.data):
        mask = seg.cpu().numpy()
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        color = np.random.rand(3)  # random color for each mask
        colored_mask = np.zeros_like(img, dtype=np.uint8)
        for c in range(3):
            colored_mask[:, :, c] = mask * int(color[c] * 255)

        # Overlay
        img = cv2.addWeighted(img, 1, colored_mask, 0.4, 0)

    plt.imshow(img)
    plt.title("Segment Mask Overlay")
    plt.show()


# -------- INFERENCE --------
# Read a directory of images and run inference for all of them

dir_path = "/Users/pbanavara/dev/inference/yolo_seg_clip/test_image"
for filename in os.listdir(dir_path):
    if not filename.endswith(".jpeg"): 
        continue

    image_path = os.path.join(dir_path, filename)
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = yolo_model(image_path)[0]
    #visualize_masks(dir_path + "/" + filename, results)

    if not results.masks:
        print("No objects detected.")
        exit()

    for idx, seg in enumerate(results.masks.data):
        print(f"\n--- Object {idx + 1} ---")
        mask = seg.cpu().numpy()
        crop = apply_mask_and_crop(image_rgb, mask)

        pil_crop = Image.fromarray(crop)
        image_input = clip_preprocess(pil_crop).unsqueeze(0).to(device)
        text_tokens = clip.tokenize(TEXT_PROMPTS).to(device)

        with torch.no_grad():
            image_features = clip_model.encode_image(image_input)
            text_features = clip_model.encode_text(text_tokens)
            logits_per_image = image_features @ text_features.T
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]

        #for i, prob in enumerate(probs):
        #    print(f"{TEXT_PROMPTS[i]}: {prob:.2f}")

        top_idx = int(np.argmax(probs))
        print(f"\nPrediction: {TEXT_PROMPTS[top_idx]} ({probs[top_idx]:.2f})")

        # Store the image file name and the predictions in a csv file
        with open("predictions.csv", "a") as f:
            f.write(f"{filename},{TEXT_PROMPTS[top_idx]},{probs[top_idx]}\n")
