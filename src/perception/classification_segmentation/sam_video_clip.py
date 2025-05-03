import torch
import clip
import numpy as np
import cv2
from PIL import Image
from segment_anything import sam_model_registry, SamPredictor
import matplotlib.pyplot as plt

# -------- CONFIG --------
VIDEO_PATH = "/Users/pbanavara/Downloads/IMG_9605.MOV"  # Replace with your video path
SAM_CHECKPOINT = "/Users/pbanavara/Downloads/sam_vit_b_01ec64.pth"  # Download from Meta's SAM release
MODEL_TYPE = "vit_b"
TEXT_PROMPTS = [
    "a transparent cylindrical 8 oz drinking glass",
    "a transparent cylindrical 12 oz drinking glass",
    "a transparent conical 12 oz glass",
    "a transparent 16 oz cylindrical glass",
    "a ceramic coffee mug with a handle",
    "a stainless steel saucepan",
    "a pressure cooker with black handles",
]

# -------- SETUP --------
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
sam = sam_model_registry[MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
sam.to(device)
predictor = SamPredictor(sam)

# -------- VIDEO PROCESSING --------
cap = cv2.VideoCapture(VIDEO_PATH)
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1
    print(f"\n=== Frame {frame_count} ===")

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    predictor.set_image(image_rgb)

    # Grid sampling for prompts
    height, width = frame.shape[:2]
    step = 150
    grid_points = [
        [x, y] for y in range(0, height, step) for x in range(0, width, step)
    ]
    input_points = np.array(grid_points)
    input_labels = np.ones(len(grid_points))

    masks, scores, _ = predictor.predict(
        point_coords=input_points, point_labels=input_labels, multimask_output=False
    )

    predictions = []
    for i, mask in enumerate(masks):
        mask_resized = cv2.resize(
            mask.astype(np.uint8), (width, height), 
            interpolation=cv2.INTER_NEAREST
        )
        if mask_resized.sum() < 500:
            continue

        masked = cv2.bitwise_and(image_rgb, image_rgb, mask=mask_resized)
        coords = cv2.findNonZero(mask_resized)
        if coords is None:
            continue
        x, y, w, h = cv2.boundingRect(coords)
        if w < 20 or h < 20:
            continue

        crop = masked[y : y + h, x : x + w]
        pil_crop = Image.fromarray(crop)
        image_input = clip_preprocess(pil_crop).unsqueeze(0).to(device)
        text_tokens = clip.tokenize(TEXT_PROMPTS).to(device)

        with torch.no_grad():
            image_features = clip_model.encode_image(image_input)
            text_features = clip_model.encode_text(text_tokens)
            logits_per_image = image_features @ text_features.T
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]

        top_idx = int(np.argmax(probs))
        if probs[top_idx] < 0.5:
            continue

        predictions.append((probs[top_idx], TEXT_PROMPTS[top_idx], (x, y, w, h)))

    predictions.sort(key=lambda x: -x[0])

    for conf, label, bbox in predictions:
        print(f"Prediction: {label} ({conf:.2f}) at {bbox}")

cap.release()