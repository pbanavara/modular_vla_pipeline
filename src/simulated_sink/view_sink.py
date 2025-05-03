import mujoco
from mujoco import viewer
import time
import numpy as np
from PIL import Image
from ultralytics import YOLO

model = mujoco.MjModel.from_xml_path("/Users/pbanavara/dev/mujoco_menagerie/aloha/aloha.xml")
data = mujoco.MjData(model)

# Set up the renderer
renderer = mujoco.Renderer(model, height=480, width=640)
mujoco.mj_forward(model, data)

# Render from your defined camera
renderer.update_scene(data, camera="teleoperator_pov")
rgb = renderer.render()

# Save or view the image
Image.fromarray(rgb).save("teleoperator_view.png")
model = YOLO("yolov8n.pt")  # or your fine-tuned one
results = model("teleoperator_view.png")
results[0].show()  # optional
