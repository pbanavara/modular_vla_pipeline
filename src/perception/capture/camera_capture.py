import mujoco
from mujoco import viewer
import mujoco
import numpy as np
from PIL import Image

class CameraCapture:
    def __init__(self, model_path: str):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=480, width=640)

    def capture_image(self, camera_name):
        mujoco.mj_forward(self.model, self.data)
        self.renderer.update_scene(self.data, camera=camera_name)
        rgb = self.renderer.render()
        # Call mjforward
        return Image.fromarray(rgb)
    
    def save_image(self, image: Image, filename):
        image.save(filename)


def main():
    # Load the MuJoCo model
    model = mujoco.MjModel.from_xml_path("aloha/aloha.xml")
    data = mujoco.MjData(model)
    capture = CameraCapture(model, data)
    image = capture.capture_image("teleoperator_pov")
    capture.save_image(image, "teleoperator_view.png") 

if __name__ == "__main__":
    main() 