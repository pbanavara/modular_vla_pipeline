import mujoco
import mujoco.viewer
import os

model_path = "./custom_2_dof.xml"
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
