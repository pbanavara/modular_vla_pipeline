import mujoco
import numpy as np

# 1. Sink info
sink_pos = np.array([0.0, 0.0, 0.1])
sink_quat = np.array([0.7071, 1.0, 0.0, 0.0])  # w, x, y, z

# 2. Plate local position (relative to sink)
plate_local = np.array([-0.1, -0.25, 0.25])

# 3. Convert quaternion to rotation matrix
R = np.zeros((3, 3))
mujoco.mju_quat2Mat(R.ravel(), sink_quat)  # R will be filled in-place

# 4. Transform plate position to world
plate_world = R @ plate_local + sink_pos
print("Transformed world position of plate:", plate_world)