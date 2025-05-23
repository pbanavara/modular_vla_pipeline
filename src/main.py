from pipeline import pipeline
from pipeline.async_sim.async_simulation import MujocoRealtimeExecutor
from utils.utilities import get_resolved_path
import asyncio

def main():
    #pipeline.run_pipeline()
    mujoco_model_path = str(get_resolved_path("../simulated_sink/aloha/aloha.xml"))
    executor = MujocoRealtimeExecutor(mujoco_model_path)
    asyncio.run(executor.start())

if __name__ == "__main__":
    main()