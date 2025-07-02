from pipeline.async_sim.async_simulation import MujocoRealtimeExecutor
from utils.utilities import get_resolved_path
import asyncio
import time
from log.setup_logger import setup_logger
logger = setup_logger("main")

def main():
    #pipeline.run_pipeline()
    start = time.time()
    logger.info(f"Starting main {start}")
    mujoco_model_path = str(get_resolved_path("../simulated_sink/aloha/aloha.xml"))
    logger.info(f"Mujoco model loading time {time.time() - start}")
    executor = MujocoRealtimeExecutor(mujoco_model_path)
    asyncio.run(executor.start())

if __name__ == "__main__":
    main()