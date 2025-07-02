import mujoco
import mujoco.viewer
import threading
import time
import os
import signal
import psutil
from typing import Optional, Tuple
from log.setup_logger import setup_logger


class ViewerManager:
    """
    Manages MuJoCo viewer instances to prevent crashes when multiple viewers are launched.
    """
    
    def __init__(self):
        self.logger = setup_logger("ViewerManager")
        self._viewer_lock = threading.Lock()
        self._current_viewer = None
        self._viewer_process = None
        
    def _find_mujoco_viewer_processes(self) -> list:
        """Find running MuJoCo viewer processes."""
        mujoco_processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                # Look for processes that might be MuJoCo viewers
                if proc.info['name'] and any(keyword in proc.info['name'].lower() 
                                           for keyword in ['mujoco', 'viewer', 'glfw']):
                    mujoco_processes.append(proc)
                elif proc.info['cmdline']:
                    cmdline = ' '.join(proc.info['cmdline']).lower()
                    if any(keyword in cmdline for keyword in ['mujoco', 'viewer']):
                        mujoco_processes.append(proc)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return mujoco_processes
    
    def _kill_existing_viewers(self):
        """Kill any existing MuJoCo viewer processes."""
        mujoco_processes = self._find_mujoco_viewer_processes()
        if mujoco_processes:
            self.logger.info(f"Found {len(mujoco_processes)} existing MuJoCo viewer processes")
            for proc in mujoco_processes:
                try:
                    self.logger.info(f"Terminating process {proc.info['pid']}: {proc.info['name']}")
                    proc.terminate()
                    proc.wait(timeout=5)  # Wait up to 5 seconds for graceful termination
                except psutil.TimeoutExpired:
                    self.logger.warning(f"Force killing process {proc.info['pid']}")
                    proc.kill()
                except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                    self.logger.warning(f"Could not terminate process {proc.info['pid']}: {e}")
            
            # Give some time for processes to fully terminate
            time.sleep(1.0)
        else:
            self.logger.info("No existing MuJoCo viewer processes found")
    
    def _check_viewer_alive(self) -> bool:
        """Check if the current viewer is still alive and responsive."""
        if self._current_viewer is None:
            return False
        
        try:
            # Try to access a viewer property to check if it's still responsive
            _ = self._current_viewer.is_running()
            return True
        except (AttributeError, RuntimeError, OSError):
            self.logger.warning("Current viewer is no longer responsive")
            return False
    
    def get_viewer(self, model, data, force_restart: bool = False) -> mujoco.viewer.Viewer:
        """
        Get a MuJoCo viewer, either reusing an existing one or creating a new one.
        
        Args:
            model: MuJoCo model
            data: MuJoCo data
            force_restart: If True, kill existing viewers and start fresh
            
        Returns:
            MuJoCo viewer instance
        """
        with self._viewer_lock:
            if force_restart:
                self.logger.info("Force restart requested - killing existing viewers")
                self._kill_existing_viewers()
                self._current_viewer = None
                self._viewer_process = None
            
            # Check if we have a valid existing viewer
            if self._current_viewer is not None and self._check_viewer_alive():
                self.logger.info("Reusing existing viewer")
                return self._current_viewer
            
            # No valid viewer exists, create a new one
            self.logger.info("Creating new MuJoCo viewer")
            
            # Kill any existing viewers first to prevent conflicts
            self._kill_existing_viewers()
            
            try:
                # Launch new viewer
                self._current_viewer = mujoco.viewer.launch_passive(model, data)
                self.logger.info("Successfully launched new MuJoCo viewer")
                return self._current_viewer
                
            except Exception as e:
                self.logger.error(f"Failed to launch viewer: {e}")
                # Try one more time after a brief delay
                time.sleep(0.5)
                try:
                    self._current_viewer = mujoco.viewer.launch_passive(model, data)
                    self.logger.info("Successfully launched MuJoCo viewer on second attempt")
                    return self._current_viewer
                except Exception as e2:
                    self.logger.error(f"Failed to launch viewer on second attempt: {e2}")
                    raise
    
    def close_viewer(self):
        """Close the current viewer if it exists."""
        with self._viewer_lock:
            if self._current_viewer is not None:
                try:
                    self.logger.info("Closing current viewer")
                    self._current_viewer.close()
                except Exception as e:
                    self.logger.warning(f"Error closing viewer: {e}")
                finally:
                    self._current_viewer = None
                    self._viewer_process = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close_viewer()


# Global viewer manager instance
_viewer_manager = None

def get_viewer_manager() -> ViewerManager:
    """Get the global viewer manager instance."""
    global _viewer_manager
    if _viewer_manager is None:
        _viewer_manager = ViewerManager()
    return _viewer_manager


def safe_launch_viewer(model, data, force_restart: bool = False) -> mujoco.viewer.Viewer:
    """
    Safely launch a MuJoCo viewer, handling existing viewers.
    
    Args:
        model: MuJoCo model
        data: MuJoCo data
        force_restart: If True, kill existing viewers and start fresh
        
    Returns:
        MuJoCo viewer instance
    """
    manager = get_viewer_manager()
    return manager.get_viewer(model, data, force_restart=force_restart)


def close_all_viewers():
    """Close all MuJoCo viewers."""
    manager = get_viewer_manager()
    manager.close_viewer()
    manager._kill_existing_viewers() 