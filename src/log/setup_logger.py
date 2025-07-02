# logger.py
import logging
import os
from datetime import datetime

# Cache for loggers to avoid duplicate setup
_logger_cache = {}

def setup_logger(name="pipeline"):
    # Return cached logger if it exists
    if name in _logger_cache:
        return _logger_cache[name]
    
    log_dir = "run_logs"
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logfile = os.path.join(log_dir, f"{name}_{timestamp}.log")

    logger = logging.getLogger(name)
    
    # Only setup handlers if they don't already exist
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)

        fh = logging.FileHandler(logfile)
        fh.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        logger.addHandler(fh)
        logger.addHandler(ch)
    
    # Cache the logger
    _logger_cache[name] = logger
    return logger
