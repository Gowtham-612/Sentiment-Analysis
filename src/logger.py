import logging
import os
import sys

LOG_DIR = "logs"
LOG_FILE = "pipeline.log"
os.makedirs(LOG_DIR, exist_ok=True)

# Ensure UTF-8 for Windows console
sys.stdout.reconfigure(encoding='utf-8')

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(module)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, LOG_FILE), encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("pipeline_logger")
