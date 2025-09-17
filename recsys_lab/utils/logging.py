from loguru import logger
import sys

def setup_logger(level: str = "INFO"):
    logger.remove()
    logger.add(sys.stderr, level=level,
               format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | {message}")
    return logger
