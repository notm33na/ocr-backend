"""
GPU Health Check Module (Stub for Docker deployment)
In Docker/cloud environments, GPU is typically not available
"""
import torch
import logging

logger = logging.getLogger(__name__)

def assert_gpu_ok(device_id=0, require_gpu=False):
    """Check GPU availability (stub for cloud deployment)"""
    has_gpu = torch.cuda.is_available()
    if require_gpu and not has_gpu:
        raise RuntimeError("GPU required but not available")
    if not has_gpu:
        logger.info("GPU not available - will use CPU")
    return has_gpu

def check_gpu_health(device_id=0):
    """Check GPU health (stub for cloud deployment)"""
    has_gpu = torch.cuda.is_available()
    if has_gpu:
        logger.info(f"GPU {device_id} is available")
    else:
        logger.info("GPU not available - using CPU")
    return has_gpu

