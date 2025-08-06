"""
Resource Cleanup Utilities

Provides functions for cleaning up various resources including LLMs, GPU memory,
Ray clusters, and multiprocessing resources to prevent memory leaks and
ensure proper shutdown.
"""

import atexit
import multiprocessing
import contextlib
import gc
from pathlib import Path
import torch
import ray
from torch.utils.tensorboard import SummaryWriter


def full_cleanup(backend_type: str = "litellm"):
    """Cleans up all resources: LLMs, GPUs, Ray, and multiprocessing."""
    print("Shutting down: Clearing all resources...")

    # Shut down Ray if it's running
    if ray.is_initialized():
        ray.shutdown()

    # Only run vLLM cleanup if the backend is vLLM
    if backend_type.lower() == "vllm":
        try:
            from vllm.distributed import (
                destroy_model_parallel,
                destroy_distributed_environment
            )
            # Ensure vLLM's distributed model is fully shut down
            destroy_model_parallel()
            destroy_distributed_environment()
            print("vLLM distributed cleanup completed.")
        except ImportError:
            print("vLLM not available, skipping vLLM-specific cleanup.")
        except Exception as e:
            print(f"Error during vLLM cleanup: {e}")

    # Ensure all multiprocessing child processes are terminated
    for child in multiprocessing.active_children():
        child.terminate()

    # Clean up PyTorch Distributed
    with contextlib.suppress(AssertionError):
        torch.distributed.destroy_process_group()

    # Run garbage collection to clear lingering references
    gc.collect()

    # Free unused GPU memory (only if CUDA is available)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print("GPU memory cleared.")
    else:
        print("CUDA not available, skipping GPU cleanup.")

    print("Cleanup complete: All processes and memory released.")


# Register cleanup to run automatically at exit
atexit.register(full_cleanup)


def cleanup_old_tensorboard_logs(log_dir: str = "runs/",
                                 keep_last: int = 5):
    """
    Deletes older TensorBoard logs to save disk space.

    Args:
        log_dir (str): The directory where TensorBoard logs are stored.
        keep_last (int): Number of most recent logs to keep.
    """
    log_path = Path(log_dir)
    if not log_path.exists():
        return  # No logs to delete

    log_files = list(log_path.glob("events.out.tfevents.*"))
    files = sorted(log_files, key=lambda f: f.stat().st_mtime)

    # Delete all but the most recent 'keep_last' logs
    for file in files[:-keep_last]:
        try:
            file.unlink()
            print(f"Deleted old TensorBoard log: {file}")
        except OSError as e:
            print(f"Error deleting {file}: {e}")


# Call cleanup before initializing TensorBoard logging
# cleanup_tensorboard_logs(log_dir="runs/kuhn_poker", keep_last=5)
