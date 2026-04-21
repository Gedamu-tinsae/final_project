"""Runtime utility helpers from Group 1 notebook."""

import functools

import humanize
import jax


def show_hbm_usage():
    """Display per-device high-bandwidth memory usage."""
    fmt_size = functools.partial(humanize.naturalsize, binary=True)

    for device in jax.local_devices():
        stats = device.memory_stats()
        used = stats["bytes_in_use"]
        limit = stats["bytes_limit"]
        print(f"Using {fmt_size(used)} / {fmt_size(limit)} ({used/limit:%}) on {device}")
