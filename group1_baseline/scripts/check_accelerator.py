"""Print JAX accelerator/backend info for safe run planning."""

from __future__ import annotations


def main() -> int:
    try:
        import jax
    except Exception as exc:
        print("JAX import failed:", exc)
        print("Install the project requirements first.")
        return 1

    backend = jax.default_backend()
    devices = jax.devices()
    print("JAX backend:", backend)
    print("Device count:", len(devices))
    for i, d in enumerate(devices):
        print(f"  [{i}] {d}")

    if backend == "cpu":
        print("WARNING: Running on CPU. GPU/TPU acceleration is not active.")
    else:
        print("Acceleration active.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

