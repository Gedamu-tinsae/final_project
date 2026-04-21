"""Environment preflight checks for Group 1 baseline.

Usage:
  python scripts/check_env.py --profile core
  python scripts/check_env.py --profile notebook
  python scripts/check_env.py --profile tpu
  python scripts/check_env.py --profile all
"""

from __future__ import annotations

import argparse
import importlib.util
import sys


REQUIRED = {
    "core": [
        "numpy",
        "PIL",
        "jax",
        "flax",
        "optax",
        "transformers",
        "huggingface_hub",
        "datasets",
        "humanize",
    ],
    "notebook": [
        "dotenv",
        "kagglehub",
        "nest_asyncio",
        "ipywidgets",
        "wandb",
        "tensorflow",
        "tensorflow_datasets",
        "tensorboardX",
        "grain",
    ],
    "tpu": [
        "tunix",
        "qwix",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check Python dependencies by profile.")
    parser.add_argument(
        "--profile",
        choices=["core", "notebook", "tpu", "all"],
        default="core",
        help="Dependency profile to validate.",
    )
    return parser.parse_args()


def module_exists(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def main() -> int:
    args = parse_args()
    profiles = ["core", "notebook", "tpu"] if args.profile == "all" else [args.profile]

    missing: list[tuple[str, str]] = []
    print(f"Checking profiles: {', '.join(profiles)}")

    for profile in profiles:
        print(f"\n[{profile}]")
        for module in REQUIRED[profile]:
            ok = module_exists(module)
            status = "OK" if ok else "MISSING"
            print(f"  - {module}: {status}")
            if not ok:
                missing.append((profile, module))

    if not missing:
        print("\nAll required imports are available.")
        return 0

    print("\nMissing imports found:")
    for profile, module in missing:
        print(f"  - ({profile}) {module}")

    print("\nSuggested installs:")
    if any(p == "core" for p, _ in missing):
        print("  pip install -r requirements-core.txt")
    if any(p == "notebook" for p, _ in missing):
        print("  pip install -r requirements-notebook.txt")
    if any(p == "tpu" for p, _ in missing):
        print("  pip install -r requirements-tpu.txt")

    return 1


if __name__ == "__main__":
    sys.exit(main())
