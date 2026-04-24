"""Group4 modular pipeline package."""

from .peft_smoke import main as run_peft_smoke
from .workflow import main as run_workflow

__all__ = ["run_peft_smoke", "run_workflow"]
