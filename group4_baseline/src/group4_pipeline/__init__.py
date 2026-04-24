"""Group4 modular pipeline package."""

from .peft_smoke import main as run_peft_smoke
from .workflow import main as run_workflow
from .eval import (
    aggregate_pairwise_results,
    build_pairwise_requests,
    run_openai_judging,
    update_group4_results_with_eval,
    write_human_eval_pack,
)

__all__ = [
    "run_peft_smoke",
    "run_workflow",
    "build_pairwise_requests",
    "write_human_eval_pack",
    "aggregate_pairwise_results",
    "run_openai_judging",
    "update_group4_results_with_eval",
]
