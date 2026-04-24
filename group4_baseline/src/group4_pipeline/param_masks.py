"""Parameter mask and counting utilities for Group4 PEFT."""

from __future__ import annotations

import random
from typing import Any

import jax
import jax.numpy as jnp


def keypath_to_str(path) -> str:
    parts: list[str] = []
    for p in path:
        if hasattr(p, "name"):
            parts.append(str(p.name))
        elif hasattr(p, "key"):
            parts.append(str(p.key))
        elif hasattr(p, "idx"):
            parts.append(str(p.idx))
        else:
            parts.append(str(p))
    return "/".join(parts)


def module_match(path_str: str, target_modules: str) -> bool:
    s = path_str.lower()
    if target_modules == "all":
        return True
    if target_modules == "qv":
        return ("q_proj" in s) or ("v_proj" in s)
    raise ValueError(f"Unknown target_modules: {target_modules}")


def build_lora_mask(llama_state):
    def mask_fn(path, leaf):
        if not hasattr(leaf, "dtype"):
            return False
        s = keypath_to_str(path).lower()
        return ("lora_a" in s) or ("lora_b" in s)

    return jax.tree_util.tree_map_with_path(mask_fn, llama_state)


def build_selective_mask(llama_state, budget_pct: float, target_modules: str, seed: int, strategy: str):
    leaves_with_path = jax.tree_util.tree_leaves_with_path(llama_state)
    candidates: list[tuple[Any, str, Any]] = []
    for path, leaf in leaves_with_path:
        if not hasattr(leaf, "dtype"):
            continue
        if not jnp.issubdtype(leaf.dtype, jnp.number):
            continue
        pstr = keypath_to_str(path)
        if module_match(pstr, target_modules):
            candidates.append((path, pstr, leaf))

    if not candidates:
        raise ValueError("No candidate LLaMA parameters matched selective fine-tuning filter.")

    k = max(1, int(round((budget_pct / 100.0) * len(candidates))))
    if strategy == "magnitude":
        scored = sorted(candidates, key=lambda x: float(jnp.mean(jnp.abs(x[2]))), reverse=True)
        selected = scored[:k]
    elif strategy == "random":
        rng = random.Random(seed)
        selected = rng.sample(candidates, k=k)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    selected_paths = {keypath_to_str(p) for p, _, _ in selected}

    def mask_fn(path, leaf):
        if not hasattr(leaf, "dtype"):
            return False
        return keypath_to_str(path) in selected_paths

    mask_tree = jax.tree_util.tree_map_with_path(mask_fn, llama_state)
    return mask_tree, len(candidates), len(selected_paths)


def zero_grads_where_mask_false(grads, mask):
    return jax.tree_util.tree_map(
        lambda g, m: g if m else jnp.zeros_like(g),
        grads,
        mask,
    )


def count_params(tree, mask=None) -> int:
    if mask is None:
        leaves = jax.tree_util.tree_leaves(tree)
        return int(sum(int(x.size) for x in leaves if hasattr(x, "size")))
    leaves = jax.tree_util.tree_leaves(tree)
    mask_leaves = jax.tree_util.tree_leaves(mask)
    total = 0
    for x, m in zip(leaves, mask_leaves):
        if bool(m) and hasattr(x, "size"):
            total += int(x.size)
    return int(total)


def materialize_abstract_leaves(tree: Any) -> tuple[Any, int]:
    """Replace ShapeDtypeStruct leaves with concrete zero arrays for JIT compatibility."""
    replaced = 0

    def to_array(x):
        nonlocal replaced
        if isinstance(x, jax.ShapeDtypeStruct):
            replaced += 1
            return jnp.zeros(x.shape, dtype=x.dtype)
        return x

    return jax.tree_util.tree_map(to_array, tree), replaced
