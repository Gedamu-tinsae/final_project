"""ReLoRA utilities for periodic adapter merge/reset."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp


def _path_parts(path: Any) -> tuple[str, ...]:
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
    return tuple(parts)


def _collect_path_leaves(tree: Any) -> tuple[list[tuple[Any, Any]], dict[tuple[str, ...], Any]]:
    path_leaves, _ = jax.tree_util.tree_flatten_with_path(tree)
    by_key: dict[tuple[str, ...], Any] = {}
    for path, leaf in path_leaves:
        by_key[_path_parts(path)] = leaf
    return path_leaves, by_key


def _iter_adapter_roots(keys: list[tuple[str, ...]]) -> list[tuple[str, ...]]:
    roots: set[tuple[str, ...]] = set()
    for k in keys:
        if len(k) < 2:
            continue
        lo = k[-2].lower()
        if lo in {"lora_a", "lora_b"}:
            roots.add(k[:-2])
    return sorted(roots)


def _apply_updates(tree: Any, updates: dict[tuple[str, ...], Any]) -> Any:
    def map_fn(path, leaf):
        return updates.get(_path_parts(path), leaf)

    return jax.tree_util.tree_map_with_path(map_fn, tree)


def relora_merge_and_reset(
    llama_state: Any,
    *,
    rng: jax.Array,
    reset_std: float = 0.01,
) -> tuple[Any, int]:
    """Merge LoRA adapters into base weights then reinitialize adapters."""
    _, by_key = _collect_path_leaves(llama_state)
    roots = _iter_adapter_roots(list(by_key.keys()))

    updates: dict[tuple[str, ...], Any] = {}
    merged = 0
    key = rng
    for root in roots:
        a_key = root + ("lora_A", "w")
        b_key = root + ("lora_B", "w")
        # Some legacy trees may use lower-case keys.
        if a_key not in by_key:
            a_key = root + ("lora_a", "w")
        if b_key not in by_key:
            b_key = root + ("lora_b", "w")
        w_key = root + ("w",)
        if a_key not in by_key or b_key not in by_key or w_key not in by_key:
            continue
        a = by_key[a_key]
        b = by_key[b_key]
        w = by_key[w_key]
        if not (hasattr(a, "dtype") and hasattr(b, "dtype") and hasattr(w, "dtype")):
            continue

        merged_w = w + jnp.dot(a, b).astype(w.dtype)
        key, ka, kb = jax.random.split(key, 3)
        new_a = (jax.random.normal(ka, a.shape, dtype=a.dtype) * jnp.asarray(reset_std, dtype=a.dtype)).astype(a.dtype)
        new_b = jnp.zeros(b.shape, dtype=b.dtype)
        updates[w_key] = merged_w
        updates[a_key] = new_a
        updates[b_key] = new_b
        merged += 1

    return _apply_updates(llama_state, updates), int(merged)


def relora_merge_only(llama_state: Any) -> tuple[Any, int]:
    """Merge LoRA adapters into base weights without resetting adapters."""
    _, by_key = _collect_path_leaves(llama_state)
    roots = _iter_adapter_roots(list(by_key.keys()))

    updates: dict[tuple[str, ...], Any] = {}
    merged = 0
    for root in roots:
        a_key = root + ("lora_A", "w")
        b_key = root + ("lora_B", "w")
        if a_key not in by_key:
            a_key = root + ("lora_a", "w")
        if b_key not in by_key:
            b_key = root + ("lora_b", "w")
        w_key = root + ("w",)
        if a_key not in by_key or b_key not in by_key or w_key not in by_key:
            continue
        a = by_key[a_key]
        b = by_key[b_key]
        w = by_key[w_key]
        if not (hasattr(a, "dtype") and hasattr(b, "dtype") and hasattr(w, "dtype")):
            continue
        updates[w_key] = (w + jnp.dot(a, b).astype(w.dtype)).astype(w.dtype)
        merged += 1

    return _apply_updates(llama_state, updates), int(merged)
