"""ReLoRA utilities for periodic adapter merge/reset.

This module intentionally matches adapter/base leaves using path-string patterns
because NNX state trees may expose different leaf suffixes (e.g., ``/w`` or
``/value`` depending on serialization/version).
"""

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


def _find_adapter_triples(by_key: dict[tuple[str, ...], Any]) -> list[tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]]:
    """Find (A_key, B_key, W_key) triples in a robust way.

    - Detects A/B leaves by path segment containing lora_A/lora_B (case-insensitive).
    - Locates a sibling base weight leaf under the same module root.
    """
    key_list = list(by_key.keys())
    triples: list[tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]] = []
    seen: set[tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]] = set()

    for k in key_list:
        parts_low = [p.lower() for p in k]
        if "lora_a" not in parts_low:
            continue
        ia = parts_low.index("lora_a")
        root = k[:ia]
        suffix = k[ia + 1 :]
        b_key = root + ("lora_B",) + suffix
        if b_key not in by_key:
            b_key = root + ("lora_b",) + suffix
        if b_key not in by_key:
            continue

        # Candidate base weights under same root with same leaf suffix.
        w_candidates: list[tuple[str, ...]] = []
        for wk in key_list:
            if not wk[: len(root)] == root:
                continue
            wk_low = [p.lower() for p in wk]
            if "lora_a" in wk_low or "lora_b" in wk_low:
                continue
            if len(wk) >= len(suffix) and wk[-len(suffix) :] == suffix:
                w_candidates.append(wk)

        # Prefer common base-weight segment names if present.
        preferred = [wk for wk in w_candidates if any(seg.lower() in {"w", "kernel"} for seg in wk)]
        ordered = preferred if preferred else w_candidates
        if not ordered:
            continue
        w_key = ordered[0]

        t = (k, b_key, w_key)
        if t not in seen:
            seen.add(t)
            triples.append(t)
    return triples


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
    triples = _find_adapter_triples(by_key)

    updates: dict[tuple[str, ...], Any] = {}
    merged = 0
    key = rng
    for a_key, b_key, w_key in triples:
        a = by_key[a_key]
        b = by_key[b_key]
        w = by_key[w_key]
        if not (hasattr(a, "dtype") and hasattr(b, "dtype") and hasattr(w, "dtype")):
            continue

        delta = jnp.dot(a, b)
        if delta.shape != w.shape:
            if int(delta.size) != int(w.size):
                continue
            delta = jnp.reshape(delta, w.shape)
        merged_w = w + delta.astype(w.dtype)
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
    triples = _find_adapter_triples(by_key)

    updates: dict[tuple[str, ...], Any] = {}
    merged = 0
    for a_key, b_key, w_key in triples:
        a = by_key[a_key]
        b = by_key[b_key]
        w = by_key[w_key]
        if not (hasattr(a, "dtype") and hasattr(b, "dtype") and hasattr(w, "dtype")):
            continue
        delta = jnp.dot(a, b)
        if delta.shape != w.shape:
            if int(delta.size) != int(w.size):
                continue
            delta = jnp.reshape(delta, w.shape)
        updates[w_key] = (w + delta.astype(w.dtype)).astype(w.dtype)
        merged += 1

    return _apply_updates(llama_state, updates), int(merged)
