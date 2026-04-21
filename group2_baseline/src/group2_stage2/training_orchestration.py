from __future__ import annotations

import json
import pickle
import random
from pathlib import Path
from typing import Any, Callable

import numpy as np


def stage2_collate_fn(batch: list[dict], pad_list: Callable[[list[int], int, int], list[int]]) -> dict:
    if not batch:
        raise ValueError("stage2_collate_fn received an empty batch.")

    max_text_len = max(len(x["input_ids"]) for x in batch)
    vision_feats = []
    input_ids = []
    labels = []
    expected_vis_shape = None

    for sample in batch:
        vf = np.load(sample["vision_path"], allow_pickle=False).astype(np.float32, copy=False)
        if vf.ndim != 2:
            raise ValueError(f"Expected [n_vis_tokens, vis_dim], got {vf.shape} from {sample['vision_path']}")
        if expected_vis_shape is None:
            expected_vis_shape = vf.shape
        elif vf.shape != expected_vis_shape:
            raise ValueError(f"Inconsistent vision feature shapes: expected {expected_vis_shape}, got {vf.shape}")

        vision_feats.append(vf)
        input_ids.append(pad_list(sample["input_ids"], max_text_len, 0))
        labels.append(pad_list(sample["labels"], max_text_len, -100))

    return {
        "vision_feats": np.stack(vision_feats, axis=0).astype(np.float32, copy=False),
        "input_ids": np.asarray(input_ids, dtype=np.int32),
        "labels": np.asarray(labels, dtype=np.int32),
    }


def iterate_stage2_minibatches(
    data: list[dict],
    batch_size: int,
    pad_list: Callable[[list[int], int, int], list[int]],
    shuffle: bool = True,
):
    idxs = list(range(len(data)))
    if shuffle:
        random.shuffle(idxs)
    for i in range(0, len(idxs), batch_size):
        batch = [data[j] for j in idxs[i : i + batch_size]]
        yield stage2_collate_fn(batch, pad_list=pad_list)


def evaluate_stage2(
    val_manifest_json: Path,
    batch_size: int,
    eval_step_fn: Callable[[dict], float],
    pad_list: Callable[[list[int], int, int], list[int]],
) -> dict:
    manifest = json.loads(val_manifest_json.read_text(encoding="utf-8"))
    losses = []
    for batch in iterate_stage2_minibatches(manifest, batch_size=batch_size, pad_list=pad_list, shuffle=False):
        loss = eval_step_fn(batch)
        losses.append(float(loss))
    mean_loss = sum(losses) / len(losses) if losses else None
    return {"mean_loss": mean_loss, "num_batches": len(losses), "batch_losses": losses}


def run_stage2_training(
    manifest_json: Path,
    val_manifest_json: Path | None,
    batch_size: int,
    num_epochs: int,
    log_every_steps: int,
    train_step_fn: Callable[[dict], float],
    eval_step_fn: Callable[[dict], float] | None,
    pad_list: Callable[[list[int], int, int], list[int]],
) -> dict:
    manifest = json.loads(manifest_json.read_text(encoding="utf-8"))
    if not manifest:
        raise ValueError(f"Training manifest is empty: {manifest_json}")

    history = {
        "train_step_losses": [],
        "epoch_train_averages": [],
        "val_epoch_averages": [],
        "train_cumulative_averages": [],
    }

    global_step = 0
    for epoch in range(num_epochs):
        epoch_losses = []
        for batch in iterate_stage2_minibatches(manifest, batch_size=batch_size, pad_list=pad_list):
            loss = train_step_fn(batch)
            loss_value = float(loss)
            history["train_step_losses"].append(loss_value)
            epoch_losses.append(loss_value)
            global_step += 1
            if log_every_steps > 0 and global_step % log_every_steps == 0:
                cumulative_mean = sum(history["train_step_losses"]) / len(history["train_step_losses"])
                history["train_cumulative_averages"].append(
                    {"epoch": epoch + 1, "global_step": global_step, "mean_loss": cumulative_mean}
                )

        epoch_train_avg = sum(epoch_losses) / len(epoch_losses) if epoch_losses else None
        history["epoch_train_averages"].append(
            {"epoch": epoch + 1, "mean_loss": epoch_train_avg, "num_steps": len(epoch_losses)}
        )

        if val_manifest_json is not None and eval_step_fn is not None:
            val_result = evaluate_stage2(val_manifest_json, batch_size, eval_step_fn, pad_list=pad_list)
            history["val_epoch_averages"].append(
                {"epoch": epoch + 1, "mean_loss": val_result["mean_loss"], "num_batches": val_result["num_batches"]}
            )

    final_train_avg = sum(history["train_step_losses"]) / len(history["train_step_losses"]) if history["train_step_losses"] else None
    final_val_result = None
    if val_manifest_json is not None and eval_step_fn is not None:
        final_val_result = evaluate_stage2(val_manifest_json, batch_size, eval_step_fn, pad_list=pad_list)

    return {
        "history": history,
        "final_train_mean_loss": final_train_avg,
        "final_val_result": final_val_result,
        "num_total_train_steps": global_step,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "log_every_steps": log_every_steps,
    }


def save_stage2_snapshot(reset_root: Path, projector_state: Any, llama_state: Any, opt_state: Any) -> dict:
    reset_root.mkdir(parents=True, exist_ok=True)
    projector_path = reset_root / "projector_state.pkl"
    llama_path = reset_root / "llama_state.pkl"
    opt_path = reset_root / "opt_state.pkl"
    projector_path.write_bytes(pickle.dumps(projector_state))
    llama_path.write_bytes(pickle.dumps(llama_state))
    opt_path.write_bytes(pickle.dumps(opt_state))
    return {"projector": str(projector_path), "llama": str(llama_path), "opt": str(opt_path)}


def load_stage2_snapshot(reset_root: Path) -> dict:
    projector_path = reset_root / "projector_state.pkl"
    llama_path = reset_root / "llama_state.pkl"
    opt_path = reset_root / "opt_state.pkl"
    return {
        "projector_state": pickle.loads(projector_path.read_bytes()),
        "llama_state": pickle.loads(llama_path.read_bytes()),
        "opt_state": pickle.loads(opt_path.read_bytes()),
    }

