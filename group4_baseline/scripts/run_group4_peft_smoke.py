"""Run Group4 PEFT smoke training on top of Group1 artifacts.

Supports:
- LoRA (using vendored Group4 backbone files in group4_baseline/src/group4_backbones)
- Selective fine-tuning (mask-based updates on Group1 model)
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import pickle
import random
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx
import optax

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
GROUP1_ROOT = REPO_ROOT / "group1_baseline"
GROUP4_BACKBONES_ROOT = PROJECT_ROOT / "src" / "group4_backbones"

for p in (GROUP1_ROOT, REPO_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from common.run_metrics import RunTracker
from src.config_loader import load_dotenv_file  # type: ignore
from src.model_internals.loader_pipeline import ensure_llama_artifacts, load_llama_model_and_tokenizer  # type: ignore
from src.training.batching import iterate_minibatches  # type: ignore
from src.training.losses import masked_cross_entropy_loss  # type: ignore
from src.training.multimodal import make_multimodal_inputs  # type: ignore
from src.training.projector import VisionProjector  # type: ignore
from src.training.train_pipeline import (  # type: ignore
    _infer_clip_dim_from_manifest,
    build_smoke_manifest_from_existing_features,
)


def _expand_project_root(cfg: dict[str, Any], project_root: Path) -> dict[str, Any]:
    token = "${PROJECT_ROOT}"
    out: dict[str, Any] = {}
    for k, v in cfg.items():
        if isinstance(v, str):
            out[k] = v.replace(token, str(project_root))
        elif isinstance(v, list):
            out[k] = [x.replace(token, str(project_root)) if isinstance(x, str) else x for x in v]
        elif isinstance(v, dict):
            out[k] = _expand_project_root(v, project_root)
        else:
            out[k] = v
    return out


def _load_module_from_file(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load module spec: {file_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _keypath_to_str(path) -> str:
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


def _module_match(path_str: str, target_modules: str) -> bool:
    s = path_str.lower()
    if target_modules == "all":
        return True
    if target_modules == "qv":
        return ("q_proj" in s) or ("v_proj" in s)
    if target_modules == "qv_mlp":
        return ("q_proj" in s) or ("v_proj" in s) or ("mlp" in s)
    raise ValueError(f"Unknown target_modules: {target_modules}")


def _build_lora_mask(llama_state):
    def mask_fn(path, leaf):
        if not hasattr(leaf, "dtype"):
            return False
        s = _keypath_to_str(path).lower()
        return ("lora_a" in s) or ("lora_b" in s)

    return jax.tree_util.tree_map_with_path(mask_fn, llama_state)


def _build_selective_mask(llama_state, budget_pct: float, target_modules: str, seed: int, strategy: str):
    leaves_with_path = jax.tree_util.tree_leaves_with_path(llama_state)
    candidates: list[tuple[Any, str, Any]] = []
    for path, leaf in leaves_with_path:
        if not hasattr(leaf, "dtype"):
            continue
        if not jnp.issubdtype(leaf.dtype, jnp.number):
            continue
        pstr = _keypath_to_str(path)
        if _module_match(pstr, target_modules):
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

    selected_paths = {_keypath_to_str(p) for p, _, _ in selected}

    def mask_fn(path, leaf):
        if not hasattr(leaf, "dtype"):
            return False
        return _keypath_to_str(path) in selected_paths

    mask_tree = jax.tree_util.tree_map_with_path(mask_fn, llama_state)
    return mask_tree, len(candidates), len(selected_paths)


def _zero_grads_where_mask_false(grads, mask):
    return jax.tree_util.tree_map(
        lambda g, m: g if m else jnp.zeros_like(g),
        grads,
        mask,
    )


def _count_params(tree, mask=None) -> int:
    if mask is None:
        leaves = jax.tree_util.tree_leaves(tree)
        return int(
            sum(int(x.size) for x in leaves if hasattr(x, "size"))
        )
    leaves = jax.tree_util.tree_leaves(tree)
    mask_leaves = jax.tree_util.tree_leaves(mask)
    total = 0
    for x, m in zip(leaves, mask_leaves):
        if bool(m) and hasattr(x, "size"):
            total += int(x.size)
    return int(total)


def _materialize_abstract_leaves(tree: Any) -> tuple[Any, int]:
    """Replace ShapeDtypeStruct leaves with concrete zero arrays for JIT compatibility."""
    replaced = 0

    def to_array(x):
        nonlocal replaced
        if isinstance(x, jax.ShapeDtypeStruct):
            replaced += 1
            return jnp.zeros(x.shape, dtype=x.dtype)
        return x

    return jax.tree_util.tree_map(to_array, tree), replaced


def _query_gpu_snapshot() -> dict[str, float] | None:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total,power.draw",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None

    if not out:
        return None

    first = out.splitlines()[0]
    parts = [p.strip() for p in first.split(",")]
    if len(parts) < 4:
        return None

    try:
        return {
            "gpu_util_pct": float(parts[0]),
            "gpu_mem_used_mb": float(parts[1]),
            "gpu_mem_total_mb": float(parts[2]),
            "gpu_power_w": float(parts[3]),
        }
    except Exception:
        return None


class _GPUSampler:
    def __init__(self, interval_sec: float = 2.0) -> None:
        self.interval_sec = interval_sec
        self._samples: list[dict[str, float]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        def loop() -> None:
            while not self._stop.is_set():
                s = _query_gpu_snapshot()
                if s is not None:
                    self._samples.append(s)
                self._stop.wait(self.interval_sec)

        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def summary(self) -> dict[str, Any] | None:
        if not self._samples:
            return None
        util = [s["gpu_util_pct"] for s in self._samples]
        mem = [s["gpu_mem_used_mb"] for s in self._samples]
        pwr = [s["gpu_power_w"] for s in self._samples]
        return {
            "num_samples": len(self._samples),
            "gpu_util_avg_pct": statistics.fmean(util),
            "gpu_util_max_pct": max(util),
            "gpu_mem_used_avg_mb": statistics.fmean(mem),
            "gpu_mem_used_max_mb": max(mem),
            "gpu_power_avg_w": statistics.fmean(pwr),
            "gpu_power_max_w": max(pwr),
            "gpu_mem_total_mb": self._samples[-1]["gpu_mem_total_mb"],
        }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Group4 PEFT smoke training.")
    p.add_argument("--config", default="configs/workflow_paths.json")
    p.add_argument("--method", choices=["lora", "selective_ft"], required=True)
    p.add_argument(
        "--lora-variant",
        choices=["qv", "all_weights"],
        default="qv",
        help="Which partner LoRA model variant to use when --method lora.",
    )
    p.add_argument("--target-modules", choices=["qv", "qv_mlp", "all"], default="qv")
    p.add_argument("--selection-strategy", choices=["magnitude", "random"], default="magnitude")
    p.add_argument("--budget-pct", type=float, default=1.0, help="Selective FT budget percentage of candidate leaves.")
    p.add_argument("--max-rows", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--val-frac", type=float, default=0.2, help="Fraction of smoke rows held out for validation.")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--append-manual-results",
        action="store_true",
        help="Append this run to group4_results_manual.json (with placeholder win_rate/val_loss).",
    )
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--output-root", default=str(REPO_ROOT / "outputs"))
    p.add_argument("--run-name", default="group4_peft")
    return p.parse_args()


def _try_plot_series(out_png: Path, xs: list[float], ys: list[float], title: str, xlabel: str, ylabel: str) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return
    if not xs or not ys:
        return
    plt.figure(figsize=(8, 4))
    plt.plot(xs, ys)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=150)
    plt.close()


def main() -> int:
    args = parse_args()
    load_dotenv_file(GROUP1_ROOT / ".env")

    cfg_path = (PROJECT_ROOT / args.config).resolve()
    raw_cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    cfg = _expand_project_root(raw_cfg, PROJECT_ROOT)
    tracker = RunTracker(
        group="group4",
        output_root=Path(args.output_root),
        run_name=args.run_name,
        config={"args": vars(args), "project_root": str(PROJECT_ROOT), "config": str(cfg_path)},
    )
    stdio = tracker.start_stdio_capture()
    try:
        print("RUN_DIR:", tracker.run_dir)

        req = cfg["required_inputs"]
        stage1_manifest = Path(req["group1_stage1_manifest"])
        stage1_projector_state_path = Path(req["group1_stage1_projector_state"])
        if not stage1_manifest.exists():
            raise FileNotFoundError(stage1_manifest)
        if not stage1_projector_state_path.exists():
            raise FileNotFoundError(stage1_projector_state_path)
    
        out_dir = PROJECT_ROOT / "artifacts" / "peft_smoke"
        out_dir.mkdir(parents=True, exist_ok=True)
        smoke_manifest = out_dir / "stage1_manifest_smoke_group4.json"
        smoke_info = build_smoke_manifest_from_existing_features(
            source_manifest_json=stage1_manifest,
            output_manifest_json=smoke_manifest,
            max_rows=args.max_rows,
        )
    
        print("JAX backend:", jax.default_backend())
        print("JAX devices:", [str(d) for d in jax.devices()])
        print("smoke manifest:", smoke_info)
    
        llama_dir = GROUP1_ROOT / "data" / "models" / "Llama-3.2-1B-Instruct"
        art = ensure_llama_artifacts(
            repo_id="meta-llama/Llama-3.2-1B-Instruct",
            local_dir=llama_dir,
        )
        print("llama artifacts:", art["mode"], art["local_dir"])
    
        if args.method == "lora":
            if args.lora_variant == "qv":
                model_file = GROUP4_BACKBONES_ROOT / "model_qv.py"
                params_file = GROUP4_BACKBONES_ROOT / "params_qv.py"
            else:
                model_file = GROUP4_BACKBONES_ROOT / "model_all_weights.py"
                params_file = GROUP4_BACKBONES_ROOT / "params_all_weights.py"
    
            g4_model = _load_module_from_file("group4_model", model_file)
            g4_params = _load_module_from_file("group4_params", params_file)
            mcfg = g4_model.ModelConfig.llama3p2_1b_instruct()
            model_dtype = jnp.bfloat16 if args.dtype == "bfloat16" else jnp.float32
            llama_model = g4_params.create_model_from_safe_tensors(
                file_dir=str(llama_dir),
                config=mcfg,
                mesh=None,
                dtype=model_dtype,
            )
            print("loaded Group4 LoRA model:", model_file.name, params_file.name)
        else:
            loaded = load_llama_model_and_tokenizer(local_dir=llama_dir, dtype=args.dtype, use_mesh=False)
            llama_model = loaded["llama_model"]
            print("loaded Group1 base model for selective_ft")
    
        in_dim = _infer_clip_dim_from_manifest(smoke_manifest)
        out_dim = int(llama_model.config.embed_dim)
        projector = VisionProjector(in_dim=in_dim, out_dim=out_dim, rngs=nnx.Rngs(args.seed))
        projector_graphdef, _ = nnx.split(projector)
        with stage1_projector_state_path.open("rb") as f:
            projector_state = pickle.load(f)
    
        llama_graphdef, llama_state = nnx.split(llama_model)
        llama_state, replaced = _materialize_abstract_leaves(llama_state)
        if replaced:
            print(f"materialized abstract leaves in llama_state: {replaced}")
    
        if args.method == "lora":
            llama_mask = _build_lora_mask(llama_state)
            # collect selected count for logging
            selected_count = sum(1 for x in jax.tree_util.tree_leaves(llama_mask) if bool(x))
            candidate_count = len(jax.tree_util.tree_leaves(llama_mask))
            print(f"mask lora: selected_leaves={selected_count} / total_leaves={candidate_count}")
        else:
            llama_mask, candidate_count, selected_count = _build_selective_mask(
                llama_state,
                budget_pct=args.budget_pct,
                target_modules=args.target_modules,
                seed=args.seed,
                strategy=args.selection_strategy,
            )
            print(
                "mask selective_ft:",
                f"target_modules={args.target_modules}",
                f"strategy={args.selection_strategy}",
                f"budget_pct={args.budget_pct}",
                f"selected={selected_count}/{candidate_count}",
            )
    
        projector_param_count = _count_params(projector_state)
        llama_param_count = _count_params(llama_state)
        llama_trainable_count = _count_params(llama_state, mask=llama_mask)
        total_trainable = projector_param_count + llama_trainable_count
        print(
            "param counts:",
            f"projector={projector_param_count}",
            f"llama_total={llama_param_count}",
            f"llama_trainable={llama_trainable_count}",
            f"total_trainable={total_trainable}",
        )
    
        tx = optax.adamw(learning_rate=args.learning_rate, weight_decay=0.0)
        opt_state = tx.init({"projector": projector_state, "llama": llama_state})
    
        @nnx.jit
        def train_step(projector_state, llama_state, opt_state, batch):
            def loss_fn(proj_state, llm_state):
                proj = nnx.merge(projector_graphdef, proj_state)
                llm = nnx.merge(llama_graphdef, llm_state)
                mm = make_multimodal_inputs(llm, proj, batch)
                logits, _ = llm.forward_from_embeddings(
                    input_embeds=mm["input_embeds"],
                    positions=mm["positions"],
                    cache=None,
                    attention_mask=mm["attention_mask"],
                )
                return masked_cross_entropy_loss(logits, mm["labels"])
    
            loss, grads = jax.value_and_grad(loss_fn, argnums=(0, 1))(projector_state, llama_state)
            projector_grads, llama_grads = grads
            llama_grads = _zero_grads_where_mask_false(llama_grads, llama_mask)
    
            updates, opt_state_new = tx.update(
                {"projector": projector_grads, "llama": llama_grads},
                opt_state,
                {"projector": projector_state, "llama": llama_state},
            )
            new_projector_state = optax.apply_updates(projector_state, updates["projector"])
            new_llama_state = optax.apply_updates(llama_state, updates["llama"])
            return new_projector_state, new_llama_state, opt_state_new, loss
    
        manifest_rows = json.loads(smoke_manifest.read_text(encoding="utf-8"))
        if not manifest_rows:
            raise ValueError("Smoke manifest is empty.")
    
        val_count = max(1, int(round(len(manifest_rows) * float(args.val_frac))))
        if val_count >= len(manifest_rows):
            val_count = max(1, len(manifest_rows) - 1)
        train_rows = manifest_rows[:-val_count]
        val_rows = manifest_rows[-val_count:]
        if not train_rows:
            train_rows = manifest_rows[:1]
            val_rows = manifest_rows[1:] or manifest_rows[:1]
        print(f"split: train_rows={len(train_rows)} val_rows={len(val_rows)} val_frac={args.val_frac}")
        step = 0
        losses: list[float] = []
        train_history: list[dict[str, Any]] = []
        samples_seen = 0
        timer_start = time.perf_counter()
        gpu_sampler = _GPUSampler(interval_sec=2.0)
        gpu_sampler.start()
        for epoch in range(args.epochs):
            for batch in iterate_minibatches(train_rows, batch_size=args.batch_size):
                projector_state, llama_state, opt_state, loss = train_step(projector_state, llama_state, opt_state, batch)
                lv = float(loss)
                losses.append(lv)
                input_ids_obj = batch.get("input_ids") if isinstance(batch, dict) else None
                input_shape = getattr(input_ids_obj, "shape", None)
                if input_shape is not None:
                    try:
                        samples_seen += int(input_shape[0])
                    except Exception:
                        samples_seen += int(args.batch_size)
                else:
                    samples_seen += int(args.batch_size)
                if step % args.log_every == 0:
                    print(f"epoch={epoch} step={step} loss={lv:.4f}")
                train_history.append({"epoch": int(epoch), "step": int(step), "loss": lv})
                step += 1
        gpu_sampler.stop()
        wall_time_sec = max(1e-9, time.perf_counter() - timer_start)
        steps_per_sec = float(step) / wall_time_sec
        samples_per_sec = float(samples_seen) / wall_time_sec
        gpu_stats = gpu_sampler.summary()
    
        @nnx.jit
        def eval_step(projector_state, llama_state, batch):
            proj = nnx.merge(projector_graphdef, projector_state)
            llm = nnx.merge(llama_graphdef, llama_state)
            mm = make_multimodal_inputs(llm, proj, batch)
            logits, _ = llm.forward_from_embeddings(
                input_embeds=mm["input_embeds"],
                positions=mm["positions"],
                cache=None,
                attention_mask=mm["attention_mask"],
            )
            return masked_cross_entropy_loss(logits, mm["labels"])
    
        val_losses: list[float] = []
        for batch in iterate_minibatches(val_rows, batch_size=args.batch_size):
            val_losses.append(float(eval_step(projector_state, llama_state, batch)))
        val_loss = float(sum(val_losses) / max(1, len(val_losses)))
        print(f"validation: val_loss={val_loss:.4f} batches={len(val_losses)}")
        val_history = [{"epoch": int(args.epochs - 1), "val_loss": val_loss}]
    
        run_id = (
            f"{args.method}"
            f"_lora-{args.lora_variant if args.method == 'lora' else 'na'}"
            f"_target-{args.target_modules}"
            f"_rows-{args.max_rows}"
            f"_seed-{args.seed}"
        )
        out_projector = out_dir / f"{run_id}_projector.pkl"
        out_llama = out_dir / f"{run_id}_llama.pkl"
        out_metrics = out_dir / f"{run_id}_metrics.json"
    
        if (out_projector.exists() or out_llama.exists() or out_metrics.exists()) and not args.overwrite:
            print("outputs already exist; pass --overwrite to replace.")
            return 0
    
        with out_projector.open("wb") as f:
            pickle.dump(projector_state, f)
        with out_llama.open("wb") as f:
            pickle.dump(llama_state, f)
    
        metrics = {
            "run_id": run_id,
            "method": args.method,
            "lora_variant": args.lora_variant if args.method == "lora" else None,
            "target_modules": args.target_modules,
            "selection_strategy": args.selection_strategy if args.method == "selective_ft" else None,
            "budget_pct": args.budget_pct if args.method == "selective_ft" else None,
            "smoke_rows": args.max_rows,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "dtype": args.dtype,
            "steps": step,
            "loss_first": losses[0] if losses else None,
            "loss_last": losses[-1] if losses else None,
            "val_loss": val_loss,
            "projector_params": projector_param_count,
            "llama_params_total": llama_param_count,
            "llama_params_trainable": llama_trainable_count,
            "trainable_params_total": total_trainable,
            "trainable_params_millions": total_trainable / 1_000_000.0,
            "selected_leaves": selected_count,
            "candidate_leaves": candidate_count,
            "wall_time_sec": wall_time_sec,
            "steps_per_sec": steps_per_sec,
            "samples_per_sec": samples_per_sec,
            "gpu_stats": gpu_stats,
            "projector_state_path": str(out_projector),
            "llama_state_path": str(out_llama),
        }
        out_metrics.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        print("saved metrics:", out_metrics)
        tracker.write_json("peft/metrics.json", metrics)
        tracker.write_csv("peft/train_history.csv", train_history)
        tracker.write_csv("peft/val_history.csv", val_history)
        _try_plot_series(
            tracker.run_dir / "peft" / "fig_train_loss.png",
            [float(r["step"]) for r in train_history],
            [float(r["loss"]) for r in train_history],
            "Group4 PEFT Train Loss",
            "step",
            "loss",
        )
        _try_plot_series(
            tracker.run_dir / "peft" / "fig_loss_train_vs_step.png",
            [float(r["step"]) for r in train_history],
            [float(r["loss"]) for r in train_history],
            "Group4 PEFT Train Loss",
            "step",
            "loss",
        )
        _try_plot_series(
            tracker.run_dir / "peft" / "fig_val_loss.png",
            [float(r["epoch"]) for r in val_history],
            [float(r["val_loss"]) for r in val_history],
            "Group4 PEFT Val Loss",
            "epoch",
            "val_loss",
        )
        _try_plot_series(
            tracker.run_dir / "peft" / "fig_loss_val_vs_epoch_or_evalstep.png",
            [float(r["epoch"]) for r in val_history],
            [float(r["val_loss"]) for r in val_history],
            "Group4 PEFT Val Loss",
            "epoch",
            "val_loss",
        )
        print(
            "performance:",
            f"wall_time_sec={wall_time_sec:.2f}",
            f"steps_per_sec={steps_per_sec:.3f}",
            f"samples_per_sec={samples_per_sec:.3f}",
        )
        if gpu_stats is not None:
            print(
                "gpu:",
                f"util_avg={gpu_stats['gpu_util_avg_pct']:.1f}%",
                f"util_max={gpu_stats['gpu_util_max_pct']:.1f}%",
                f"mem_max={gpu_stats['gpu_mem_used_max_mb']:.1f}MB",
                f"power_avg={gpu_stats['gpu_power_avg_w']:.1f}W",
            )

        if args.append_manual_results:
            manual_path = Path(cfg["group4_outputs"]["results_manual_json"])
            if manual_path.exists():
                manual = json.loads(manual_path.read_text(encoding="utf-8"))
            else:
                manual = {"results": []}
            results = manual.setdefault("results", [])

            results = [r for r in results if r.get("experiment_id") != run_id]
            results.append(
                {
                    "experiment_id": run_id,
                    "method": args.method,
                    "lora_variant": args.lora_variant if args.method == "lora" else None,
                    "target_modules": args.target_modules,
                    "selection_strategy": args.selection_strategy if args.method == "selective_ft" else None,
                    "budget_pct": args.budget_pct if args.method == "selective_ft" else None,
                    "trainable_params_millions": total_trainable / 1_000_000.0,
                    "smoke_loss_first": losses[0] if losses else None,
                    "smoke_loss_last": losses[-1] if losses else None,
                    "win_rate_vs_baseline": 0.0,
                    "val_loss": val_loss,
                    "notes": "Auto-appended from run_group4_peft_smoke.py; replace win_rate_vs_baseline after pairwise eval.",
                }
            )
            manual["results"] = results
            manual_path.parent.mkdir(parents=True, exist_ok=True)
            manual_path.write_text(json.dumps(manual, ensure_ascii=False, indent=2), encoding="utf-8")
            print("updated manual results:", manual_path)

        summary = tracker.finalize()
        print("Run summary:", summary["run_dir"])
        return 0
    finally:
        stdio.stop()


if __name__ == "__main__":
    raise SystemExit(main())
