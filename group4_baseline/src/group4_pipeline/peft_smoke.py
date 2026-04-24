"""Run Group4 PEFT smoke training on top of Group1 artifacts.

Supports:
- LoRA (using vendored Group4 backbone files in group4_baseline/src/group4_backbones)
- Selective fine-tuning (mask-based updates on Group1 model)
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx
import optax

PROJECT_ROOT = Path(__file__).resolve().parents[2]
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
from src.group4_pipeline.gpu_sampler import GPUSampler
from src.group4_pipeline.helpers import load_module_from_file, resolve_group4_config, try_plot_series
from src.group4_pipeline.param_masks import (
    build_lora_mask,
    build_selective_mask,
    count_params,
    materialize_abstract_leaves,
    zero_grads_where_mask_false,
)




def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Group4 PEFT smoke training.")
    p.add_argument("--config", default="configs/workflow_paths_subset_10000.json")
    p.add_argument("--method", choices=["lora", "selective_ft"], required=True)
    p.add_argument(
        "--lora-variant",
        choices=["qv", "all_weights"],
        default="qv",
        help="Which partner LoRA model variant to use when --method lora.",
    )
    p.add_argument(
        "--target-modules",
        choices=["qv", "all"],
        default="qv",
        help="Target scope for trainable parameters.",
    )
    p.add_argument("--selection-strategy", choices=["magnitude", "random"], default="magnitude")
    p.add_argument("--budget-pct", type=float, default=1.0, help="Selective FT budget percentage of candidate leaves.")
    p.add_argument("--max-rows", type=int, default=64)
    p.add_argument("--max-rows-guard", type=int, default=10000)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--val-frac", type=float, default=0.2, help="Fraction of smoke rows held out for validation.")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument(
        "--val-every-steps",
        type=int,
        default=200,
        help="Run validation every N train steps (0 disables step-level validation).",
    )
    p.add_argument(
        "--val-max-batches",
        type=int,
        default=0,
        help="Max validation batches per validation pass (0 means full validation split).",
    )
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--append-manual-results",
        action="store_true",
        help="Append this run to group4_results_manual.json with computed val_loss/win_rate/accuracy/perplexity.",
    )
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--output-root", default=str(REPO_ROOT / "outputs"))
    p.add_argument("--run-name", default="group4_peft")
    p.add_argument("--subset-token", default="subset_10000_seed42")
    p.add_argument("--allow-non-subset", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    load_dotenv_file(GROUP1_ROOT / ".env")

    cfg, cfg_path = resolve_group4_config(PROJECT_ROOT, args.config)
    tracker = RunTracker(
        group="group4",
        output_root=Path(args.output_root),
        run_name=args.run_name,
        config={"args": vars(args), "project_root": str(PROJECT_ROOT), "config": str(cfg_path) if cfg_path else "<built-in-defaults>"},
    )
    stdio = tracker.start_stdio_capture()
    try:
        print("RUN_DIR:", tracker.run_dir)
        print("CONFIG_SOURCE:", str(cfg_path) if cfg_path else "<built-in-defaults>")

        req = cfg["required_inputs"]
        stage1_manifest = Path(req["group1_stage1_manifest"])
        stage1_projector_state_path = Path(req["group1_stage1_projector_state"])
        if args.max_rows_guard > 0 and args.max_rows > args.max_rows_guard:
            raise ValueError(f"--max-rows ({args.max_rows}) exceeds guard ({args.max_rows_guard})")
        if not args.allow_non_subset and args.subset_token not in str(stage1_manifest):
            raise RuntimeError(
                f"Refusing non-subset stage1 manifest: {stage1_manifest}. "
                f"Expected token '{args.subset_token}'. Pass --allow-non-subset to override."
            )
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
    
            g4_model = load_module_from_file("group4_model", model_file)
            g4_params = load_module_from_file("group4_params", params_file)
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
        llama_state, replaced = materialize_abstract_leaves(llama_state)
        if replaced:
            print(f"materialized abstract leaves in llama_state: {replaced}")
        # Keep immutable baseline references for automatic win-rate comparison.
        baseline_projector_state = projector_state
        baseline_llama_state = llama_state
    
        if args.method == "lora":
            llama_mask = build_lora_mask(llama_state)
            # collect selected count for logging
            selected_count = sum(1 for x in jax.tree_util.tree_leaves(llama_mask) if bool(x))
            candidate_count = len(jax.tree_util.tree_leaves(llama_mask))
            print(f"mask lora: selected_leaves={selected_count} / total_leaves={candidate_count}")
        else:
            llama_mask, candidate_count, selected_count = build_selective_mask(
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
    
        projector_param_count = count_params(projector_state)
        llama_param_count = count_params(llama_state)
        llama_trainable_count = count_params(llama_state, mask=llama_mask)
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
            llama_grads = zero_grads_where_mask_false(llama_grads, llama_mask)
    
            updates, opt_state_new = tx.update(
                {"projector": projector_grads, "llama": llama_grads},
                opt_state,
                {"projector": projector_state, "llama": llama_state},
            )
            new_projector_state = optax.apply_updates(projector_state, updates["projector"])
            new_llama_state = optax.apply_updates(llama_state, updates["llama"])
            return new_projector_state, new_llama_state, opt_state_new, loss

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

        @nnx.jit
        def eval_step_metrics(projector_state, llama_state, batch):
            proj = nnx.merge(projector_graphdef, projector_state)
            llm = nnx.merge(llama_graphdef, llm_state := llama_state)
            mm = make_multimodal_inputs(llm, proj, batch)
            logits, _ = llm.forward_from_embeddings(
                input_embeds=mm["input_embeds"],
                positions=mm["positions"],
                cache=None,
                attention_mask=mm["attention_mask"],
            )
            labels = mm["labels"]
            loss = masked_cross_entropy_loss(logits, labels)
            preds = jnp.argmax(logits, axis=-1)
            mask = labels != -100
            correct = jnp.sum((preds == labels) & mask)
            token_count = jnp.sum(mask)
            return loss, correct, token_count
    
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

        def _compute_val_loss(projector_state, llama_state) -> tuple[float, int]:
            val_losses: list[float] = []
            val_batches = 0
            for batch in iterate_minibatches(val_rows, batch_size=args.batch_size):
                val_losses.append(float(eval_step(projector_state, llama_state, batch)))
                val_batches += 1
                if args.val_max_batches > 0 and val_batches >= args.val_max_batches:
                    break
            val_loss_local = float(sum(val_losses) / max(1, len(val_losses)))
            return val_loss_local, val_batches

        step = 0
        losses: list[float] = []
        train_history: list[dict[str, Any]] = []
        val_history_steps: list[dict[str, Any]] = []
        val_history_epochs: list[dict[str, Any]] = []
        samples_seen = 0
        timer_start = time.perf_counter()
        gpu_sampler = GPUSampler(interval_sec=2.0)
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
                if args.val_every_steps > 0 and step > 0 and (step % args.val_every_steps == 0):
                    val_loss_step, val_batches_step = _compute_val_loss(projector_state, llama_state)
                    print(
                        f"validation(step): epoch={epoch} step={step} val_loss={val_loss_step:.4f} "
                        f"batches={val_batches_step}"
                    )
                    val_history_steps.append(
                        {
                            "epoch": int(epoch),
                            "step": int(step),
                            "val_loss": float(val_loss_step),
                            "batches": int(val_batches_step),
                            "scope": "step",
                        }
                    )
                step += 1
        gpu_sampler.stop()
        wall_time_sec = max(1e-9, time.perf_counter() - timer_start)
        steps_per_sec = float(step) / wall_time_sec
        samples_per_sec = float(samples_seen) / wall_time_sec
        gpu_stats = gpu_sampler.summary()
    
        val_loss, val_batches = _compute_val_loss(projector_state, llama_state)
        print(f"validation(epoch-end): val_loss={val_loss:.4f} batches={val_batches}")
        val_history_epochs.append(
            {
                "epoch": int(args.epochs - 1),
                "step": int(step),
                "val_loss": float(val_loss),
                "batches": int(val_batches),
                "scope": "epoch",
            }
        )
        val_history = val_history_steps + val_history_epochs

        # Rich validation metrics (accuracy/perplexity) and win-rate vs baseline.
        val_losses_weighted = 0.0
        val_tokens_total = 0
        val_correct_total = 0
        win_count = 0
        tie_count = 0
        compare_batches = 0
        for batch in iterate_minibatches(val_rows, batch_size=args.batch_size):
            tuned_loss_b, tuned_correct_b, tuned_tokens_b = eval_step_metrics(projector_state, llama_state, batch)
            tuned_loss_v = float(tuned_loss_b)
            tuned_tokens_i = int(tuned_tokens_b)
            tuned_correct_i = int(tuned_correct_b)
            if tuned_tokens_i > 0:
                val_losses_weighted += tuned_loss_v * tuned_tokens_i
                val_tokens_total += tuned_tokens_i
                val_correct_total += tuned_correct_i

            base_loss_v = float(eval_step(baseline_projector_state, baseline_llama_state, batch))
            if tuned_loss_v < base_loss_v:
                win_count += 1
            elif tuned_loss_v == base_loss_v:
                tie_count += 1
            compare_batches += 1
            if args.val_max_batches > 0 and compare_batches >= args.val_max_batches:
                break

        val_token_accuracy = (float(val_correct_total) / float(val_tokens_total)) if val_tokens_total > 0 else None
        val_loss_token_weighted = (
            float(val_losses_weighted) / float(val_tokens_total) if val_tokens_total > 0 else val_loss
        )
        try:
            val_perplexity = float(jnp.exp(jnp.array(val_loss_token_weighted, dtype=jnp.float32)))
        except Exception:
            val_perplexity = None
        win_rate_vs_baseline = (
            float(win_count + 0.5 * tie_count) / float(compare_batches) if compare_batches > 0 else None
        )
        print(
            "validation(metrics):",
            f"token_accuracy={val_token_accuracy:.4f}" if val_token_accuracy is not None else "token_accuracy=None",
            f"perplexity={val_perplexity:.4f}" if val_perplexity is not None else "perplexity=None",
            f"win_rate_vs_baseline={win_rate_vs_baseline:.4f}" if win_rate_vs_baseline is not None else "win_rate_vs_baseline=None",
            f"compare_batches={compare_batches}",
        )
    
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
            "val_loss_token_weighted": val_loss_token_weighted,
            "val_token_accuracy": val_token_accuracy,
            "val_perplexity": val_perplexity,
            "win_rate_vs_baseline": win_rate_vs_baseline,
            "baseline_compare_batches": compare_batches,
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
        tracker.write_csv("peft/val_history_steps.csv", val_history_steps)
        tracker.write_csv("peft/val_history_epochs.csv", val_history_epochs)
        try_plot_series(
            tracker.run_dir / "peft" / "fig_train_loss.png",
            [float(r["step"]) for r in train_history],
            [float(r["loss"]) for r in train_history],
            "Group4 PEFT Train Loss",
            "step",
            "loss",
        )
        try_plot_series(
            tracker.run_dir / "peft" / "fig_loss_train_vs_step.png",
            [float(r["step"]) for r in train_history],
            [float(r["loss"]) for r in train_history],
            "Group4 PEFT Train Loss",
            "step",
            "loss",
        )
        try_plot_series(
            tracker.run_dir / "peft" / "fig_val_loss.png",
            [float(r["epoch"]) for r in val_history],
            [float(r["val_loss"]) for r in val_history],
            "Group4 PEFT Val Loss",
            "epoch",
            "val_loss",
        )
        try_plot_series(
            tracker.run_dir / "peft" / "fig_val_loss_steps.png",
            [float(r["step"]) for r in val_history_steps],
            [float(r["val_loss"]) for r in val_history_steps],
            "Group4 PEFT Val Loss (Step-level)",
            "step",
            "val_loss",
        )
        try_plot_series(
            tracker.run_dir / "peft" / "fig_val_loss_epochs.png",
            [float(r["epoch"]) for r in val_history_epochs],
            [float(r["val_loss"]) for r in val_history_epochs],
            "Group4 PEFT Val Loss (Epoch-level)",
            "epoch",
            "val_loss",
        )
        try_plot_series(
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
                    "win_rate_vs_baseline": win_rate_vs_baseline,
                    "val_loss": val_loss,
                    "val_token_accuracy": val_token_accuracy,
                    "val_perplexity": val_perplexity,
                    "wall_time_sec": wall_time_sec,
                    "steps_per_sec": steps_per_sec,
                    "samples_per_sec": samples_per_sec,
                    "notes": "Auto-appended from run_group4_peft_smoke.py.",
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
