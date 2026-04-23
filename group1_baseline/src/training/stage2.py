"""Stage-2 training loop extracted from Group 1 notebook."""

import json

import jax
from flax import nnx
import optax

from .batching import iterate_minibatches
from .losses import masked_cross_entropy_loss
from .multimodal import make_multimodal_inputs


def train_step_stage2(
    projector_state,
    llama_state,
    opt_state,
    batch,
    projector_graphdef,
    llama_graphdef,
    tx,
):
    """One stage-2 optimization step for projector + LLaMA."""

    def loss_fn(proj_state, llm_state):
        projector = nnx.merge(projector_graphdef, proj_state)
        llama_model = nnx.merge(llama_graphdef, llm_state)

        mm = make_multimodal_inputs(llama_model, projector, batch)
        logits, _ = llama_model.forward_from_embeddings(
            input_embeds=mm["input_embeds"],
            positions=mm["positions"],
            cache=None,
            attention_mask=mm["attention_mask"],
        )
        return masked_cross_entropy_loss(logits, mm["labels"])

    loss, grads = jax.value_and_grad(loss_fn, argnums=(0, 1))(projector_state, llama_state)
    projector_grads, llama_grads = grads

    updates, opt_state = tx.update(
        {"projector": projector_grads, "llama": llama_grads},
        opt_state,
        {"projector": projector_state, "llama": llama_state},
    )

    new_projector_state = optax.apply_updates(projector_state, updates["projector"])
    new_llama_state = optax.apply_updates(llama_state, updates["llama"])
    return new_projector_state, new_llama_state, opt_state, loss


def eval_step_stage2(projector_state, llama_state, batch, projector_graphdef, llama_graphdef):
    """One stage-2 evaluation step (no gradient update)."""
    projector = nnx.merge(projector_graphdef, projector_state)
    llama_model = nnx.merge(llama_graphdef, llama_state)
    mm = make_multimodal_inputs(llama_model, projector, batch)
    logits, _ = llama_model.forward_from_embeddings(
        input_embeds=mm["input_embeds"],
        positions=mm["positions"],
        cache=None,
        attention_mask=mm["attention_mask"],
    )
    return masked_cross_entropy_loss(logits, mm["labels"])


def run_stage2_training(
    manifest_json,
    projector_state,
    llama_state,
    opt_state,
    projector_graphdef,
    llama_graphdef,
    tx,
    num_epochs=1,
    batch_size=2,
    log_every=10,
    val_frac=0.1,
):
    """Run stage-2 training over a manifest of precomputed vision features."""
    with open(manifest_json, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    if not manifest:
        raise ValueError(f"Manifest is empty: {manifest_json}")

    val_count = max(1, int(round(len(manifest) * float(val_frac))))
    if val_count >= len(manifest):
        val_count = max(1, len(manifest) - 1)
    train_rows = manifest[:-val_count]
    val_rows = manifest[-val_count:]
    if not train_rows:
        train_rows = manifest[:1]
        val_rows = manifest[1:] or manifest[:1]

    step = 0
    train_history: list[dict] = []
    val_history: list[dict] = []
    for epoch in range(num_epochs):
        for batch in iterate_minibatches(train_rows, batch_size=batch_size):
            projector_state, llama_state, opt_state, loss = train_step_stage2(
                projector_state,
                llama_state,
                opt_state,
                batch,
                projector_graphdef,
                llama_graphdef,
                tx,
            )
            if step % log_every == 0:
                print(f"epoch={epoch} step={step} loss={float(loss):.4f}")
            train_history.append(
                {
                    "epoch": int(epoch),
                    "step": int(step),
                    "loss": float(loss),
                }
            )
            step += 1

        val_losses = []
        for batch in iterate_minibatches(val_rows, batch_size=batch_size):
            val_losses.append(
                float(eval_step_stage2(projector_state, llama_state, batch, projector_graphdef, llama_graphdef))
            )
        val_loss = float(sum(val_losses) / max(1, len(val_losses)))
        print(f"epoch={epoch} val_loss={val_loss:.4f}")
        val_history.append({"epoch": int(epoch), "val_loss": val_loss})

    return projector_state, llama_state, opt_state, train_history, val_history
