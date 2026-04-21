"""Stage-1 training loop extracted from Group 1 notebook."""

import json

import jax
from flax import nnx
import optax

from .batching import iterate_minibatches
from .losses import masked_cross_entropy_loss
from .multimodal import make_multimodal_inputs


def train_step(projector_state, opt_state, batch, projector_graphdef, llama_state, llama_graphdef, tx):
    """One stage-1 optimization step for projector-only training."""

    def loss_fn(proj_state):
        projector = nnx.merge(projector_graphdef, proj_state)
        llama_model = nnx.merge(llama_graphdef, llama_state)

        mm = make_multimodal_inputs(llama_model, projector, batch)
        logits, _ = llama_model.forward_from_embeddings(
            input_embeds=mm["input_embeds"],
            positions=mm["positions"],
            cache=None,
            attention_mask=mm["attention_mask"],
        )
        return masked_cross_entropy_loss(logits, mm["labels"])

    loss, grads = jax.value_and_grad(loss_fn)(projector_state)
    updates, opt_state = tx.update(grads, opt_state, projector_state)
    projector_state = optax.apply_updates(projector_state, updates)
    return projector_state, opt_state, loss


def run_stage1_training(
    manifest_json,
    projector_state,
    opt_state,
    projector_graphdef,
    llama_state,
    llama_graphdef,
    tx,
    num_epochs=1,
    batch_size=2,
    log_every=10,
):
    """Run stage-1 training over a manifest of precomputed vision features."""
    with open(manifest_json, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    step = 0
    for epoch in range(num_epochs):
        for batch in iterate_minibatches(manifest, batch_size=batch_size):
            projector_state, opt_state, loss = train_step(
                projector_state,
                opt_state,
                batch,
                projector_graphdef,
                llama_state,
                llama_graphdef,
                tx,
            )
            if step % log_every == 0:
                print(f"epoch={epoch} step={step} loss={float(loss):.4f}")
            step += 1

    return projector_state, opt_state
