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
):
    """Run stage-2 training over a manifest of precomputed vision features."""
    with open(manifest_json, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    step = 0
    for epoch in range(num_epochs):
        for batch in iterate_minibatches(manifest, batch_size=batch_size):
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
            step += 1

    return projector_state, llama_state, opt_state
