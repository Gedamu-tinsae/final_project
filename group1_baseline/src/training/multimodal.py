"""Multimodal input assembly from Group 1 notebook."""

import jax.numpy as jnp


def make_multimodal_inputs(llama_model, projector, batch):
    """Concatenate projected visual tokens and text embeddings."""
    vision_feats = jnp.array(batch["vision_feats"])
    input_ids = jnp.array(batch["input_ids"])
    labels = jnp.array(batch["labels"])

    vis_embeds = projector(vision_feats)
    txt_embeds = llama_model.embedder.encode(input_ids)
    input_embeds = jnp.concatenate([vis_embeds, txt_embeds], axis=1)

    bsz, n_vis, _ = vis_embeds.shape
    _, tlen = input_ids.shape
    seq_len = n_vis + tlen

    vis_labels = -100 * jnp.ones((bsz, n_vis), dtype=jnp.int32)
    full_labels = jnp.concatenate([vis_labels, labels], axis=1)

    positions = jnp.broadcast_to(jnp.arange(seq_len)[None, :], (bsz, seq_len))

    attention_mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=bool))
    attention_mask = jnp.broadcast_to(attention_mask[None, :, :], (bsz, seq_len, seq_len))

    return {
        "input_embeds": input_embeds,
        "labels": full_labels,
        "positions": positions,
        "attention_mask": attention_mask,
    }

