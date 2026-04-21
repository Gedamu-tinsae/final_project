"""Loss functions from Group 1 notebook training cells."""

import jax
import jax.numpy as jnp


def masked_cross_entropy_loss(logits, labels):
    """Token-level cross entropy with -100 ignore mask."""
    # logits: [B, L, V], labels: [B, L]
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]

    valid = shift_labels != -100
    safe_labels = jnp.where(valid, shift_labels, 0)

    log_probs = jax.nn.log_softmax(shift_logits, axis=-1)
    token_logp = jnp.take_along_axis(log_probs, safe_labels[..., None], axis=-1).squeeze(-1)

    loss = -jnp.sum(token_logp * valid) / jnp.maximum(jnp.sum(valid), 1)
    return loss

