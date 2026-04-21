"""Vision projector module from Group 1 notebook."""

from flax import nnx
import jax.numpy as jnp


class VisionProjector(nnx.Module):
    """Map CLIP visual features [B, N_vis, D_clip] to LLaMA embed dim."""

    def __init__(self, in_dim: int = 768, out_dim: int = 2048, *, rngs: nnx.Rngs):
        self.proj = nnx.Linear(
            in_features=in_dim,
            out_features=out_dim,
            use_bias=False,
            rngs=rngs,
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.proj(x)

