from functools import partial

import jax
from jax.experimental import pallas as pl
import jax.numpy as jnp
import numpy as np

def matmul_kernel(x_ref, y_ref, z_ref):
  z_ref[...] = x_ref[...] @ y_ref[...]

def matmul(x: jax.Array, y: jax.Array):
  return pl.pallas_call(
    matmul_kernel,
    out_shape=jax.ShapeDtypeStruct((x.shape[0], y.shape[1]), x.dtype),
    grid=(4, 2),
    in_specs=[
        pl.BlockSpec(lambda i, j: (i, 0), (x.shape[0] // 4, x.shape[1])),
        pl.BlockSpec(lambda i, j: (0, j), (y.shape[0], y.shape[1] // 2))
    ],
    out_specs=pl.BlockSpec(
        lambda i, j: (i, j), (x.shape[0] // 4, y.shape[1] // 2)
    )
  )(x, y)



def kernel(o_ref):
  assert o_ref.shape == (2,)
  o_ref[...] = jnp.full((2,), 10 * pl.program_id(1) + pl.program_id(0))
pl.pallas_call(kernel,
               jax.ShapeDtypeStruct((3, 4), dtype=np.int32),
               out_specs=pl.BlockSpec((None, 2), lambda i, j: (i, j)),
               grid=(3, 2), interpret=True)()
