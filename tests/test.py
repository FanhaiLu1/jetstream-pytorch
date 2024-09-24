from functools import partial 

import numpy as np

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P 
from jax.experimental import mesh_utils
from jax.experimental.shard_map import shard_map

devices = mesh_utils.create_device_mesh((4, 2))
mesh = Mesh(devices, axis_names=('i', 'j'))

a = jnp.arange(4 * 2).reshape(4, 2)
b = jnp.arange(2 * 4).reshape(2, 4)

@partial(shard_map, mesh=mesh, in_specs=(P('i', 'j'), P('j', None)), out_specs=P('i', None))
def matmual_basic(a_block, b_block):
  c_partial_sum = jnp.dot(a_block, b_block)
  c_block = jax.lax.psum(c_partial_sum, 'j')
  print(c_block)
  return c_block


@partial(shard_map, mesh=mesh, in_specs=(P('i', 'j'), P('j', None)), out_specs=P('i', 'j'))
def matmual_reduce_scatter(a_block, b_block):
  c_partial_sum = jnp.dot(a_block, b_block)
  c_block = jax.lax.psum_scatter(c_partial_sum, 'j', scatter_dimension=1, tiled=True)
  print(c_block)
  return c_block



@partial(shard_map, mesh=mesh, in_specs=P('i', 'j'), out_specs=P(None, None))
def f3(x_block):
  return jax.lax.psum(x_block, ('i', 'j'))

x = np.arange(4 * 2).reshape(4, 2)
print(x)
y3 = f3(x)
print(y3)
  