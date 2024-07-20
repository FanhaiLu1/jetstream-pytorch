from collections.abc import Callable
import functools
from typing import Any
import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.pallas.ops.tpu.paged_attention.paged_attention_kernel import paged_attention
import humanize
from jax.experimental import shard_map
from flax import struct

@struct.dataclass
# pylint: disable-next=all
class DecodeState:
  start: jax.Array  # [batch_size, 1], the starting pos for each slot
  input_pos: jax.Array  # [batch_size, 1] input pos for each slot
  
@struct.dataclass
# pylint: disable-next=all
class Env:
  cache_len: int  # max cache len
  block_size: int  # ragged attention block size

def precompute_ragged_block_indices(env: Env, decode_state: DecodeState):
  """Precompute the ragged attention block indices. Ragged attention iterates the grid
  and relies on the computed grid index to skip the unnecessary blocks. The basic idea
  is to use input_pos, which is the length of each slot to determine if we should
  work on the next block of the slot or move to the next slot."""
  start = decode_state.start
  end = (start + decode_state.input_pos) % env.cache_len
  batch_size = start.shape[0]
  bk = env.block_size
  # The batch index
  b = jnp.arange(batch_size).reshape((batch_size, 1))
  num_bk = env.cache_len // env.block_size
  # The block index
  i = jnp.arange(num_bk).reshape((1, num_bk))
  i = jnp.broadcast_to(i, (batch_size, num_bk))

  start = start.reshape((batch_size, 1))
  end = end.reshape((batch_size, 1))

  am_last_batch = b == batch_size - 1
  last_good_block = jnp.where(
      start < end,
      jax.lax.div(end - 1, bk),
      jax.lax.div(env.cache_len - 1, bk),
  )

  next_b = jnp.where(am_last_batch, b, b + 1)
  next_i = jnp.where(am_last_batch, last_good_block, 0)

  # start < end, continue work on the block is there is overlap with the [start, end)
  def true_comp(b, i, bk, start, end, next_b, next_i):
    b_next = jnp.where(i * bk >= end, next_b, b)
    i_next = jnp.where(i * bk >= end, next_i, i)
    i_next = jnp.where((i + 1) * bk <= start, jax.lax.div(start, bk), i_next)
    return b_next, i_next

  # start > end, continue work on the block is there is no overlap with [end, start)
  def false_comp(b, i, bk, start, end):
    b_next = b
    i_next = jnp.where(
        jnp.logical_and(i * bk >= end, (i + 1) * bk <= start),
        jax.lax.div(start, bk),
        i,
    )
    return b_next, i_next

  true_comp_b, true_comp_i = true_comp(b, i, bk, start, end, next_b, next_i)
  false_comp_b, false_comp_i = false_comp(b, i, bk, start, end)

  b_next = jnp.where(
      start < end, true_comp_b, jnp.where(start == end, next_b, false_comp_b)
  )
  i_next = jnp.where(
      start < end, true_comp_i, jnp.where(start == end, next_i, false_comp_i)
  )
  return b_next, i_next

decode_state = DecodeState(jnp.asarray([0, 10]), 
                           jnp.asarray([9, 8]))

env = Env(16, 4)
precompute_ragged_block_indices(env, decode_state)