# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections.abc import Callable
import math
import functools
from typing import Any
import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.pallas.ops.tpu.paged_attention.paged_attention_kernel import paged_attention
from jax.experimental import shard_map
import jetstream_pt.attention_kernel as ak
from jetstream_pt import torchjax



DEFAULT_MASK_VALUE = -0.7 * float(np.finfo(np.dtype("float32")).max)
P = jax.sharding.PartitionSpec




multi_page_grouped_query_attention_fully_pipelined = (
    paged_attention
)

def dense_attention(xq, keys, values, mask=None):

  bsz, _, _, head_dim = xq.shape

  scores = jnp.einsum("ikjl,ikml->ikjm", xq, keys) / math.sqrt(head_dim)

  if mask is not None:
    scores = scores + mask  # (bs, n_local_heads, seqlen, max_seqlen)

  scores = jax.nn.softmax(scores, axis=-1)

  output = jnp.einsum("ikjm,ikml->ikjl", scores, values)
  return output


def shard_kv_heads(
    paged_attention_impl: Callable[..., Any],
    mesh: jax.sharding.Mesh,
    kv_head_mesh_axis_name: str,
):
  """Shards GQA PagedAttention along KV heads."""
  in_specs = (
      P(None, kv_head_mesh_axis_name, None),  # q
      P(kv_head_mesh_axis_name, None, None, None),  # k
      P(kv_head_mesh_axis_name, None, None, None),  # v
      P(),  # lengths
      P(),  # page_indices
  )

  out_specs = P(None, kv_head_mesh_axis_name, None)  # q

  return jax.jit(
      shard_map.shard_map(
          paged_attention_impl,
          mesh=mesh,
          in_specs=in_specs,
          out_specs=out_specs,
          check_rep=False,
      )
  )


def get_data(step:int=0):
  n_heads = 8
  total_num_pages = 256
  page_size = 64
  xq = get_xq(1, n_heads, step)
  keys = get_keys(n_heads, total_num_pages, page_size, step)
  values = get_values(n_heads, total_num_pages, page_size, step)
  seq_lens = get_seq_lens(step)
  page_indices = get_page_indices()
  return xq, keys, values, seq_lens, page_indices

def get_dense_data(xq, keys, values, seq_lens, page_indices):
  n_heads = 8
  total_num_pages = 256
  page_size = 64
  batch_size = 1
  dim = 128
  
  xq = jnp.expand_dims(xq, 2)
  
  seq_len = seq_lens[0]
  num_pages = (seq_len + page_size - 1) // page_size
  page_indices = page_indices[0, 0:num_pages]
  
  keys = keys[:, page_indices, :, :]
  keys = keys.reshape(batch_size, n_heads, num_pages * page_size, dim)
  values = values[:, page_indices, :, :]
  values = values.reshape(batch_size, n_heads, num_pages * page_size, dim)
  
  mask = jnp.full(
    (batch_size,num_pages * page_size),
    float("-inf"),
    dtype=jnp.float32,
  )
  batch = jnp.arange(batch_size)
  mask = mask.at[batch, 0:seq_len].set(0)
  return xq, keys, values, mask
  
         

def get_xq(batch, head, step):
  # xq[0:1,0:1,0:2]
  xq1= [[[0.976562, 0.335938]]]
  xq2 = [[[0.494141, 1.66406]]]
  xq = xq1 if step == 0 else xq2
  xq = jnp.asarray(xq, dtype=jnp.float32)
  xq = jnp.tile(xq, (batch, head, 64))
  return xq
  
def get_keys(head, pages, page_size, step):
  # keys[0, 0, 0:9, 0:2]
  key1 = [[[[-0.419922, -0.194336],
        [-0.0270996, -0.211914],
        [-0.234375, -0.178711],
        [0.279297, 0.699219],
        [0.0114746, -0.138672],
        [-0.886719, -0.296875],
        [0.106934, -0.269531],
        [0.133789, -0.363281],
        [0.953125, 0.165039],]]]  
        
  # keys[0, 0, 0:10, 0:2]
  key2 = [[[-0.419922, -0.194336],
      [-0.0270996, -0.211914],
      [-0.234375, -0.178711],
      [0.279297, 0.699219],
      [0.0114746, -0.138672],
      [-0.886719, -0.296875],
      [0.106934, -0.269531],
      [0.133789, -0.363281],
      [0.953125, 0.165039],
      [-0.0247803, 0.197266]]]
  
  key = key1 if step == 0 else key2
  key = jnp.asarray(key)
  key = jnp.tile(key, (1, 1, 1, 64))
  
  r = jnp.zeros((head, pages, page_size, 128),  dtype=jnp.float32)
  r = r.at[:, 0:1, 0:key.shape[2], :].set(key)
  return r
  
def get_values(head, pages, page_size, step):
  # values[0:1, 0:1, :9, 0:2]
  v1 = [[[[-0.000770569, -0.019043],
        [-0.00793457, -0.00564575],
        [0.00234985, -0.0113525],
        [0.00311279, -0.00210571],
        [0.012085, 0.00242615],
        [-0.00665283, -0.00382996],
        [-0.000762939, -0.00738525],
        [-0.00811768, 0.00646973],
        [-0.00352478, -0.00128174]]]]   
  
  # values[0:1, 0:1, :10, 0:2]
  v2 = [[-0.000770569, -0.019043],
      [-0.00793457, -0.00564575],
      [0.00234985, -0.0113525],
      [0.00311279, -0.00210571],
      [0.012085, 0.00242615],
      [-0.00665283, -0.00382996],
      [-0.000762939, -0.00738525],
      [-0.00811768, 0.00646973],
      [-0.00352478, -0.00128174],
      [0.0014801, -0.00915527]] 
  v = v1 if step == 0 else v2
  v = jnp.asarray(v)  
  v = jnp.tile(v, (1, 1, 1, 64))
  
  r = jnp.zeros((head, pages, page_size, 128), dtype=jnp.float32)
  r = r.at[:, 0:1, 0:v.shape[2], :].set(v)
  return r     


def get_seq_lens(step):
  lens = [9] if step == 0 else [10]
  return jnp.asarray(lens, dtype=jnp.int32)

def get_page_indices():
  #(1, 32)
  indices = [ 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]  
  return jnp.asarray(indices, dtype=jnp.int32).reshape(-1, 32)



def get_output():
  # (1, 1, 2)
  output = [[[-0.00352478, -0.001297]]]
  return jnp.asarray(output)


def test_sharded_multi_page_grouped_query_attention( ):
    xq, keys, values, seq_lens, page_indices = get_data(0)
 

    page_size = 64
    block_size = 512
  

    print(f"mesh shape:{mesh.shape}")
    q_pspec = jax.sharding.NamedSharding(mesh, P(None, 'x', None))
    kv_pspec = jax.sharding.NamedSharding(mesh, P('x', None, None, None))
    q_sharded = jax.device_put(xq, q_pspec)
    k_pages_sharded = jax.device_put(keys, kv_pspec)
    v_pages_sharded = jax.device_put(values, kv_pspec)

    paged_attention_impl = functools.partial(
        multi_page_grouped_query_attention_fully_pipelined,
        pages_per_compute_block=block_size // page_size,
    )
    sharded_paged_attention_impl = shard_kv_heads(
        paged_attention_impl,
        mesh,
        kv_head_mesh_axis_name='x',
    )

    def run():
      o_sharded = sharded_paged_attention_impl(
          q_sharded,
          k_pages_sharded,
          v_pages_sharded,
          seq_lens,
          page_indices,
      )
      return o_sharded

    with mesh:
      return run()  # warm up
    
def test_dense_attention():
    xq, keys, values, seq_lens, page_indices = get_data(0)    
    xq, keys, values, mask = get_dense_data(xq, keys, values, seq_lens, page_indices)
    return dense_attention( xq, keys, values, mask = mask)

def test_torch_dense_attention():
    xq, keys, values, seq_lens, page_indices = get_data(0)    
    xq, keys, values, mask = get_dense_data(xq, keys, values, seq_lens, page_indices)
    
    xq, keys, values, mask = torchjax.to_torch((xq, keys, values, mask))
    
    return ak.dense_attention(xq, keys, values, mask=mask)


def test_sharded_multi_page_grouped_query_attention_with_saved_data( ):
    loaded_data = jnp.load("/home/fanhai/data/test/paged_attention1.npy.npz")
    xq = loaded_data['xq']
    keys = loaded_data['keys']
    values = loaded_data['values']
    seq_lens = loaded_data['seq_lens']
    page_indices = loaded_data['page_indices']
    output = loaded_data['output']
 

    page_size = 64
    block_size = 512
  

    print(f"mesh shape:{mesh.shape}")
    q_pspec = jax.sharding.NamedSharding(mesh, P(None, 'x', None))
    kv_pspec = jax.sharding.NamedSharding(mesh, P('x', None, None, None))
    q_sharded = jax.device_put(xq, q_pspec)
    k_pages_sharded = jax.device_put(keys, kv_pspec)
    v_pages_sharded = jax.device_put(values, kv_pspec)

    paged_attention_impl = functools.partial(
        multi_page_grouped_query_attention_fully_pipelined,
        pages_per_compute_block=block_size // page_size,
    )
    sharded_paged_attention_impl = shard_kv_heads(
        paged_attention_impl,
        mesh,
        kv_head_mesh_axis_name='x',
    )

    def run():
      o_sharded = sharded_paged_attention_impl(
          q_sharded,
          k_pages_sharded,
          v_pages_sharded,
          seq_lens,
          page_indices,
      )
      return o_sharded

    with mesh:
      return run()  # warm up  
  

jax.config.update("jax_traceback_filtering", "off")    
mesh = jax.sharding.Mesh(np.array(jax.devices()), axis_names=('x',))
output = test_sharded_multi_page_grouped_query_attention()
# output = test_dense_attention()
#output = test_torch_dense_attention()–
# output = test_sharded_multi_page_grouped_query_attention_with_saved_data()

print(output.shape)
print(output)
