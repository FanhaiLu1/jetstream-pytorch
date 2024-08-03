import math
import queue
import functools
from typing import Any, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import jax.sharding as jsharding
import torch
from jetstream_pt import torchjax

class PageAttentionManager:

  def __init__(
      self,
      batch_size: int, 
      total_num_pages: int,
      page_size: int,
      max_pages_per_sequence: int,
  ):
    self.unused_pages = queue.Queue()
    self.page_indices = jnp.full((batch_size, max_pages_per_sequence), -1, dtype=jnp.int32 )
    self.lengths = jnp.zeros(batch_size, dtype=jnp.int32)
    self.page_size = page_size
    for i in range(total_num_pages):
      self.unused_pages.put(i, block=False)
         

  def reserve_pages_insert(self,
                           slot: int,
                           seq_len: int) -> Tuple[int, list]:
    self.lengths = self.lengths.at[slot].set(seq_len)
    num_pages = seq_len //self.page_size
    if seq_len % self.page_size != 0:
      num_pages = num_pages + 1 
    
    indices = [self.unused_pages.get(block=False) for _ in range(num_pages)]
    indices = [1, 3]
    self.page_indices = self.page_indices.at[slot, :num_pages].set(indices)
    return num_pages
 
  def prefill_cache_padding(self,
              caches: List[Tuple[jax.Array, jax.Array]],
              seq_len: int,
              num_pages: int) -> List[Tuple[jax.Array, jax.Array]]:
    
    pad_width = num_pages * self.page_size - seq_len
    if pad_width == 0:
      return caches
    
    caches = [(self.pad_sequences(k, pad_width), self.pad_sequences(v, pad_width)) for k, v in caches]
    return caches
  
  def insert_prefill_cache(self,
              prefill_caches: List[Tuple[jax.Array, jax.Array]],
              decode_caches: List[Tuple[jax.Array, jax.Array]],
              slot: int,
              seq_len: int,
              sharding: jsharding.Sharding) -> List[Tuple[jax.Array, jax.Array]]:
    num_pages = self.reserve_pages_insert(slot, seq_len)
    padded_caches = self.prefill_cache_padding(prefill_caches, seq_len, num_pages)
    # Reduce cache batch deminsion 
    # [kv_heads, seq_len, dim]
    squeezed_caches =  [(jnp.squeeze(k, axis=0), jnp.squeeze(v, axis=0)) for k, v in padded_caches]
    kv_heads, _, dim = squeezed_caches[0][0].shape
    # [kv_heads, num_pages, page_size, dim]
    paged_caches = [(jnp.reshape(k, (kv_heads, -1, self.page_size, dim)), jnp.reshape(v, (kv_heads, -1, self.page_size, dim))) for k, v in squeezed_caches]
    update_indexes = self.page_indices[slot, :num_pages]
    
    # @functools.partial(jax.jit, donate_argnums=(0, 1), inline=True)
    def insert(cache, new_entry):
      new_entry = new_entry.squeeze(0)
      res = cache.at[:, update_indexes, :, :].set(new_entry)
      res = jax.lax.with_sharding_constraint(res, sharding)
      return res

    caches = [
        (insert(k, newk), insert(v, newv))
        for (k, v), (newk, newv) in zip(decode_caches, paged_caches)
    ]
    
    return caches
    

  def reserve_pages_decode():
    return None


  def pad_sequences(self, array, pad_width=10):
      padding_config = [(0, 0), (0, 0), (0, pad_width), (0, 0)]  # Pad only seq_len and dim
      padded_array = jnp.pad(array, padding_config, mode='constant')
      return padded_array
  
  
  def free_pages_resource():
    return None
 
  def calculate_total_pages(page_size: int, total_token_len: int):
    total_pages = total_token_len // page_size
    
 
  