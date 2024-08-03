import unittest
import os

import numpy as np
import jax
import jax.numpy as jnp
import torch
import torch_xla2
from torch.utils import _pytree as pytree

from jetstream_pt.engine import PyTorchEngine
from jetstream_pt.third_party.llama import model_exportable, model_args
from jetstream_pt.third_party.llama.generation_original import LlamaOriginal
from jetstream_pt import environment
from jetstream_pt.page_attention_manager import PageAttentionManager
from jetstream_pt.cache_manager import PageKVCacheGenerate, KVCachePrefill
from tests import helpers
from jetstream_pt import torchjax
from absl.testing import parameterized

class PageAttentnioTest(parameterized.TestCase):

  def _make_env(self, bf16_enable=True):
    torch_dtype = torch.bfloat16 if bf16_enable else torch.float32
    torch.set_default_dtype(torch_dtype)
    jax.config.update("jax_dynamic_shapes", False)
    jax.config.update("jax_traceback_filtering", "off")
    jax.config.update("jax_platform_name", "cpu")
    config = model_args.get_model_args("tiny", 128, 1, True)
    environment_data = environment.JetEngineEnvironmentData()
    environment_data.max_input_sequence_length = 128
    environment_data.max_input_sequence_length = 128
    environment_data.cache_sequence_length = 128
    environment_data.bf16_enable = bf16_enable
    environment_data.model_type = "llama-2-tiny"
    environment_data.batch_size = 3
    environment_data.num_layers = config.n_layers
    environment_data.cache_shape = (
        1,
        config.n_kv_heads,
        environment_data.cache_sequence_length,
        config.dim // config.n_heads,
    )
    env = environment.JetEngineEnvironment(environment_data)
    env.apply_sharding = lambda *args, **kwargs: None  # don't shard on cpu
    return env, config  

  # def test_1_layer_prefill_insert(self):
    
  #   env,_=self._make_env()

  #   pam = PageAttentionManager(batch_size=3, total_num_pages=20, page_size=4, max_pages_per_sequence=4)
  #   shape = (1, 6, 4, 2)
  #   decode_caches = []
  #   decode_caches.append(
  #           PageKVCacheGenerate.empty(
  #               shape=shape, device=None, env=env
  #           )
  #       )
  #   decode_caches = [c.state() for c in decode_caches]
    

    
  #   # batch:1, kv_head: 1, seq:3, dim: 2
  #   prefill_chache = KVCachePrefill()
  #   k, v = jnp.arange(6), jnp.arange(6)
  #   k, v = jnp.reshape(k, (1, 1, 3, 2)), jnp.reshape(k, (1, 1, 3, 2))
  #   prefill_chache.update(k, v, 0)
  #   prefill_caches = [prefill_chache]
  #   prefill_caches = [c.state() for c in prefill_caches]
    
  #   pam.insert_prefill_cache(prefill_caches, decode_caches, 1, 3)

  def test_1_layer_prefill_insert_multiple_pages(self):
    
    jax.config.update("jax_platform_name", "cpu")
    print(f"---------> {jax.devices()}")
    
    env,_=self._make_env()

    pam = PageAttentionManager(batch_size=3, total_num_pages=20, page_size=4, max_pages_per_sequence=4)
    shape = (1, 6, 4, 2)
    decode_caches = []
    decode_caches.append(
            PageKVCacheGenerate.empty(
                shape=shape, device=None, env=env
            )
        )
    decode_caches = [c.state() for c in decode_caches]
    
    self.cache_sharding = env.cache_sharding
    
    # batch:1, kv_head: 1, seq:3, dim: 2
    prefill_chache = KVCachePrefill()
    k, v = jnp.arange(12), jnp.arange(12)
    k, v = jnp.reshape(k, (1, 1, 6, 2)), jnp.reshape(k, (1, 1, 6, 2))
    prefill_chache.update(k, v, 0)
    prefill_caches = [prefill_chache]
    prefill_caches = [c.state() for c in prefill_caches]
    
    pam.insert_prefill_cache(prefill_caches, decode_caches, 1, 6, env.cache_sharding)    
    
if __name__ == "__main__":
  unittest.main()
    