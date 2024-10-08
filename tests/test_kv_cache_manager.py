import unittest

import jax
import numpy as np
import jax.numpy as jnp
import torch

from jetstream_pt.third_party.llama import model_args
from jetstream_pt import environment
from jetstream_pt.page_attention_manager import PageAttentionManager
from jetstream_pt.cache_manager import PageKVCacheGenerate, KVCachePrefill
from jetstream_pt import torchjax
from absl.testing import parameterized

P = jax.sharding.PartitionSpec


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
    mesh = jax.sharding.Mesh(np.array(jax.devices()), axis_names=("x",))
    replicated = jax.sharding.NamedSharding(mesh, P())
    env.sharding = replicated
    return env, config

  def test_page_attention_update(self):
    jax.config.update("jax_platform_name", "cpu")
    print(f"---------> {jax.devices()}")

    env, _ = self._make_env()

    pam = PageAttentionManager(
        batch_size=5,
        paged_attention_total_num_pages=20,
        paged_attention_page_size=4,
        max_pages_per_sequence=4,
    )
    shape = (1, 20, 4, 2)
    decode_caches = []
    decode_caches.append(
        PageKVCacheGenerate.empty(shape=shape, device=None, env=env)
    )
    decode_caches = [c.state() for c in decode_caches]

    self.cache_sharding = env.cache_sharding

    def _insert_prefill(seq_len, dim, slot):
      prefill_chache = KVCachePrefill()
      k, v = jnp.arange(seq_len * dim), jnp.arange(seq_len * dim)
      k, v = jnp.reshape(k, (1, 1, seq_len, dim)), jnp.reshape(
          k, (1, 1, seq_len, dim)
      )
      prefill_chache.update(k, v, 0)
      prefill_caches = [prefill_chache]
      prefill_caches = [c.state() for c in prefill_caches]
      num_pages, update_indexes = pam.reserve_pages_insert(slot, seq_len)
      _, kv_heads, _, dim = prefill_caches[0][0].shape
      tep_kv = jnp.zeros((kv_heads, num_pages * 4, dim), dtype=jnp.bfloat16)

      caches = pam.insert_prefill_cache(
          prefill_caches=prefill_caches,
          decode_caches=decode_caches,
          update_indexes=update_indexes,
          tep_kv=tep_kv,
          sharding=env.sharding,
      )

      return caches

    decode_caches = _insert_prefill(3, 2, 0)
    decode_caches = _insert_prefill(8, 2, 1)
    decode_caches = _insert_prefill(13, 2, 3)

    lens = np.asarray([3, 8, 0, 13, 0])
    pam.fill_new_pages(lens)
    np_page_token_indices = pam.get_page_token_indices(lens)
    page_token_indices = jnp.asarray(np_page_token_indices)
    page_token_indices = torchjax.to_torch(page_token_indices)

    caches_obj = [
        PageKVCacheGenerate(
            k, v, pam, page_token_indices, self.cache_sharding, env=env
        )
        for k, v in torchjax.to_torch(decode_caches)
    ]
    xk, xv = jnp.arange(-1, -11, -1).reshape(5, 1, 1, 2), jnp.arange(
        -1, -11, -1
    ).reshape(5, 1, 1, 2)
    xk = torchjax.to_torch(xk)
    xv = torchjax.to_torch(xv)
    decode_caches = caches_obj[0].update(xk, xv)
    expected = jnp.asarray([[0, 1], [2, 3], [4, 5], [-1, -2]])
    self.assertTrue(jnp.array_equal(decode_caches[0][0][0], expected))
    expected = jnp.asarray([[-3, -4], [0, 0], [0, 0], [0, 0]])
    self.assertTrue(jnp.array_equal(decode_caches[0][0][7], expected))
    expected = jnp.asarray([[24, 25], [-7, -8], [0, 0], [0, 0]])
    self.assertTrue(jnp.array_equal(decode_caches[0][0][6], expected))


if __name__ == "__main__":
  unittest.main()
