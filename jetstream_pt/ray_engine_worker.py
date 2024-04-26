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

"""Implement Jet Engine API."""

import queue
from typing import Any, List, Optional, Tuple, Union
import threading
import functools
import humanize

from etils import epath
from safetensors import safe_open
from flax import struct
import jax
from jax import numpy as jnp
import torch
import numpy as np
import ray


from jetstream.engine import engine_api, tokenizer_pb2, token_utils
import torch_xla2
from jetstream_pt.third_party.llama2 import model_exportable, model_args

from jetstream_pt import cache_manager
from jetstream_pt import quantize
from jetstream_pt.environment import JetEngineEnvironment, JetEngineEnvironmentData

from torch.utils import _pytree as pytree
from jax.experimental import multihost_utils




Mesh = jax.sharding.Mesh
P = jax.sharding.PartitionSpec

Params = jax.Array
PrefillInputs = np.ndarray

@struct.dataclass
class Prefix:
  token: jax.Array  # [1, seqlen]
  caches: List[Tuple[jax.Array, jax.Array]]
  seq_len: int  # true seqlen front pad
  

@struct.dataclass
class DecodeState:
  tokens: jax.Array   # [batch_size, seqlen]
  caches: List[Tuple[jax.Array, jax.Array]]
  cache_scales: List[Tuple[jax.Array, jax.Array]]  # only present in quantized kv
  current_position: int
  lens: jax.Array # [batch_size, 1]
  input_pos: jax.Array # [batch_size, 1] input pos for each slot
  mask: jax.Array # [batch_size, seqlen] -inf for invalid; 0 for valid

@struct.dataclass
class DecodeStateLogits:
  logits: jax.Array   # [batch_size, seqlen]
  caches: List[Tuple[jax.Array, jax.Array]]
  cache_scales: List[Tuple[jax.Array, jax.Array]]  # only present in quantized kv
  current_position: int
  lens: jax.Array # [batch_size, 1]
  input_pos: jax.Array # [batch_size, 1] input pos for each slot
  mask: jax.Array # [batch_size, seqlen] -inf for invalid; 0 for valid  

# NOTE model specific


@ray.remote
# pylint: disable-next=all
class PyTorchEngineRayWorker:
  """Wraps functions to the Jet Engine API format."""

  # pylint: disable-next=all
  def __init__(
      self,
      tokenizer_path: str,
      ckpt_path: Optional[str] = None,
      samples_per_slot: int = 1,
      bf16_enable: bool = False,
      param_size: str = "7b",
      context_length: int = 1024,
      batch_size: int = 1,
      max_decode_length: int = 4096,
      model_name="llama",
      quantize_weights=False,
      quantize_kv=False,
      max_cache_length=1024,
  ):

    print("------------------------> inside engine worker")
    jax.config.update("jax_default_prng_impl", "unsafe_rbg")
    jax.config.update("jax_dynamic_shapes", False)
    # Pytorch exports has int64 constants.
    # jax.config.update('jax_enable_x64', True)
    jax.config.update("jax_traceback_filtering", "off")
    torch_dtype = torch.bfloat16 if bf16_enable else torch.float32
    torch.set_default_dtype(torch_dtype)
    print("------------------------> jax get devices")
    self.devices = jax.devices()
    device_count = jax.device_count()
    local_device_count = jax.local_device_count()
    print(
        f"---Jax device_count:{device_count}, local_device_count{local_device_count} "
    )

    checkpoint_format = ""
    checkpoint_path = ""

    if not ckpt_path or ckpt_path is None:
      print("WARNING: Using random weights instead of checkpoints.")
    elif ".safetensors" in ckpt_path:
      checkpoint_format = "safetensors"
      checkpoint_path = ckpt_path
    elif ".pth" in ckpt_path:
      raise NotImplementedError(
          "Loading from Pytorch raw checkpoint is not supported!"
      )
    else:
      path = (
          epath.Path(ckpt_path) if ckpt_path and ckpt_path is not None else ""
      )
      if not path.exists():
        raise ValueError(f"Checkpoint path {ckpt_path} not exists!")
      paths = list(path.glob("*.safetensors"))
      assert (
          len(paths) == 1
      ), f"Expects 1 *.safetensors in the checkpoint dir, see {len(paths)}"
      checkpoint_format = "safetensors"
      checkpoint_path = paths[0]

    env_data = JetEngineEnvironmentData(
        tokenizer_path=tokenizer_path,
        checkpoint_path=checkpoint_path,
        checkpoint_format=checkpoint_format,
        model_type="llama-2-" + param_size,
        batch_size=batch_size,
        max_decode_length=max_decode_length,
        max_input_sequence_length=context_length,
        enable_weight_quantization=quantize_weights,
        enable_kv_quantization=quantize_kv,
        cache_sequence_length=max_cache_length,
        bf16_enable=bf16_enable,
    )
    env = JetEngineEnvironment(env_data)

    tokenizer = token_utils.load_vocab(tokenizer_path)
    pt_model = None
    if model_name == "llama":
      args = model_args.get_model_args(
          param_size,
          context_length,
          batch_size,
          tokenizer.vocab_size,
          bf16_enable,
      )
      args.device = "meta"
      args.quantize = quantize_weights
      pt_model = model_exportable.Transformer(args, env)

      num_params_size = 0
      num_params = 0
      for _, v in pt_model.state_dict().items():
        num_params += 1
        num_params_size += np.prod(v.shape) * (1 if v.dtype == jnp.int8 else 2)
    print("Number of param Gbytes:", num_params_size / (1 << 30))
    print("Number of param: ", num_params)

    self.decode_state = None
    self.prefix_queue = queue.Queue()
    self.pt_model = pt_model
    self.env = env
    self.default_dtype = jnp.bfloat16 if env.bf16_enable else jnp.float32

    # NOTE: this is llama2 specific now.
    self.param = pt_model.params

    self.y_sharding = env.sharding_by_axis(1)
    self.x_sharding = env.sharding_by_axis(0)
    self.replicated = env.sharding_by_axis(-1)  # replicated
    self.cache_sharding = self.y_sharding

    self._compiled_call_model_prefill = jax.jit(
        self._call_model_prefill,
        donate_argnums=(1, 2),
        out_shardings=(self.replicated, self.cache_sharding),
    )
    self._compiled_insert = jax.jit(
        self._insert,
        donate_argnums=(0, 1),
        out_shardings=(
            self.replicated,
            self.cache_sharding,
            self.replicated,
            self.replicated,
            self.replicated,
            self.replicated,
            self.replicated,
        ),
    )

    self._compiled_call_model_generate = jax.jit(
        self._call_model_generate,
        donate_argnums=(2, 3, 4, 5, 6, 7),
        out_shardings=(
            self.replicated,
            self.cache_sharding,
            self.replicated,
            self.replicated,
            self.replicated,
            self.replicated,
            self.replicated,
        ),
    )
    self._lock = threading.RLock()

  # pylint: disable-next=all
  def sharding_by_name(self, name):

    # This allows easier way to edit shardings
    """
    for key, val in self.env._data.experimental_sharding_axis_override.items():
      if name.endswith(key):
        return self.env.sharding_by_axis(val)
    """

    if "weight_scaler" in name:
      return self.x_sharding
    if "tok_embeddings." in name:
      return self.y_sharding
    if "attention." in name:
      if "wo" in name:
        return self.y_sharding
      return self.x_sharding
    if "feed_forward." in name:
      if "w2" in name:
        return self.y_sharding
      return self.x_sharding
    if "output" in name:
      return self.x_sharding
    return self.replicated

  # pylint: disable-next=all
  def init_decode_state(
      self,
  ) -> DecodeState:
    caches_obj = self.env.make_caches_generate()
    caches = [c.state() for c in caches_obj]
    scalers = []
    if self.env.enable_kv_quantization:
      scalers = [c.scalers() for c in caches_obj]
    return DecodeState(
        jnp.zeros((self.env.batch_size, 1), dtype=jnp.int32),
        caches,
        scalers,
        self.env.max_input_sequence_length,
        jnp.zeros((self.env.batch_size, 1), dtype=jnp.int32),
        jnp.zeros((self.env.batch_size,), dtype=jnp.int32),  # input pos
        jnp.full(
            (self.env.batch_size, self.env.cache_sequence_length),
            float("-inf"),
            dtype=self.default_dtype,
        ),
    )