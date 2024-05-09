from typing import Any, Optional, Union

import numpy as np
import jax
import ray
from ray.util.accelerators import tpu

from jetstream.engine import engine_api, tokenizer_pb2
from jetstream_pt.ray_worker import PyTorchRayWorker


Params = Any
Prefix = Any
DecodeState = Any


class LocalRayEngine(engine_api.Engine):
  """Ray engine master to orchestrate requests and collect token response"""

  def __init__(
      self, engine_worker, tokenizer_path, context_length, batch_size
  ):
    self.engine_worker = engine_worker
    self.tokenizer_path = tokenizer_path
    self.context_length = context_length
    self.batch_size = batch_size

  # pylint: disable-next=all
  def load_params(self) -> Params:
    return self.engine_worker.load_params_ray()

  # pylint: disable-next=all
  def init_decode_state(
      self,
  ) -> DecodeState:
    return self.engine_worker.init_decode_state_ray()

  def prefill(
      self,
      *,
      params: Any,  # Weights
      existing_prefix: Optional[Prefix] = None,
      padded_tokens: np.ndarray,  # PrefillInputs[np.ndarray],
      true_length: int,
  ) -> Prefix:

    return self.engine_worker.prefill_ray(    params=params,
          existing_prefix=existing_prefix,
          padded_tokens=padded_tokens,
          true_length=true_length,)

  def insert(
      self,
      prefix: Prefix,
      decode_state: DecodeState,
      slot: int,
  ) -> DecodeState:
    return self.engine_worker.insert_ray(
          prefix=prefix, decode_state=decode_state, slot=slot
      )

  def generate(
      self, params: Any, decode_state: DecodeState
  ) -> tuple[None, engine_api.ResultTokens]:
    return self.engine_worker.generate_ray(
          params=params, decode_state=decode_state
      )

  # pylint: disable-next=all
  def get_tokenizer(self) -> tokenizer_pb2.TokenizerParameters:
    # pylint: disable-next=all
    return tokenizer_pb2.TokenizerParameters(path=self.tokenizer_path)

  @property
  def max_concurrent_decodes(self) -> int:
    return self.batch_size

  @property
  def samples_per_slot(self) -> int:
    return 1

  @property
  def max_prefill_length(self) -> int:
    return self.context_length

  @property
  def colocated_cpus(self) -> Union[list[engine_api.CpuDevices], None]:
    return jax.devices("cpu")[0]

  def get_prefix_destination_sharding(self) -> Prefix:
    "No implementation"
    return None

  @property
  def mesh(self):
    "No implementation"
    return None


# pylint: disable-next=all
def create_pytorch_ray_local_engine(
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
    sharding_config=None,
) -> LocalRayEngine:


  engine_worker = PyTorchRayWorker(
      tokenizer_path=tokenizer_path,
      ckpt_path=ckpt_path,
      samples_per_slot=samples_per_slot,
      bf16_enable=bf16_enable,
      param_size=param_size,
      context_length=context_length,
      batch_size=batch_size,
      max_decode_length=max_decode_length,
      model_name=model_name,
      quantize_weights=quantize_weights,
      quantize_kv=quantize_kv,
      max_cache_length=max_cache_length,
      sharding_config=sharding_config,
  )
  engine_master = LocalRayEngine(
      engine_worker=engine_worker,
      tokenizer_path=tokenizer_path,
      context_length=context_length,
      batch_size=batch_size,
  )
  return engine_master