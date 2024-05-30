from collections import defaultdict
from typing import Any, Iterable, Optional, Union

import numpy as np
import ray
from ray.util.accelerators import tpu

from jetstream.engine import engine_api, tokenizer_pb2
from jetstream_pt.ray_worker import PyTorchRayWorker

Params = Any
Prefix = Any
DecodeState = Any
NpPrefix = Any


class PyTorchLocalEngine(engine_api.Engine):
  """Ray PyTorch Local Engine to test ray_woker performance with ray remote.
  To use this class, please remove @ray.remote from ray_worker.
  """

  def __init__(
      self,
      engine_worker: PyTorchRayWorker,
      tokenizer_path: str,
      context_length: int,
      batch_size: int,
      is_disaggregated: bool = False,
      pod_slice_name: str = None,
  ):
    self.engine_worker = engine_worker
    self.tokenizer_path = tokenizer_path
    self.context_length = context_length
    self.batch_size = batch_size
    self.is_disaggregated = is_disaggregated
    self.pod_slice_name = pod_slice_name

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
    return self.engine_worker.prefill_ray(
          params=params,
          existing_prefix=existing_prefix,
          padded_tokens=padded_tokens,
          true_length=true_length,
      )

  def transfer(self, np_prefix: NpPrefix) -> Any:
    """Store prefill result into object store, then transfer to decode engine workers."""

    return self.engine_worker.transfer(np_prefix)

  def insert(
      self,
      prefix: Prefix,
      decode_state: DecodeState,
      slot: int,
  ) -> DecodeState:
    return  self.engine_worker.insert_ray(
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
    # ray head doesn't load any parameters
    return None

  def get_prefix_destination_sharding(self) -> Prefix:
    "No implementation"
    return None

  @property
  def mesh(self):
    "No implementation"
    return None


# pylint: disable-next=all
def create_pytorch_ray_engine(
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
    is_disaggregated: bool = False,
    num_hosts: int = 0,
    decode_pod_slice_name: str = None,
) -> Any:

  # Return tuple as reponse: issues/107
  supported_models = ["llama-2", "llama-3", "gemma"]
  if model_name not in supported_models:
    raise NotImplementedError(
        f"Model name should be one of{','.join(supported_models)}"
    )
  ray.init(ignore_reinit_error=True)
  pod_name = tpu.get_current_pod_name()
  num_hosts = (
      num_hosts if is_disaggregated else tpu.get_current_pod_worker_count()
  )
  print(f"pod_name:{pod_name}, number of host: {num_hosts}")
  assert (
      pod_name is not None
  ), f"TPU pod name (current value:{pod_name}) can not be None"
  assert (
      num_hosts > 0
  ), f"num_hosts (current value {num_hosts}) should be a positive number"
  
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



  engine = PyTorchLocalEngine(
    engine_worker=engine_worker,
    tokenizer_path=tokenizer_path,
    context_length=context_length,
    batch_size=batch_size,
    is_disaggregated=is_disaggregated,
    pod_slice_name=pod_name,
  )
  return engine
