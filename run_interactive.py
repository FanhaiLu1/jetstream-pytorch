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

import functools
import os
import random
import time
from typing import List

import humanize
import jax
import numpy as np
from absl import app, flags
from colorama import Fore, Style
from jetstream.engine import token_utils
from jetstream_pt import engine as je
from jetstream_pt import local_engine
from jetstream_pt.config import FLAGS, create_engine_from_config_flags


def create_engine():
  """create a pytorch engine"""
  jax.config.update("jax_default_prng_impl", "unsafe_rbg")
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"

  start = time.perf_counter()
  engine = local_engine.create_pytorch_ray_engine(
      model_name=FLAGS.model_name,
      tokenizer_path=FLAGS.tokenizer_path,
      ckpt_path=FLAGS.checkpoint_path,
      bf16_enable=FLAGS.bf16_enable,
      param_size=FLAGS.size,
      context_length=FLAGS.context_length,
      batch_size=FLAGS.batch_size,
      quantize_weights=FLAGS.quantize_weights,
      quantize_kv=FLAGS.quantize_kv_cache,
      max_cache_length=FLAGS.max_cache_length,
      sharding_config=FLAGS.sharding_config,
  )

  print("Initialize engine", time.perf_counter() - start)
  return engine

def print_mem_usage():
  """Print current mem usage"""
  fmt_size = functools.partial(humanize.naturalsize, binary=True)

  for d in jax.local_devices():
    stats = d.memory_stats()
    used = stats["bytes_in_use"]
    limit = stats["bytes_limit"]
    print(
        f"memory using {fmt_size(used)} / {fmt_size(limit)} ({used/limit:%}) on {d}"
    )

# pylint: disable-next=all
def main(argv):

  engine = create_engine()

  start = time.perf_counter()
  params = engine.load_params()
  print("Load params ", time.perf_counter() - start)

  metadata = engine.get_tokenizer()
  tokenizer = engine.build_tokenizer(metadata)
  max_output_length = 1024

  profiling_output = FLAGS.profiling_output
  profiling_prefill = FLAGS.profiling_prefill
  if profiling_output and profiling_prefill:
    jax.profiler.start_trace(profiling_output)

  decode_state = engine.init_decode_state()
  prompts: List[str] = [
      "I believe the meaning of life is",
      "To add an element to an ArrayList of a specific class type in Java, you can follow the following steps:\n\n1. Create an instance of the class to be added.\n2. Get a reference to the ArrayList.\n3. Call the `add()` method on the ArrayList, passing the instance of the class as the argument.\n\nHere's an example of how to add an object of type `Person` to an ArrayList of type `ArrayList<Person>`:\n```csharp\n// Create a new instance of the Person class\nPerson person = new Person(\"John\", 25);\n\n// Get a reference to the ArrayList\nArrayList<Person> peopleList = new ArrayList<>();\n\n// Add the person object to the ArrayList\npeopleList.add(person);\n```\nIn this example, the `Person` class is assumed to have a constructor that takes two arguments: a String for the person's name, and an int for their age. You can substitute your own class and constructor as necessary.",
      "<s>[INST] <<SYS>>\nYou are an AI assistant. User will you give you a task. Your goal is to complete the task as faithfully as you can. While performing the task think step-by-step and justify your steps.\n<</SYS>>\n\nQuestion 1: What is commercial real estate finance?\nQuestion 2: What are Commercial Real Estate services?\nOptions are:\n[a]. no.\n[b]. yes.\nWould the answer to these two questions be the same? [/INST]",
      "<s>[INST] <<SYS>>\nYou are an AI assistant that helps people find information. Provide a detailed answer so user don\u2019t need to search outside to understand the answer.\n<</SYS>>\n\nUse reasoning to lead to the answer of the following question:\nWhere are you likely to find water underneath?\nOptions:\n- toilet\n- sink\n- jar\n- bridge\n- house\n Reasoning process: [/INST",
      "<s>[INST] <<SYS>>\nYou are an AI assistant. You will be given a task. You must generate a detailed and long answer.\n<</SYS>>\n\nContinue the following story.\n\nKay didn't have shoes that fit her feet properly. She only wore sneakers, because the \nChoose from: [I] shoes  fitted badly. [II] sneakers  fitted badly. [/INST]",
  ]
  for prompt in prompts:
    slot = random.randint(0, FLAGS.batch_size - 1)
    tokens, true_length = tokenizer.encode(prompt)

    print(f"---- Input prompts are: {prompt}")
    print(f"---- Encoded tokens are: {tokens}")

    for i in range(FLAGS.batch_size):
      # pylint: disable-next=all
      prefill_result = engine.prefill(
          params=params, padded_tokens=tokens, true_length=true_length
      )
      print("---- print_mem_usage  after prefill.")
      print_mem_usage()
      # pylint: disable-next=all
      decode_state = engine.insert(prefill_result, decode_state, slot=i)
    print("print_mem_usage  after insert")
    print_mem_usage() 
    sampled_tokens_list = []
    print(f"---- Streaming decode started on #slot{slot}.")
    complete = np.zeros((1,), dtype=np.bool_)
    while True:
      if profiling_output and not profiling_prefill:
        jax.profiler.start_trace(profiling_output)
      decode_state, result_tokens = engine.generate(params, decode_state)
      if profiling_output and not profiling_prefill:
        jax.profiler.stop_trace()
      result_tokens = result_tokens.convert_to_numpy()
      output, complete = token_utils.process_result_tokens(
          tokenizer=tokenizer,
          slot=slot,
          slot_max_length=max_output_length,
          result_tokens=result_tokens,
          complete=complete,
      )
      if complete[0]:
        break
      token_ids = output[0].token_ids
      sampled_tokens_list.extend(token_ids)

    # print("---- All output tokens.")
    # print(sampled_tokens_list)
    # print("---- All output text.")
    # print(tokenizer.decode(sampled_tokens_list))
    
    print("---- print_mem_usage  text.")
    print_mem_usage()

  if profiling_output and profiling_prefill:
    jax.profiler.stop_trace()


if __name__ == "__main__":
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
  app.run(main)
