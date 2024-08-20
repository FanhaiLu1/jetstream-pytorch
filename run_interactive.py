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
import humanize
import os
import random
import time
from typing import List

# import torch_xla2 first!
import jax
import numpy as np
from absl import app
from jetstream.engine import token_utils
from jetstream_pt.config import FLAGS, create_engine_from_config_flags


# pylint: disable-next=all
def main(argv):

  engine = create_engine_from_config_flags()

  start = time.perf_counter()
  params = engine.load_params()
  print("Load params ", time.perf_counter() - start)

  metadata = engine.get_tokenizer()
  tokenizer = engine.build_tokenizer(metadata)
  max_output_length = 1024

  profiling_output = FLAGS.profiling_output
  profiling_prefill = (
      FLAGS.profiling_prefill
      and profiling_output is not None
      and profiling_output != ""
  )

  if profiling_prefill:
    jax.profiler.start_trace(profiling_output)
  
  print(f"--------------------------> after load_params") 
  print_mem_usage()  

  decode_state = engine.init_decode_state()
  print(f"--------------------------> after init_decode_state") 
  print_mem_usage()  

  if profiling_prefill:
    jax.profiler.stop_trace()

  prompts: List[str] = [
      # pylint: disable-next=all
      "I believe the meaning of life is",
      # pylint: disable-next=all
      "To add an element to an ArrayList of a specific class type in Java, you can follow the following steps:\n\n1. Create an instance of the class to be added.\n2. Get a reference to the ArrayList.\n3. Call the `add()` method on the ArrayList, passing the instance of the class as the argument.\n\nHere's an example of how to add an object of type `Person` to an ArrayList of type `ArrayList<Person>`:\n```csharp\n// Create a new instance of the Person class\nPerson person = new Person(\"John\", 25);\n\n// Get a reference to the ArrayList\nArrayList<Person> peopleList = new ArrayList<>();\n\n// Add the person object to the ArrayList\npeopleList.add(person);\n```\nIn this example, the `Person` class is assumed to have a constructor that takes two arguments: a String for the person's name, and an int for their age. You can substitute your own class and constructor as necessary.",
      # pylint: disable-next=all
      "<s>[INST] <<SYS>>\nYou are an AI assistant. User will you give you a task. Your goal is to complete the task as faithfully as you can. While performing the task think step-by-step and justify your steps.\n<</SYS>>\n\nQuestion 1: What is commercial real estate finance?\nQuestion 2: What are Commercial Real Estate services?\nOptions are:\n[a]. no.\n[b]. yes.\nWould the answer to these two questions be the same? [/INST]",
      # pylint: disable-next=all
      "<s>[INST] <<SYS>>\nYou are an AI assistant that helps people find information. Provide a detailed answer so user don\u2019t need to search outside to understand the answer.\n<</SYS>>\n\nUse reasoning to lead to the answer of the following question:\nWhere are you likely to find water underneath?\nOptions:\n- toilet\n- sink\n- jar\n- bridge\n- house\n Reasoning process: [/INST",
      # pylint: disable-next=all
      "<s>[INST] <<SYS>>\nYou are an AI assistant. You will be given a task. You must generate a detailed and long answer.\n<</SYS>>\n\nContinue the following story.\n\nKay didn't have shoes that fit her feet properly. She only wore sneakers, because the \nChoose from: [I] shoes  fitted badly. [II] sneakers  fitted badly. [/INST]",
  ]
  for prompt in prompts:
    slot = random.randint(0, FLAGS.batch_size - 1)
    tokens, true_length = tokenizer.encode(prompt)

    print(f"---- Input prompts are: {prompt}")
    print(f"---- Encoded tokens are: {tokens}")

    # pylint: disable-next=all
    if profiling_prefill:
      jax.profiler.start_trace(profiling_output)

    prefill_result, _ = engine.prefill(
        params=params, padded_tokens=tokens, true_length=true_length
    )
    print(f"--------------------------> after prefill_result") 
    #print_mem_usage()  
    # pylint: disable-next=all
    decode_state = engine.insert(prefill_result, decode_state, slot=slot)
    print(f"--------------------------> after insert") 
    #print_mem_usage()

    if profiling_prefill:
      jax.profiler.stop_trace()

    sampled_tokens_list = []
    print(f"---- Streaming decode started on #slot{slot}.")
    complete = np.zeros((1,), dtype=np.bool_)
    while True:
      if profiling_output:
        jax.profiler.start_trace(profiling_output)
      
      print(f"--------------------------> before generate")  
      #print_mem_usage()
      decode_state = engine.generate(params, decode_state)
      # result_tokens = result_tokens.convert_to_numpy()
      print(f"--------------------------> after generate")  
      #print_mem_usage()

      if profiling_output:
        jax.profiler.stop_trace()

      output, complete = token_utils.process_result_tokens(
          tokenizer=tokenizer,
          slot=slot,
          slot_max_length=max_output_length,
          result_tokens=None,
          complete=complete,
      )
      if complete[0]:
        break
      token_ids = output[0].token_ids
      sampled_tokens_list.extend(token_ids)

    print("---- All output tokens.")
    print(sampled_tokens_list)
    print("---- All output text.")
    print(tokenizer.decode(sampled_tokens_list))

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

if __name__ == "__main__":
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
  app.run(main)
