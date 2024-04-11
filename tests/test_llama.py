import torch
import os
from torch.utils import _pytree as pytree
import torch_xla2
import jax
import jax.numpy as jnp
import numpy as np
from jetstream_pt.engine import PyTorchEngine
from jetstream_pt.third_party.llama2 import model_exportable
from jetstream_pt.third_party.llama2.generation_original import LlamaOriginal
from jetstream_pt import environment

import unittest


class LlamaE2ETest(unittest.TestCase):

    def setup(self):
        torch.set_default_dtype(torch.bfloat16)

    def _to_jax(self, tree):
        return pytree.tree_map_only(
            torch.Tensor,
            torch_xla2.tensor.t2j, tree)  

    def _diff_value(self, value, expected, name):
        # value = value.astype(jnp.float32)
        value = torch_xla2.tensor.j2t(value)
        print(f"Single Attention: {name} diff norm:  {(value - expected).norm()}")
        # self.assertTrue(torch.allclose(value, expected, atol=1e-4))


    def _make_env(self):
        jax.config.update('jax_dynamic_shapes', False)
        jax.config.update('jax_traceback_filtering', 'off')
        env_data = environment.JetEngineEnvironmentData()
        env_data.max_input_sequence_length = 128
        env_data.max_input_sequence_length = 128
        env_data.cache_sequence_length = 128
        env_data.model_type = 'llama-2-tiny'
        env_data.batch_size = 1
        env = environment.JetEngineEnvironment(env_data)
        env.apply_sharding = lambda *args, **kwargs: None  # don't shard on cpu
        return env


    # def test_original_only(self):
    #     jax.config.update('jax_platform_name', 'cpu')
    #     print(f"---------> {jax.devices()}")

    #     torch.set_default_dtype(torch.bfloat16)
    #     env = self._make_env()
    #     model_arg = env._model_arg 
    #     tokens = np.arange(10, dtype=np.int32)
    #     true_length = tokens.shape[-1]
    #     padded_tokens = np.pad(tokens, (0, 6))
    #     padded_tokens = jnp.array(padded_tokens)

    #     seed = 1
    #     torch.manual_seed(1)
    #     max_output_length = 5

    #     file_dir = os.path.dirname(__file__)
    #     tokenizer_path = os.path.join(file_dir, '../jetstream_pt/third_party/llama2/tokenizer.model')

    #     # orginal
    #     llama_original = LlamaOriginal.build(tokenizer_path, model_arg, seed)
    #     model_orig = llama_original.model
    #     state_dict = dict(model_orig.state_dict())
    #     state_dict['freqs_cis'] = model_orig.freqs_cis
    #     model_ours = model_exportable.Transformer(model_arg, env)
    #     engine = PyTorchEngine(
    #         pt_model=model_ours,
    #         env=env
    #     )
    #     params = self._to_jax(state_dict)
    #     slot = 0 
    #     expected_output_tokens = []
    #     out_tokens = []


    #     prompt_tokens = [tokens]
    #     decode_state_original = llama_original.prefill(prompt_tokens, max_output_length)

    #     print(f"-------------------->: prefill: --->")
    #     print(f"-------------------->orginal: {decode_state_original.out_tokens}")
    
    #     for i in range(0, max_output_length - 1):
    #         decode_state_original = llama_original.decode(decode_state_original)

    #         print(f"-------------------->: decode {i}: --->")
    #         print(f"-------------------->orginal: {decode_state_original.out_tokens}")


    # def test_original_32(self):
    #     jax.config.update('jax_platform_name', 'cpu')
    #     print(f"---------> {jax.devices()}")

    #     torch.set_default_dtype(torch.float32)
    #     env = self._make_env()
    #     model_arg = env._model_arg 
    #     tokens = np.arange(10, dtype=np.int32)
    #     true_length = tokens.shape[-1]
    #     padded_tokens = np.pad(tokens, (0, 6))
    #     padded_tokens = jnp.array(padded_tokens)

    #     seed = 1
    #     torch.manual_seed(1)
    #     max_output_length = 5

    #     file_dir = os.path.dirname(__file__)
    #     tokenizer_path = os.path.join(file_dir, '../jetstream_pt/third_party/llama2/tokenizer.model')

    #     # orginal
    #     llama_original = LlamaOriginal.build(tokenizer_path, model_arg, seed)
    #     model_orig = llama_original.model
    #     state_dict = dict(model_orig.state_dict())
    #     state_dict['freqs_cis'] = model_orig.freqs_cis
    #     model_ours = model_exportable.Transformer(model_arg, env)
    #     engine = PyTorchEngine(
    #         pt_model=model_ours,
    #         env=env
    #     )
    #     params = self._to_jax(state_dict)
    #     slot = 0 
    #     expected_output_tokens = []
    #     out_tokens = []


    #     prompt_tokens = [tokens]
    #     decode_state_original = llama_original.prefill(prompt_tokens, max_output_length)


    #     print(f"-------------------->: prefill_32: --->")
    #     print(f"-------------------->orginal_32: {decode_state_original.out_tokens}")
    
    #     for i in range(0, max_output_length - 1):
    #         decode_state_original = llama_original.decode(decode_state_original)
    #         print(f"-------------------->: decode_32 {i}: --->")
    #         print(f"-------------------->orginal_32: {decode_state_original.out_tokens}")

   
    def test_llama_e2e(self):
        jax.config.update('jax_platform_name', 'cpu')
        print(f"---------> {jax.devices()}")

        torch.set_default_dtype(torch.float32)
        env = self._make_env()
        model_arg = env._model_arg 
        tokens = np.arange(10, dtype=np.int32)
        true_length = tokens.shape[-1]
        padded_tokens = np.pad(tokens, (0, 6))
        padded_tokens = jnp.array(padded_tokens)

        seed = 1
        torch.manual_seed(1)
        max_output_length = 5

        file_dir = os.path.dirname(__file__)
        tokenizer_path = os.path.join(file_dir, '../jetstream_pt/third_party/llama2/tokenizer.model')

        # orginal
        llama_original = LlamaOriginal.build(tokenizer_path, model_arg, seed)
        model_orig = llama_original.model
        state_dict = dict(model_orig.state_dict())
        state_dict['freqs_cis'] = model_orig.freqs_cis
        model_ours = model_exportable.Transformer(model_arg, env)
        engine = PyTorchEngine(
            pt_model=model_ours,
            env=env
        )
        params = self._to_jax(state_dict)
        slot = 0 
        expected_output_tokens = []
        out_tokens = []


        prompt_tokens = [tokens]
        decode_state_original = llama_original.prefill(prompt_tokens, max_output_length)

        decode_state = engine.init_decode_state()
        prefill_result = engine.prefill(
            params=params, padded_tokens=padded_tokens, true_length=true_length
        )
        self._diff_value(prefill_result.logits[0:10, :], torch.squeeze(decode_state_original.logits), "prefill logits")
        decode_state = engine.insert(
            prefill_result, decode_state, slot=slot
        )
        print(f"-------------------->: prefill step: --->")
        print(f"-------------------->orginal out_tokens: {decode_state_original.out_tokens}")
        print(f"-------------------->out_tokens: {prefill_result.token}")     
    
        for i in range(0, max_output_length - 1):
            decode_state_original = llama_original.decode(decode_state_original)
            decode_state, result_tokens = engine.generate(params, decode_state)

            self._diff_value(decode_state.logits, decode_state_original.logits, "prefill logits")
            print(f"-------------------->: decode step {i}: --->")
            print(f"-------------------->orginal out_tokens: {decode_state_original.out_tokens}")
            print(f"-------------------->out_tokens: {decode_state.tokens}")          

if __name__ == '__main__':
    unittest.main()
