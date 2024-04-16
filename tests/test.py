import os

import jax
import numpy as np
from jax import random, numpy as jnp
from jax.experimental import mesh_utils
from jax.experimental.multihost_utils import process_allgather,host_local_array_to_global_array
from jax.experimental.pjit import pjit
from jax.sharding import Mesh, PartitionSpec as P
from tests.environment import JetEngineEnvironment

jax.distributed.initialize()

print(f"----------> devices: {jax.devices()}, local_devices: {jax.local_devices()}")
env = JetEngineEnvironment()


@pjit
def func(x):
    return jax.numpy.sqrt(x)


key = random.key(0)
input = random.normal(key, (32,64))
x = jax.make_array_from_callback(input.shape, env.x_sharding, lambda idx: input[idx])

z = func(x)
jax.debug.visualize_array_sharding(z)
jax.debug.visualize_array_sharding(x)

result = np.asarray(process_allgather(z))
print(result.shape)
start = np.asarray(process_allgather(x))
assert np.allclose(np.sqrt(start), result)
print(result.size)