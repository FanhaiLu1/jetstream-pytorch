import jax
import jax.numpy as jnp
import jax.sharding as jsharding
from jax.sharding import Mesh, PartitionSpec

# Define a mesh (assuming 2 devices)
mesh = Mesh(jax.devices(), axis_names=('x',)) 

shape = (8,) 
sharding = jsharding.NamedSharding(mesh, PartitionSpec("x"))

def data_callback(indices):
    # Create an array with values based on the indices of each shard
    print("---->",jnp.arange(indices[0].start, indices[0].stop) * 2)
    return jnp.arange(indices[0].start, indices[0].stop) * 2

arr_sharded = jax.make_array_from_callback(shape, sharding, data_callback)
print(arr_sharded.shape)
print(arr_sharded) 