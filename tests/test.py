import jax
import jax.sharding as jsharding
from jax.experimental import mesh_utils


num_of_partitions = jax.device_count()
print(f"num_of_partitions {num_of_partitions}")

mesh = jsharding.Mesh(
    mesh_utils.create_device_mesh((num_of_partitions, 1)),
    axis_names=("x", "y"),
)

mesh_2d = jsharding.Mesh(
    mesh_utils.create_device_mesh((num_of_partitions // 2, 2)),
    axis_names=("x", "y"),
)