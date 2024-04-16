import jax
import jax.sharding as jsharding
from jax.experimental import mesh_utils

class JetEngineEnvironment:

    def __init__(self):
        Mesh = jax.sharding.Mesh
        P = jax.sharding.PartitionSpec
        num_of_partitions = jax.device_count()
        self._mesh = jsharding.Mesh(
            mesh_utils.create_device_mesh((num_of_partitions, 1)),
            axis_names=("x", "y"),
        )
        self.y_sharding = jsharding.NamedSharding(self._mesh, P(None, "x"))
        self.x_sharding = jsharding.NamedSharding(self._mesh, P("x"))
        self.replicated = jsharding.NamedSharding(self._mesh, P())   