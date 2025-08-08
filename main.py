import os, socket
import mlx.core as mx

backend = os.getenv("MLX_BACKEND", "any")  # 'any' tries ring first, then MPI
world = mx.distributed.init(backend=backend)

rank, size = world.rank(), world.size()

# Simple all-reduce: sum a vector of ones across all ranks
x = mx.ones((8,), mx.float32)
y = mx.distributed.all_sum(x)
mx.eval(y)  # force compute

print(f"host={socket.gethostname()} rank={rank}/{size} first={y[0].item()} len={y.size}")