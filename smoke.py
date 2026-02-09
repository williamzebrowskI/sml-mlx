import mlx.core as mx

mx.distributed.init()
r = mx.distributed.rank()
w = mx.distributed.world_size()

x = mx.ones((1,)) * (r + 1)
y = mx.distributed.all_sum(x)
mx.eval(y)

if r == 0:
    print("world_size:", w, "all_sum:", float(y.item()))