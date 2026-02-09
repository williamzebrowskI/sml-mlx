import mlx.core as mx

g = mx.distributed.init()
r = g.rank()
w = g.size()

x = mx.ones((1,)) * (r + 1)
y = mx.distributed.all_sum(x, group=g)
mx.eval(y)

if r == 0:
    print("world_size:", w, "all_sum:", float(y.item()))
