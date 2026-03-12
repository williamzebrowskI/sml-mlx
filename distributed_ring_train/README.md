# Distributed Ring Train

This directory is a separate MLX-only distributed training bundle for the
current 4-Mac Thunderbolt ring setup.

Training mode:
- Data parallel, not sequence parallel.
- Every Mac keeps a full copy of the model.
- Each rank processes its own micro-batch on its local GPU.
- Gradients are all-reduced across the 4 ranks every step.
- Rank 0 applies the optimizer update and broadcasts the new weights back out.

Checkpoint policy:
- `mac-1` stays rank 0 because it is the first host in `hosts.json`.
- Only rank 0 writes model checkpoints.
- HF streaming cursor state is disabled in `config.json`, so remote Macs do not
  write per-rank checkpoint state files.

Files:
- `config.json`: distributed MLX training config
- `hosts.json`: Thunderbolt ring hostfile for the current cable/IP layout
- `launch_train.sh`: launch helper for `mlx.launch`

Launch:

```bash
/Users/williamzebrowski/sml-mlx/distributed_ring_train/launch_train.sh
```

Useful overrides:

```bash
/Users/williamzebrowski/sml-mlx/distributed_ring_train/launch_train.sh --max-steps 100 --save-every 50
```

If the Thunderbolt cabling changes, regenerate the ring IP layout before using
this hostfile again.
