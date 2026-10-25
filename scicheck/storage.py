from __future__ import annotations

import torch

MB = 2**20
INDEX_BYTES = torch.empty(0, dtype=torch.int64).element_size()


def tensor_bytes(obj) -> int:
    if isinstance(obj, torch.Tensor):
        return obj.numel() * obj.element_size()
    if isinstance(obj, dict):
        return sum(tensor_bytes(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(tensor_bytes(v) for v in obj)
    return 0


def checkpoint_bytes(seg) -> int:
    return tensor_bytes(seg.model_state) + tensor_bytes(seg.opt_state) + tensor_bytes(seg.aug_state)


def segment_bytes(seg) -> int:
    return checkpoint_bytes(seg) + tensor_bytes(seg.batches) + tensor_bytes(seg.end_weights)


def checkpoint_storage(seg, n_steps: int, n_rows: int, batch_size: int, seg_steps: int, n_kept: int) -> dict:
    steps_per_epoch = -(-n_rows // batch_size)
    full_epochs, rest = divmod(n_steps, steps_per_epoch)
    run_indices = full_epochs * n_rows + min(rest * batch_size, n_rows)
    ckpt, weights = checkpoint_bytes(seg), tensor_bytes(seg.model_state)
    n_all, n_epoch = n_steps // seg_steps + 1, steps_per_epoch // seg_steps + 1
    kept = n_kept * segment_bytes(seg)
    return dict(
        ckpt_mb=ckpt / MB,
        keep_all_mb=(n_all * ckpt + run_indices * INDEX_BYTES) / MB,
        keep_all_weights_mb=(n_all * weights + run_indices * INDEX_BYTES) / MB,
        bounded_peak_mb=(n_epoch * ckpt + n_rows * INDEX_BYTES + kept) / MB,
        submitted_mb=kept / MB,
    )
