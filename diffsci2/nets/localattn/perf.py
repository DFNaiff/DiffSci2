"""Optional NATTEN performance knobs.

Nothing here is enabled by default; callers must opt in explicitly.
"""

from __future__ import annotations


def enable_fast_natten() -> None:
    """Enable NATTEN's KV-parallel fused backward + unrestricted memory.

    Calls ``natten.use_kv_parallelism_in_fused_na(True)`` and
    ``natten.set_memory_usage_preference('unrestricted')``.

    Why: the fused ``na2d`` backward lacks KV-parallelism by default,
    which is pathological at small-spatial / many-heads levels.
    Measured (2026-06-10, A100 40 GB):

    - M-rung LAUNet, 432x548 input, batch 4, bf16:
      0.47 -> 2.13 it/s end-to-end (~4.5x faster), identical math.
    - The cost concentrates at the deepest level (108x137 spatial,
      16 heads): 83 ms/call without KV-parallelism vs 12 ms with.

    Trade-off: the fused backward becomes NON-DETERMINISTIC (atomic
    accumulation order). Do not use when bit-exact reproducibility of
    gradients is required.

    This function is never called by default anywhere in diffsci2 —
    opt in explicitly from training scripts.
    """
    import natten
    natten.use_kv_parallelism_in_fused_na(True)
    natten.set_memory_usage_preference('unrestricted')
