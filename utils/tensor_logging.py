from __future__ import annotations

from collections import defaultdict
from numbers import Real
import torch


def scalar_logs(values: dict[str, torch.Tensor | Real]) -> dict[str, float]:

    result = {}
    groups = defaultdict(list)
    for key, value in values.items():
        if not torch.is_tensor(value):
            result[key] = float(value)
            continue
        if value.numel() != 1:
            raise ValueError(f"Log {key!r} must be scalar, got {tuple(value.shape)}")
        groups[(value.device, value.dtype)].append((key, value.detach().reshape(())))
    for entries in groups.values():
        host_values = torch.stack([value for _, value in entries]).cpu().tolist()
        result.update((key, float(value)) for (key, _), value in zip(entries, host_values))
    return {key: result[key] for key in values}
