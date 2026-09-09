from __future__ import annotations

import numpy as np
import torch


def transfer_float_metadata(values, *, dtype, device):

    if not values:
        return {}
    arrays = [np.asarray(value) for value in values.values()]
    if any(array.dtype.kind not in "bf" for array in arrays):
        raise TypeError("Batch metadata must contain Boolean or floating arrays.")
    sizes = [array.size for array in arrays]
    packed = torch.tensor(
        np.concatenate([array.reshape(-1) for array in arrays]),
        dtype=dtype, device=device,
    )
    return {
        name: chunk.reshape(array.shape)
        for name, array, chunk in zip(values, arrays, packed.split(sizes))
    }
