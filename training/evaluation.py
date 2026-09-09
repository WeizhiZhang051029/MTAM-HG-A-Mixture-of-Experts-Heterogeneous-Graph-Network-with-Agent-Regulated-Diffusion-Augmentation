from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader
import config


def feedback_batch_size(default: int) -> int:

    configured = int(getattr(config, "FEEDBACK_EVAL_BATCH_SIZE", 0))
    if configured < 0:
        raise ValueError("FEEDBACK_EVAL_BATCH_SIZE must be zero or positive.")
    return configured or int(default)


@torch.no_grad()
def score_synthetic_candidates(
    quality_agent, dataset, feedback_features, device, *, sample_count, batch_size, num_workers=0
):

    scores = np.ones(sample_count, dtype=np.float64)
    was_training = quality_agent.training
    quality_agent.eval()
    try:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        for x, y, sample_ids in loader:
            ids = sample_ids.detach().cpu().numpy().reshape(-1)
            x, y = x.to(device), y.to(device)
            feedback = torch.tensor(feedback_features[ids], dtype=x.dtype, device=device)
            policy = quality_agent(x, y, feedback).reshape(-1).clamp(0.0, 1.0)
            scores[ids] = policy.cpu().numpy()
    finally:
        quality_agent.train(was_training)
    return scores
