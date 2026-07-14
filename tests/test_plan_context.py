import numpy as np
import torch

from sim.control.plan_context import CandidatePlanEncoder


def test_candidate_plan_encoder_outputs_candidate_logits_and_gradients():
    model = CandidatePlanEncoder(state_size=78, candidate_count=8)
    observations = {
        "state": torch.as_tensor(np.zeros((2, 78), dtype=np.float32)),
        "forecast": torch.as_tensor(np.zeros((2, 168, 9), dtype=np.float32)),
    }

    logits = model(observations)
    logits.sum().backward()

    assert logits.shape == (2, 8)
    assert all(parameter.grad is not None for parameter in model.parameters())
