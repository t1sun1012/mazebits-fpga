"""Validation measurements for actual exported integer networks."""

import hashlib
import numpy as np
import torch

from .evaluation import evaluate_rollouts
from .fixed_point import IntegerPolicy, infer


def float_scores(model, inputs: torch.Tensor, batch_size=512) -> np.ndarray:
    model.eval()
    with torch.inference_mode():
        scores = np.concatenate([
            model(inputs[offset:offset + batch_size]).numpy()
            for offset in range(0, len(inputs), batch_size)
        ])
    if not np.isfinite(scores).all():
        raise ValueError("non-finite floating-point reference scores")
    return scores


def evaluate_integer(policy, inputs, targets, metadata, reference_scores, *, batch_size=128):
    adapter = IntegerPolicy(policy)
    scores = np.concatenate([
        infer(policy, inputs[offset:offset + batch_size].numpy(),
              stats=adapter.statistics)
        for offset in range(0, len(inputs), batch_size)
    ])
    actions = scores.argmax(axis=1)
    reference_actions = reference_scores.argmax(axis=1)
    good = targets.numpy()[np.arange(len(actions)), actions].astype(bool)
    state_stats = dict(adapter.statistics)
    adapter.reset_statistics()
    rollouts = evaluate_rollouts(adapter, metadata, batch_size=batch_size)
    head = policy.layers[-1]
    real_scores = np.ldexp(scores.astype(np.float64), -(head.input_frac + head.weight_frac))
    metrics = {
        "precision": f"W{policy.bits}A{policy.bits}",
        "states": len(actions),
        "optimal_action_correct": int(good.sum()),
        "optimal_action_accuracy": float(good.mean()),
        "float_action_agreement": float((actions == reference_actions).mean()),
        "float_action_differences": int((actions != reference_actions).sum()),
        "mean_absolute_logit_error": float(np.abs(real_scores - reference_scores).mean()),
        "max_absolute_logit_error": float(np.abs(real_scores - reference_scores).max()),
        "scores_sha256_int32_little_endian": hashlib.sha256(
            scores.astype("<i4").tobytes(order="C")).hexdigest(),
        "actions_sha256_uint8": hashlib.sha256(actions.astype(np.uint8).tobytes()).hexdigest(),
        "state_arithmetic_statistics": state_stats,
        "rollout_arithmetic_statistics": adapter.statistics,
        "rollouts": rollouts["summary"],
    }
    return metrics, rollouts, scores.astype(np.int32)
