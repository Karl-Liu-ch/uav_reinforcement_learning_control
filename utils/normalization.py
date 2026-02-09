"""Normalization utilities for RL observations and actions."""

import numpy as np
from gymnasium.spaces import Box


def normalize(x: np.ndarray, space: Box) -> np.ndarray:
    """Normalize values from [space.low, space.high] to [-1, 1].

    Args:
        x: Values in the original space
        space: The gymnasium Box space defining the bounds

    Returns:
        Normalized values in [-1, 1]
    """
    return 2.0 * (x - space.low) / (space.high - space.low) - 1.0


def denormalize(x_normed: np.ndarray, space: Box) -> np.ndarray:
    """Denormalize values from [-1, 1] to [space.low, space.high].

    Args:
        x_normed: Normalized values in [-1, 1]
        space: The gymnasium Box space defining the bounds

    Returns:
        Values in the original space
    """
    return (x_normed + 1.0) / 2.0 * (space.high - space.low) + space.low
