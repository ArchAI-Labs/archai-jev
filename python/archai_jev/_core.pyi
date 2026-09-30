from collections.abc import Sequence

__all__ = ["calibrated_softmax"]

def calibrated_softmax(
    logits: Sequence[float], temperature: float = 1.0
) -> list[float]:
    """Convert raw option scores (logits) into calibrated probabilities using a temperature-scaled softmax.

    A temperature > 1 flattens the distribution (less confident), a temperature < 1
    sharpens it (more confident).

    Raises:
        ValueError: If ``logits`` is empty or not finite, or if ``temperature`` is not
            a finite number > 0.
    """
