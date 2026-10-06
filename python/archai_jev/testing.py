"""Test doubles. :class:`MockScorer` is **not a model**: its logits are a deterministic function
of the input and it declares no calibration. Use it to try the API and to test your own code.
"""

from ._core import Fault, MockScorer

__all__ = ["Fault", "MockScorer"]
