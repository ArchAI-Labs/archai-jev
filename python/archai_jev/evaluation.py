"""Labelled data and the evaluation of models on it.

Import this module on its own (like :mod:`archai_jev.testing`)::

    from archai_jev.evaluation import compare, load_dataset, split_dataset

The methods ``evaluate``, ``calibrate`` and ``with_temperature`` are on :class:`archai_jev.Jev`.
"""

from ._evaluation import LabelledRecord, compare, load_dataset, split_dataset
from ._reports import (
    CalibrationStudy,
    Comparison,
    EvaluationReport,
    QuestionReport,
    TypeReport,
)

__all__ = [
    "CalibrationStudy",
    "Comparison",
    "EvaluationReport",
    "LabelledRecord",
    "QuestionReport",
    "TypeReport",
    "compare",
    "load_dataset",
    "split_dataset",
]
