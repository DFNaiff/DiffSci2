"""Evaluation metrics used by the field-controlled pipeline and the paper.

Everything here is pure numpy / torch (no optional deps). Each metric
corresponds to a specific equation or table in main.tex:

- hellinger_gaussian     : eq. (9) of the article (sub-block diversity table).
- iae_kr_curves          : §4.2 relative-permeability IAE.
- iae_pc_curve           : §4.2 capillary-pressure IAE.
- field_pearson          : §4.3 field-correlation scatter.
- logit_field_tpc        : §4.3 two-point correlation of the logit field.

Everything is re-exported at the package level so callers can write::

    from diffsci2.metrics import hellinger_gaussian, iae_kr_curves

without having to care about the sub-module layout.
"""
from .hellinger import hellinger_gaussian
from .iae import iae_kr_curves, iae_pc_curve
from .field_diagnostics import field_pearson, logit_field_tpc

__all__ = [
    "hellinger_gaussian",
    "iae_kr_curves",
    "iae_pc_curve",
    "field_pearson",
    "logit_field_tpc",
]
