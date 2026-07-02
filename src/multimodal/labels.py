from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def encode_labels(labels: pd.Series) -> tuple[np.ndarray, dict[str, Any]]:
    non_null = labels.dropna()
    if len(non_null) != len(labels):
        raise ValueError("Labels must not contain missing values.")

    numeric = pd.to_numeric(labels, errors="coerce")
    if numeric.notna().all() and np.all(np.equal(np.mod(numeric.to_numpy(), 1), 0)):
        values = numeric.astype(int).to_numpy()
        unique = sorted(int(value) for value in np.unique(values))
        if unique and unique[0] == 0:
            return values.astype(np.int64), {"type": "integer", "offset": 0, "classes": unique}
        if unique and unique[0] == 1:
            return (values - 1).astype(np.int64), {"type": "integer", "offset": 1, "classes": unique}

    codes, uniques = pd.factorize(labels.astype(str), sort=True)
    return codes.astype(np.int64), {"type": "categorical", "classes": [str(value) for value in uniques]}
