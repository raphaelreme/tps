"""Thin Plate Spline (TPS) in Python.

Python (NumPy & SciPy) implementation of the generalized Polyharmonic Spline interpolation
(also known as Thin Plate Spline in 2D).
It learns a smooth elastic mapping between two Euclidean spaces with support for:

* Arbitrary input and output dimensions
* Arbitrary spline order `k`
* Optional regularization

Useful for interpolation, deformation fields, and smooth non-linear regression.

For a faster implementation in PyTorch, see ![torch-tps](https://github.com/raphaelreme/torch-tps).

Getting started:
---------------

```python
import numpy as np
from tps import ThinPlateSpline

# Control points
X_train = np.random.normal(0, 1, (800, 3))  # 800 points in R^3
Y_train = np.random.normal(0, 1, (800, 2))  # Values for each point (800 values in R^2)

# New source points to interpolate
X_test = np.random.normal(0, 1, (300 0, 3))

# Initialize spline model (Regularization is controlled with alpha parameter)
tps = ThinPlateSpline(alpha=0.5)

# Fit spline from control points
tps.fit(X_train, Y_train)

# Interpolate new points
Y_test = tps.transform(X_test)
```

Please refer to the ![official documentation](https://github.com/raphaelreme/tps)

License
-------

MIT License
"""

import importlib.metadata

from .polynomial_transform import PolynomialFeatures
from .thin_plate_spline import ThinPlateSpline

__all__ = ["PolynomialFeatures", "ThinPlateSpline"]
__version__ = importlib.metadata.version("thin_plate_spline")
