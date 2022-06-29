# tps

Implementation of Thin Plate Spline.
(For a faster implementation in torch, look at tps-torch)


## Install

### Pip

```bash
$ pip install thin-plate-spline
```

### Conda

Not yet available


## Getting started

```python

import numpy 
from tps import ThinPlateSpline

# Some data
X_c = np.random.normal(0, 1, (800, 3))
X_t = np.random.normal(0, 2, (800, 2))
X = np.random.normal(0, 1, (300, 3))

# Create the tps object
tps = ThinPlateSpline(alpha=0.0)  # 0 Regularization

# Fit the control and target points
tps.fit(X_c, X_t)

# Transform new points
Y = tps.transform(X)
```

Also have a look at `example.py`


## Build and Deploy

```bash
$ python -m build
$ python -m twine upload dist/*
```
