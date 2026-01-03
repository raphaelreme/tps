import numpy as np
import pytest

from tps import ThinPlateSpline


def test_interpolates_training_points_alpha0():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 3))
    Y = rng.normal(size=(50, 2))

    tps = ThinPlateSpline(alpha=0.0, order=2)
    tps.fit(X, Y)
    Y_pred = tps.transform(X)

    assert np.max(np.abs(Y_pred - Y)) == pytest.approx(0)


def test_regularization_relaxes_interpolation():
    tol = 1e-2
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 3))
    Y = rng.normal(size=(20, 2))

    tps = ThinPlateSpline(alpha=0.05, order=2).fit(X, Y)

    error = np.max(np.abs(tps.transform(X) - Y))

    assert error > tol  # should generally not be exact anymore


def test_permutation_invariance():
    rng = np.random.default_rng(0)

    X = rng.normal(size=(80, 4))
    Y = rng.normal(size=(80,))
    X_test = rng.normal(size=(30, 4))

    perm = rng.permutation(len(X))

    tps = ThinPlateSpline(alpha=0.05, order=2).fit(X, Y)
    tps_perm = ThinPlateSpline(alpha=0.05, order=2).fit(X[perm], Y[perm])

    Y_pred = tps.transform(X_test)
    Y_perm = tps_perm.transform(X_test)

    assert np.max(np.abs(Y_pred - Y_perm)) == pytest.approx(0)


def test_affine_reproduction_order_2():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 5))
    A = rng.normal(size=(5, 3))
    b = rng.normal(size=(3,))
    Y = X @ A + b

    X_test = rng.normal(size=(40, 5))
    Y_true = X_test @ A + b

    # For order=2, polynomial degree is 1: any affine function should be reproduced exactly.
    tps = ThinPlateSpline(alpha=0.0, order=2).fit(X, Y)
    Y_pred = tps.transform(X_test)

    assert np.max(np.abs(Y_pred - Y_true)) == pytest.approx(0.0)


def test_known_3d_function_regression_sanity():
    def f(X):
        return (np.sin(X[:, 0]) + np.cos(X[:, 1]) + X[:, 2] ** 2)[:, None]

    rng = np.random.default_rng(0)
    Xc = rng.uniform(-4, 4, size=(500, 3))
    Yc = f(Xc)

    Xt = rng.uniform(-3, 3, size=(50, 3))
    Yt = f(Xt)

    tps = ThinPlateSpline(alpha=0.1, order=2).fit(Xc, Yc)
    Y_pred = tps.transform(Xt)

    tol = 1e-2
    mse = np.mean((Y_pred - Yt) ** 2)
    assert mse < tol


def test_duplicate_fails_without_regularization():
    # TPS becomes singular with duplicates when alpha=0.
    X = np.array(
        [
            [0.0, 0.0],
            [0.0, 0.0],  # duplicate
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float64,
    )
    Y = np.random.default_rng(0).normal(size=(4, 1))

    tps = ThinPlateSpline(alpha=0.0, order=2)

    with pytest.raises(np.linalg.LinAlgError):
        tps.fit(X, Y)


def test_duplicate_works_with_regularization():
    X = np.array(
        [
            [0.0, 0.0],
            [0.0, 0.0],  # duplicate
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float64,
    )
    Y = np.random.default_rng(0).normal(size=(4, 1))

    tps = ThinPlateSpline(alpha=1e-3, order=2)
    tps.fit(X, Y)

    assert np.all(np.isfinite(tps.parameters)), "Parameters should be finite with regularization"


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_dtype_conserved(dtype: np.dtype):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 1)).astype(dtype)
    Y = rng.normal(size=(20, 5)).astype(dtype)
    X_test = rng.normal(size=(30, 1)).astype(dtype)

    predicted = ThinPlateSpline().fit(X, Y).transform(X_test)
    assert predicted.dtype == dtype


def test_runtime_error_if_not_fitted():
    rng = np.random.default_rng(0)
    X_test = rng.normal(size=(30, 1))

    with pytest.raises(RuntimeError):
        ThinPlateSpline().transform(X_test)


def test_value_error():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(15, 1))
    Y = rng.normal(size=(20, 5))

    with pytest.raises(ValueError, match="same number of points"):
        ThinPlateSpline().fit(X, Y)

    X = rng.normal(size=(20, 5, 5))
    Y = rng.normal(size=(20, 5))

    with pytest.raises(ValueError, match="2d"):
        ThinPlateSpline().fit(X, Y)

    X = rng.normal(size=(15, 5))
    Y = rng.normal(size=(20, 3))

    with pytest.raises(ValueError, match="features"):
        ThinPlateSpline().fit(X, X).transform(Y)
