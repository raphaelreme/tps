import math

import numpy as np
import pytest

from tps.polynomial_transform import PolynomialFeatures


def generate_random_inputs(n: int, d: int) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.normal(size=(n, d))


def test_degree_1_valid_in_2d():
    inputs = np.array([[2.0, 3.0]])
    pf = PolynomialFeatures(degree=1).fit(inputs)
    transformed = pf.transform(inputs)

    # degree 1 => [1, a, b]
    expected = np.array([[1.0, 2.0, 3.0]])
    assert np.allclose(transformed, expected)


def test_degree_2_valid_in_2d():
    inputs = np.array([[2.0, 3.0]])
    pf = PolynomialFeatures(degree=2).fit(inputs)
    predicted = pf.transform(inputs)

    # degree 2 => [1, a, b, a^2, ab, b^2]
    expected = np.array([[1.0, 2.0, 3.0, 4.0, 6.0, 9.0]])
    assert np.allclose(predicted, expected)


@pytest.mark.parametrize(("n", "d", "degree"), [(5, 4, 3), (20, 2, 2), (1, 1, 1), (2, 1, 4), (3, 3, 1)])
def test_output_dim_matches_combinatorial_count(n: int, d: int, degree: int):
    # d_out should be C(d+degree, degree)
    inputs = generate_random_inputs(n, d)
    pf = PolynomialFeatures(degree).fit(inputs)
    expected = math.comb(d + degree, degree)
    assert pf.n_output_features == expected
    assert PolynomialFeatures._combinations_length(d, degree) == len(list(PolynomialFeatures._combinations(d, degree)))

    predicted = pf.transform(inputs)
    assert predicted.shape == (n, expected)


def test_deterministic_and_order_stable():
    n = 10
    d = 3
    degree = 2
    inputs_1 = generate_random_inputs(n, d)
    inputs_2 = inputs_1.copy()

    predicted_1 = PolynomialFeatures(degree).fit(inputs_1).transform(inputs_1)
    predicted_2 = PolynomialFeatures(degree).fit(inputs_2).transform(inputs_2)

    assert np.allclose(predicted_1, predicted_2), "PolynomialFeatures should be deterministic and stable"


@pytest.mark.parametrize("dtype", [np.int16, np.int32, np.int64, np.uint8, np.uint32, np.float32, np.float64])
def test_dtype_conserved(dtype: np.dtype):
    n = 10
    d = 2
    degree = 2

    inputs = generate_random_inputs(n, d).astype(dtype)
    predicted = PolynomialFeatures(degree).fit(inputs).transform(inputs)

    assert predicted.dtype == dtype


def test_runtime_error_if_not_fitted():
    rng = np.random.default_rng(0)
    inputs = rng.normal(size=(30, 1))

    with pytest.raises(RuntimeError):
        PolynomialFeatures().transform(inputs)


def test_value_error():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(15, 5))
    Y = rng.normal(size=(20, 3))

    with pytest.raises(ValueError, match="shape"):
        PolynomialFeatures().fit(X).transform(Y)
