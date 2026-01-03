import numpy as np
import pytest

from tps import ThinPlateSpline


@pytest.mark.parametrize("d", [1, 2, 3])
@pytest.mark.parametrize("order", [1, 2, 3, 4, 5])
def test_kernel_phi_zero_is_zero_on_diagonal(d: int, order: int):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(10, d))
    Y = rng.normal(size=(10, 1))

    tps = ThinPlateSpline(alpha=0.0, order=order)  # d=2 => power=2 (even, log)
    tps.fit(X, Y)
    kernel = tps._radial_distance(X)  # (n,n)

    # Kernel should be 0 on diagonal (or extremely close)
    assert np.max(np.abs(np.diag(kernel))) == pytest.approx(0.0)


def test_kernel_fallback_to_tps_when_power_non_positive():
    # With d=6, order=2, the power is negative => It should fallback to the std TPS r**2 log r
    # and therefore be finite on diagonal
    rng = np.random.default_rng(0)
    X = rng.normal(size=(10, 6))
    Y = rng.normal(size=(10, 1))

    tps = ThinPlateSpline(alpha=0.0, order=2)
    tps.fit(X, Y)
    kernel = tps._radial_distance(X)  # (n,n)
    assert np.isfinite(kernel).all()


def test_kernel_enforce_tps_kernel_forces_log_behavior():
    # 3D spline => odd power case but enforce_tps_kernel should override
    rng = np.random.default_rng(0)
    X = rng.normal(size=(8, 3))
    Y = rng.normal(size=(8, 1))

    tps = ThinPlateSpline(alpha=0.0, order=2, enforce_tps_kernel=True)  # power = 2*2 - 3 = 1
    tps.fit(X, Y)
    kernel = tps._radial_distance(X)

    # NOTE: Would still be true if enforce_tps_kernel is not set, but just to check that the code runs
    assert np.max(np.abs(np.diag(kernel))) == pytest.approx(0.0)
