import time

import matplotlib.pyplot as plt  # type: ignore[import-untyped]
import numpy as np
import tqdm  # type: ignore[import-untyped]

from tps import ThinPlateSpline


def compute_time(f, n, *args, **kwargs):
    elapsed_time = 0.0
    for _ in tqdm.trange(n):
        t = time.time()
        f(*args, **kwargs)
        elapsed_time += time.time() - t

    return elapsed_time / n


def main_timed():
    """Check time execution.

    Also shows d_s != d_t and the usage of float32 to speed up computations.
    """
    d = 3
    v = 2
    n = 1500
    m = 5000

    X_train = np.random.normal(0, 1, (n, d)).astype(np.float32)
    Y_train = np.random.normal(0, 2, (n, v)).astype(np.float32)
    X_test = np.random.normal(0, 1, (m, d)).astype(np.float32)

    tps = ThinPlateSpline()

    fit_time = compute_time(tps.fit, 100, X_train, Y_train)
    transform_time = compute_time(tps.transform, 100, X_test)

    print(f"Control point number: {n}")
    print(f"Transformed point number: {m}")
    print(f"Source and target space dimension: {d} -> {v}")
    print(f"Fit avg time: {fit_time}")
    print(f"Transform avg time: {transform_time}")


def main_interpolation():
    """Interpolates a function.

    Inspired from:
    https://scikit-learn.org/stable/auto_examples/linear_model/plot_polynomial_interpolation.html
    """

    def f(x):
        return x * np.sin(x)

    # whole range we want to plot
    x_plot = np.linspace(-1, 11, 100)
    x_train = np.linspace(0, 10, 100)
    rng = np.random.RandomState(0)  # pylint: disable=no-member
    x_train = np.sort(rng.choice(x_train, size=20, replace=False))
    y_train = f(x_train)

    # Fit the model and transform
    tps = ThinPlateSpline(0.5)
    y_pred = tps.fit(x_train, f(x_train)).transform(x_plot)[:, 0]

    # Plot everything
    plt.plot(x_plot, f(x_plot), label="ground truth")
    plt.plot(x_plot, y_pred)
    plt.scatter(x_train, y_train, label="training points")
    plt.show()


def main_surface_mapping():
    """Maps two surfaces.

    Inspired from https://github.com/tzing/tps-deformation
    """
    samp = np.linspace(-2, 2, 4)
    xx, yy = np.meshgrid(samp, samp)

    # make source surface, get uniformed distributed control points
    source_xy = np.stack([xx, yy], axis=2).reshape(-1, 2)

    # make deformed surface
    yy[:, [0, 3]] *= 2
    deform_xy = np.stack([xx, yy], axis=2).reshape(-1, 2)

    tps = ThinPlateSpline(0.5)

    # Fit the surfaces
    tps.fit(source_xy, deform_xy)

    # make other points a left-bottom to upper-right line on source surface
    samp2 = np.linspace(-1.8, 1.8, 100)
    test_xy = np.tile(samp2, [2, 1]).T

    # get transformed points
    transformed_xy = tps.transform(test_xy)

    plt.figure()
    plt.scatter(source_xy[:, 0], source_xy[:, 1])
    plt.plot(test_xy[:, 0], test_xy[:, 1], c="orange")

    plt.figure()
    plt.scatter(deform_xy[:, 0], deform_xy[:, 1])
    plt.plot(transformed_xy[:, 0], transformed_xy[:, 1], c="orange")

    plt.show()


if __name__ == "__main__":
    main_surface_mapping()
    main_interpolation()
    main_timed()
