"""Polynomial transform adapted from scikit-learn."""

from __future__ import annotations

from itertools import chain, combinations_with_replacement
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterable


class PolynomialFeatures:
    """Generate polynomial features.

    For each feature vector, it generates polynomial features of degree d consisting
    of all polynomial combinations of the features with degree less than or equal to
    the specified degree.

    For example, if an input sample is two dimensional and of the form
    [a, b], the degree-2 polynomial features are [1, a, b, a^2, ab, b^2].

    With degree 1, it simply transform to homogenous coordinates (adding a constant 1 to the features)

    See PolynomialFeatures from scikit-learn for a complete documentation and implementation.
    """

    def __init__(self, degree=1):
        self.degree = degree
        self._fitted = False
        self.n_input_features = 0
        self.n_output_features = 0

    @staticmethod
    def _combinations(n_features: int, degree: int) -> Iterable[tuple[int, ...]]:
        return chain.from_iterable(combinations_with_replacement(range(n_features), i) for i in range(degree + 1))

    @staticmethod
    def _combinations_length(n_features: int, degree: int) -> int:
        # TODO: Can be computed mathematically
        return sum(1 for _ in PolynomialFeatures._combinations(n_features, degree))

    def fit(self, X: np.ndarray) -> PolynomialFeatures:
        """Compute number of output features."""
        _, n_features = X.shape
        self.n_input_features = n_features
        self.n_output_features = self._combinations_length(n_features, self.degree)
        self._fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform features to polynomial features.

        Args:
            X (np.ndarray): Features for multiple samples
                Shape: (n_samples, n_features)

        Returns:
            np.ndarrray: Polynomial features
                Shape: (n_samples, n_output_features)
        """
        if not self._fitted:
            raise RuntimeError("Please call `fit` before `transform`.")

        n_samples, n_features = X.shape

        if n_features != self.n_input_features:
            raise ValueError("X shape does not match training shape")

        # allocate output data
        polynomial_features = np.empty((n_samples, self.n_output_features), dtype=X.dtype)

        combinations = self._combinations(n_features, self.degree)
        for i, c in enumerate(combinations):
            polynomial_features[:, i] = X[:, c].prod(1)

        return polynomial_features
