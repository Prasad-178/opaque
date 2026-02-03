"""
Dimensionality reduction utilities.

PCA is used when full-dimension PHE operations are too slow.
Based on Phase 1 benchmarks, if 1536-dim takes >10s for 1000 dot products,
we reduce to 256 or 128 dimensions.
"""
import numpy as np
from typing import Optional
from pathlib import Path
import pickle


class PCAReducer:
    """
    PCA-based dimensionality reduction.

    Reduces high-dimensional embeddings (e.g., 1536 for OpenAI)
    to lower dimensions suitable for PHE operations.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        whiten: bool = False,
    ):
        """
        Initialize PCA reducer.

        Args:
            input_dim: Original dimension (e.g., 1536)
            output_dim: Target dimension (e.g., 256)
            whiten: Whether to whiten the output
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.whiten = whiten

        self._mean: Optional[np.ndarray] = None
        self._components: Optional[np.ndarray] = None
        self._explained_variance: Optional[np.ndarray] = None
        self._is_fitted = False

    def fit(self, vectors: np.ndarray) -> "PCAReducer":
        """
        Fit PCA on training vectors.

        Args:
            vectors: Training vectors of shape (n_samples, input_dim)

        Returns:
            self
        """
        if vectors.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected {self.input_dim} dims, got {vectors.shape[1]}"
            )

        n_samples = len(vectors)
        
        # Adjust output_dim if we have fewer samples than requested dimensions
        # SVD can only produce min(n_samples, n_features) components
        effective_output_dim = min(self.output_dim, n_samples, self.input_dim)
        if effective_output_dim != self.output_dim:
            # Update output_dim to reflect actual number of components we can produce
            self.output_dim = effective_output_dim

        # Center the data
        self._mean = np.mean(vectors, axis=0)
        centered = vectors - self._mean

        # Compute SVD (more numerically stable than covariance for large matrices)
        # For n_samples < n_features, use full SVD
        # For n_samples >= n_features, truncated is fine
        if n_samples > self.input_dim:
            # Use randomized SVD for efficiency with large datasets
            try:
                from sklearn.decomposition import PCA
                pca = PCA(n_components=self.output_dim)
                pca.fit(centered)
                self._components = pca.components_
                self._explained_variance = pca.explained_variance_
            except ImportError:
                # Fallback to numpy SVD
                U, S, Vt = np.linalg.svd(centered, full_matrices=False)
                self._components = Vt[:self.output_dim]
                self._explained_variance = (S[:self.output_dim] ** 2) / (n_samples - 1)
        else:
            # For small datasets, use full SVD
            U, S, Vt = np.linalg.svd(centered, full_matrices=False)
            self._components = Vt[:self.output_dim]
            self._explained_variance = (S[:self.output_dim] ** 2) / max(n_samples - 1, 1)

        self._is_fitted = True
        return self

    def transform(self, vectors: np.ndarray) -> np.ndarray:
        """
        Transform vectors to lower dimension.

        Args:
            vectors: Vectors of shape (n, input_dim)

        Returns:
            Reduced vectors of shape (n, output_dim)
        """
        if not self._is_fitted:
            raise RuntimeError("PCAReducer must be fitted before transform")

        centered = vectors - self._mean
        reduced = centered @ self._components.T

        if self.whiten:
            reduced /= np.sqrt(self._explained_variance + 1e-8)

        return reduced.astype(np.float32)

    def fit_transform(self, vectors: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(vectors).transform(vectors)

    @property
    def explained_variance_ratio(self) -> Optional[np.ndarray]:
        """Proportion of variance explained by each component."""
        if self._explained_variance is None:
            return None
        total_var = np.sum(self._explained_variance)
        if total_var == 0:
            return np.zeros(len(self._explained_variance))
        return self._explained_variance / total_var

    @property
    def total_explained_variance_ratio(self) -> Optional[float]:
        """Total variance explained by all components."""
        if self.explained_variance_ratio is None:
            return None
        return float(np.sum(self.explained_variance_ratio))

    def save(self, path: Path) -> None:
        """Save fitted PCA model."""
        if not self._is_fitted:
            raise RuntimeError("Cannot save unfitted PCAReducer")

        data = {
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "whiten": self.whiten,
            "mean": self._mean,
            "components": self._components,
            "explained_variance": self._explained_variance,
        }

        with open(path, "wb") as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, path: Path) -> "PCAReducer":
        """Load fitted PCA model."""
        with open(path, "rb") as f:
            data = pickle.load(f)

        reducer = cls(
            input_dim=data["input_dim"],
            output_dim=data["output_dim"],
            whiten=data["whiten"],
        )
        reducer._mean = data["mean"]
        reducer._components = data["components"]
        reducer._explained_variance = data["explained_variance"]
        reducer._is_fitted = True

        return reducer


def determine_optimal_dimension(
    target_latency_ms: float = 2000,
    vectors_per_search: int = 1000,
    benchmark_results: Optional[dict] = None,
) -> int:
    """
    Determine optimal reduced dimension based on latency requirements.

    Args:
        target_latency_ms: Target latency for PHE dot products
        vectors_per_search: Number of vectors to score per search
        benchmark_results: Results from Phase 1 benchmark (dim -> ms_per_op)

    Returns:
        Recommended dimension
    """
    # Default estimates based on typical LightPHE performance
    # These should be updated with actual benchmark results
    default_estimates = {
        128: 1.5,   # ms per dot product
        256: 2.5,
        512: 5.0,
        1536: 15.0,
    }

    estimates = benchmark_results or default_estimates

    for dim in sorted(estimates.keys()):
        estimated_latency = estimates[dim] * vectors_per_search
        if estimated_latency <= target_latency_ms:
            return dim

    # If nothing meets target, return smallest
    return min(estimates.keys())
