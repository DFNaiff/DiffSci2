import numpy as np
from scipy.special import kv, gamma
from scipy.optimize import curve_fit
from scipy.spatial.distance import cdist
from scipy.interpolate import RegularGridInterpolator


def matern_covariance(r, sigma_sq, nu, length_scale):
    """
    Matérn Covariance Function.

    Parameters
    ----------
    r : array_like
        Radial distances.
    sigma_sq : float
        Amplitude (Variance) - often called theta or sigma^2.
    nu : float
        Smoothness parameter.
    length_scale : float
        Length scale parameter (l).

    Returns
    -------
    C(r) : array_like
        Covariance values.
    """
    r = np.asarray(r)

    # 1. Handle the r=0 singularity safely
    # We create a mask for r > 0.
    # At r=0, the limit of the Matern function is exactly sigma_sq.
    with np.errstate(divide='ignore', invalid='ignore'):
        # Stein's parameterization: sqrt(2*nu) * r / l
        scaled_r = (np.sqrt(2 * nu) * r) / length_scale

        # The Formula
        # factor = (2^(1-nu)) / Gamma(nu)
        factor = (2**(1.0 - nu)) / gamma(nu)

        # K_nu calculation
        # We only calculate where r > 0 to avoid RuntimeWarnings
        result = np.zeros_like(r, dtype=np.float64)

        # Compute only for non-zero distances
        mask = r > 1e-8
        if np.any(mask):
            args = scaled_r[mask]
            result[mask] = sigma_sq * factor * (args ** nu) * kv(nu, args)

    # Fill the r=0 (or very close to 0) values with the variance
    result[~mask] = sigma_sq

    return result


def fit_matern_parameters(r_data, corr_data):
    """
    Fits Matérn parameters (sigma^2, nu, l) to experimental data.
    """
    # 1. Clean Data
    # Remove NaNs or Infs if they exist
    valid = np.isfinite(corr_data)
    r_clean = r_data[valid]
    c_clean = corr_data[valid]

    # 2. Initial Guesses (p0)
    # sigma_sq: The max value of the correlation (usually at r=0)
    p0_sigma = np.max(c_clean)

    # length_scale: Distance where correlation drops to ~36% (1/e) of max
    # A rough heuristic to help the optimizer
    drop_idx = np.abs(c_clean - p0_sigma*0.36).argmin()
    p0_l = r_clean[drop_idx] if drop_idx < len(r_clean) else r_clean[-1]/2
    if p0_l == 0: p0_l = 1.0

    # nu: Start with 1.5 (intermediate smoothness)
    p0_nu = 1.5

    p0 = [p0_sigma, p0_nu, p0_l]

    # 3. Bounds
    # All parameters must be > 0.
    # nu upper bound: High nu (e.g. > 10) becomes indistinguishable from Gaussian
    # so we cap it loosely to prevent numerical instability in Bessel functions.
    lower_bounds = [1e-6, 0.1, 1e-6]
    upper_bounds = [np.inf, 30.0, np.inf]

    # 4. Optimization
    try:
        popt, pcov = curve_fit(
            matern_covariance,
            r_clean,
            c_clean,
            p0=p0,
            bounds=(lower_bounds, upper_bounds),
            maxfev=2000
        )
    except RuntimeError:
        print("Optimization failed to converge.")
        return None, None

    return popt, pcov


class MaternFieldSampler:
    def __init__(
        self,
        mean_val,
        sigma_sq,
        nu,
        length_scale,
        jitter=1e-6
    ):
        """
        Initializes the Gaussian Process with a Matérn kernel.

        Parameters
        ----------
        mean_val : float
            The constant mean of the field (mu).
        sigma_sq : float
            Amplitude (Variance) - often called theta or sigma^2.
        nu : float
            Smoothness parameter.
        length_scale : float
            Length scale parameter (l).
        jitter : float
            Small value added to diagonal for numerical stability (white noise).
        """
        self.mean_val = mean_val
        self.sigma_sq = sigma_sq
        self.nu = nu
        self.length_scale = length_scale
        self.jitter = jitter

        # Field coordinates (set via initialize_field)
        self.X = None
        self.n_points = None
        self.K = None
        self.L = None
        self.grid_shape = None  # Shape when initialized from grid
        self.grid_axes = None   # Original axes for interpolation

    def initialize_field(self, X):
        """
        Initialize the field coordinates and pre-compute covariance matrix.

        Parameters
        ----------
        X : ndarray of shape (n_points, dim)
            The spatial coordinates where the field is defined.
        """
        self.X = np.atleast_2d(X)
        self.n_points = self.X.shape[0]

        # Pre-compute the Covariance Matrix and Cholesky Decomposition
        self.K = self._build_covariance_matrix()

        # Add jitter to diagonal (K + epsilon*I) to ensure positive definiteness
        self.L = np.linalg.cholesky(self.K + np.eye(self.n_points) * self.jitter)

    def initialize_field_from_grid(self, *axes):
        """
        Initialize field coordinates from a meshgrid defined by 1D axes.

        Parameters
        ----------
        *axes : 1D arrays
            Coordinate arrays for each dimension.
            E.g., for 2D: initialize_field_from_grid(x, y)
            E.g., for 3D: initialize_field_from_grid(x, y, z)
        """
        # Store axes for later interpolation
        self.grid_axes = tuple(np.asarray(ax) for ax in axes)

        # Store grid shape: (len(x), len(y), ...) with transpose convention
        self.grid_shape = tuple(len(ax) for ax in axes)

        # Create meshgrid and flatten to (n_points, ndim)
        grids = np.meshgrid(*axes, indexing='ij')
        X = np.stack([g.ravel() for g in grids], axis=-1)

        self.initialize_field(X)

    def _matern_kernel(self, r):
        """Vectorized Matern function handling r=0 singularity"""
        # 1. Prepare result array
        result = np.zeros_like(r, dtype=np.float64)

        # 2. Handle r > 0
        # We strictly mask 0 to avoid DivByZero or Inf in Bessel calculation
        mask = r > 1e-10
        if np.any(mask):
            r_valid = r[mask]
            # Stein's parameterization
            scaled_r = (np.sqrt(2 * self.nu) * r_valid) / self.length_scale
            factor = (2**(1.0 - self.nu)) / gamma(self.nu)

            result[mask] = self.sigma_sq * factor * (scaled_r ** self.nu) * kv(self.nu, scaled_r)

        # 3. Handle r = 0 (The limit is exactly sigma^2)
        result[~mask] = self.sigma_sq
        return result

    def _build_covariance_matrix(self):
        """Computes pairwise distance and applies kernel"""
        # cdist calculates Euclidean distance between all pairs
        # dists[i, j] = ||x_i - x_j||
        dists = cdist(self.X, self.X, metric='euclidean')
        return self._matern_kernel(dists)

    def sample(self, n_samples=1):
        """
        Generates samples from the GP.

        Returns
        -------
        samples : ndarray of shape (n_samples, n_points)

        Raises
        ------
        RuntimeError
            If initialize_field() has not been called.
        """
        if self.L is None:
            raise RuntimeError(
                "Field not initialized. Call initialize_field(X) first."
            )

        # 1. Sample standard normal noise z ~ N(0, I)
        # Shape: (n_points, n_samples)
        z = np.random.normal(size=(self.n_points, n_samples))

        # 2. Apply Cholesky: y = mu + L * z
        # L is (n_points, n_points)
        y = self.mean_val + self.L @ z

        # 3. Transpose to return (n_samples, n_points)
        return y.T

    def sample_grid(self, n_samples=1):
        """
        Generates samples from the GP and reshapes to grid dimensions.

        Returns
        -------
        samples : ndarray of shape (n_samples, *grid_shape)
            E.g., for 2D grid of shape (Nx, Ny): returns (n_samples, Nx, Ny)
            E.g., for 3D grid of shape (Nx, Ny, Nz): returns (n_samples, Nx, Ny, Nz)

        Raises
        ------
        RuntimeError
            If initialize_field_from_grid() has not been called.
        """
        if self.grid_shape is None:
            raise RuntimeError(
                "Grid shape not set. Use initialize_field_from_grid() "
                "or set grid_shape manually before calling sample_grid()."
            )

        samples = self.sample(n_samples)
        return samples.reshape((n_samples,) + self.grid_shape)

    def sample_grid_interpolated(self, n_samples, *target_axes):
        """
        Sample from GP at coarse grid and interpolate to finer target grid.

        This is an efficient approximation: instead of computing the expensive
        Cholesky decomposition at the fine grid (O(n³)), we sample at the
        coarse grid and use linear interpolation.

        Parameters
        ----------
        n_samples : int
            Number of samples to generate.
        *target_axes : 1D arrays
            Coordinate arrays for the target (fine) grid.
            Must have the same number of dimensions as the initialized grid.

        Returns
        -------
        samples : ndarray of shape (n_samples, *target_shape)
            Interpolated samples on the fine grid.

        Raises
        ------
        RuntimeError
            If initialize_field_from_grid() has not been called.
        ValueError
            If number of target axes doesn't match grid dimensions.
        """
        if self.grid_axes is None:
            raise RuntimeError(
                "Grid axes not set. Use initialize_field_from_grid() first."
            )

        if len(target_axes) != len(self.grid_axes):
            raise ValueError(
                f"Expected {len(self.grid_axes)} axes, got {len(target_axes)}"
            )

        # Sample at coarse grid
        coarse_samples = self.sample_grid(n_samples)

        # Target grid shape
        target_shape = tuple(len(ax) for ax in target_axes)

        # Create target points for interpolation
        target_grids = np.meshgrid(*target_axes, indexing='ij')
        target_points = np.stack([g.ravel() for g in target_grids], axis=-1)

        # Interpolate each sample
        result = np.empty((n_samples,) + target_shape)
        for i in range(n_samples):
            interpolator = RegularGridInterpolator(
                self.grid_axes,
                coarse_samples[i],
                method='linear',
                bounds_error=False,
                fill_value=None  # Extrapolate
            )
            result[i] = interpolator(target_points).reshape(target_shape)

        return result