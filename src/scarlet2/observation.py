import astropy.units as u
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from .bbox import Box, insert_into, overlap_slices
from .fft import transform
from .frame import Frame, get_affine, get_pixel_size
from .measure import correlation_function
from .module import Module
from .renderer import (
    ChannelTransformation,
    ConvolutionTransformation,
    LanczosResamplingTranformation,
    Renderer,
    ResamplingTransformation,
    SpatialTrimTransformation,
)
from .validation_utils import (
    ValidationError,
    ValidationInfo,
    ValidationMethodCollector,
    ValidationResult,
    ValidationWarning,
    print_validation_results,
)


class Observation(Module):
    """Content and definition of an observation"""

    data: jnp.ndarray
    """Observed data"""
    weights: jnp.ndarray
    """Statistical weights (usually inverse variance) for :py:meth:`log_likelihood`"""
    frame: Frame
    """Metadata to describe what view of the sky `data` amounts to"""
    renderer: Renderer
    """Renderer to translate from the model frame the observation frame"""
    name: str
    """Name to describe the observation"""

    def __init__(self, data, weights, psf=None, wcs=None, channels=None, renderer=None, name=None):
        self.data = jnp.asarray(data, dtype=float)
        if self.data.ndim == 2:
            # add a channel dimension if it is missing
            self.data = self.data[None, ...]

        self.weights = jnp.asarray(weights, dtype=float)
        if self.weights is not None and self.weights.ndim == 2:
            # add a channel dimension if it is missing
            self.weights = self.weights[None, ...]

        self.frame = Frame(Box(self.data.shape), psf, wcs, channels=channels)
        self.renderer = renderer
        self.name = name if name is not None else ""

        # (re)-import `VALIDATION_MODE` at runtime to avoid using a static/old value
        from .validation_utils import VALIDATION_MODE

        if VALIDATION_MODE != "off":
            from .validation import check_observation

            validation_results = check_observation(self)
            print_validation_results(
                "Observation validation results",
                validation_results,
                verbose=VALIDATION_MODE == "verbose",
            )

    @property
    def N(self):  # noqa: N802
        """Number of unmasked pixels in the observation"""
        return jnp.prod(jnp.asarray(self.data.shape)) - jnp.sum(self.weights == 0)

    def render(self, model, **kwargs):
        """Render `model` in the frame of this observation

        Parameters
        ----------
        model: array
            The (pre-rendered) predicted data cube, typically from evaluating :py:class:`~scarlet2.Scene`
        kwargs: dict
            Additional keyword arguments to pass to the renderer

        Returns
        -------
        array
            Prediction of the observation given the `model`. Has the same shape as :py:attr:`data`.
        """
        assert self.renderer is not None, (
            "Observation.render() requires a renderer. Call Observation.match(model_frame) first"
        )
        model_ = model() if isinstance(model, Module) else model
        return self.renderer(model_, **kwargs)

    def log_likelihood(self, model):
        """The logarithm the likelihood of :py:attr:`data` given `model`

        Parameters
        ----------
        model: array
            The (pre-rendered) predicted data cube, typically from evaluating :py:class:`~scarlet2.Scene`

        Returns
        -------
        float
        """

        return self._log_likelihood(model, self.data)

    def _log_likelihood(self, model, data):
        # normalization of the single-pixel likelihood:
        # 1 / [(2pi)^1/2 (sigma^2)^1/2]
        # with inverse variance weights: sigma^2 = 1/weight
        # full likelihood is sum over all (unmasked) pixels in data
        log_like = -self._chisquare(model) / 2
        log_norm = self.N / 2 * jnp.log(2 * jnp.pi)
        return log_like - log_norm

    def goodness_of_fit(self, model):
        """Evaluate the goodness of the model fit to the data

        For a Gaussian noise model, the gof is defined as the averaged squared deviation of the model from the
        data, scaled by the variance of the data, aka mean chi squared
        :math:`\\frac{1}{N}\\sum_i=1^N w_i (m_i - d_i)^2` with inverse variance weights :math:`w_i`.

        Up to a normalization, the gof is identical to :py:class:`~scarlet2.Observation.log-likelihood`.

        Parameters
        ----------
        model: array
            The (pre-rendered) predicted data cube, typically from evaluating :py:class:`~scarlet2.Scene`

        Returns
        -------
        float
        """
        # only use unmasked pixels in the data
        return self._chisquare(model) / self.N

    def _chisquare(self, model):
        return jnp.sum(self.weights * (self.render(model) - self.data) ** 2)

    def check_set_renderer(self, frame):
        """Check existence of :py:attr:`renderer`, or set it by calling :py:meth:`match`

        Parameters
        ----------
        frame: Frame
            The frame to match

        Returns
        -------
        None
        """
        if self.renderer is None:
            self.match(frame)

    def match(self, frame, renderer=None):
        """Construct the mapping between `frame` (from the model) and this observation frame

        Parameters
        ----------
        frame: Frame
            Model frame, typically :py:attr:`scarlet2.Scene.frame` for the current scene.
        renderer: Renderer, optional
            Custom transformation to translate the `frame` (from the model) to this observation frame.
            If not set, this method will attempt to create the mapping from the information in both frames.

        Returns
        -------
        None
        """
        # choose the renderer
        if renderer is None:
            renderers = []

            # note the order of renderers!
            # 1) match channels of frame
            if self.frame.channels != frame.channels:
                renderers.append(ChannelTransformation(frame, self.frame))

            # 2) match spatial properties of frame
            # if image has pixel grid (modulo an integer shift), avoid resampling
            m_self = get_affine(self.frame.wcs)
            m_frame = get_affine(frame.wcs)
            same_matrix = jnp.allclose(m_self, m_frame)

            ref_pixel = jnp.array(self.frame.bbox.spatial.origin)
            shift = frame.get_pixel(self.frame.get_sky_coord(ref_pixel)) - ref_pixel
            integer_shift = jnp.allclose(shift, jnp.round(shift), atol=1e-3)

            if same_matrix and integer_shift:
                if self.frame.psf != frame.psf:
                    renderers.append(ConvolutionTransformation(frame, self.frame))
                if self.frame.bbox.spatial != frame.bbox.spatial:
                    renderers.append(SpatialTrimTransformation(frame, self.frame))
            else:
                renderers.append(ResamplingTransformation(frame, self.frame))
            renderer = Renderer(renderers)
        else:
            assert isinstance(renderer, Renderer)
            # TODO: avoid call to renderer, use validator instead
            assert renderer(jnp.zeros(frame.bbox.shape)).shape == self.frame.bbox.shape, (
                "Renderer does not map model frame to observation frame"
            )
        object.__setattr__(self, "renderer", renderer)

    def eval_chi_square_in_box_and_border(self, scene, border_width=3):
        """
        Evaluate the weighted mean (weighted by the inverse variance weights) of the squared residuals
        for each source. Chi square is also computed for the perimeter outside the box of with `border_width`.

        Parameters
        ----------
        scene: :py:class:`~scarlet2.Scene`
            Scene containing the sources
        border_width: int
            width of the border around the source box

        Returns
        -------
        Dict of sources indices and their corresponding Dict of residuals inside and outside source box.
        """
        # TODO: combine with chi_square_in_box_and_border and move this to output validation tests (#148)
        residuals = self.render(scene()) - self.data

        chi_dict = {}
        for i, src in enumerate(scene.sources):
            bbox, _ = overlap_slices(self.frame.bbox, src.bbox, return_boxes=True)
            chi_in, chi_out = chi_square_in_box_and_border(residuals, self.weights, bbox, border_width)
            chi_dict[i] = {"in": chi_in, "out": chi_out}

        return chi_dict


def _noise_kernel(xi):
    channels = len(xi[0, 0])
    maxlength = max(max(k) for k in xi)
    kernel = jnp.zeros((channels, 2 * maxlength + 1, 2 * maxlength + 1))
    for k in xi:
        dy, dx = k
        kernel = kernel.at[:, dy + maxlength, dx + maxlength].set(xi[k])
    return kernel


def _parzen_window(maxlength):
    # Parzen lag window, separable in y and x. Truncating the correlation function at `maxlength` is
    # equivalent to a rectangular lag window, which is not positive semi-definite: the resulting power
    # spectrum has negative modes, and the ones near the zero crossing then dominate chi^2. The Parzen
    # window is positive semi-definite, so the spectrum stays non-negative. It trades that for a bias
    # towards smoother spectra, which shrinks as `maxlength` grows.
    u = jnp.arange(-maxlength, maxlength + 1) / (maxlength + 1)
    w = jnp.where(jnp.abs(u) <= 0.5, 1 - 6 * u**2 + 6 * jnp.abs(u) ** 3, 2 * (1 - jnp.abs(u)) ** 3)
    return w[:, None] * w[None, :]


def _power_spectrum_from(xi, shape, floor=1e-6):
    # NOTE: this conversion is not ideal because the correlation function is likely undersampled
    # Better would be a pure correlated noise field to measure the power spectrum directly
    kernel = _noise_kernel(xi)
    maxlength = (kernel.shape[-1] - 1) // 2
    kernel = kernel * _parzen_window(maxlength)
    # the tapered correlation function is positive semi-definite, so the spectrum should be non-negative.
    # But that only holds if the *measured* xi is itself positive semi-definite, which it need not be:
    # for sharply band-limited noise (e.g. after upsampling) the spectrum still goes negative, and the
    # near-zero modes then dominate chi^2. Clip to `floor` times the per-channel maximum so the spectrum
    # is strictly positive and no mode can be weighted by more than 1 / floor relative to the peak.
    ps = transform(kernel, shape[-2:], axes=(-2, -1)).real
    return jnp.maximum(ps, floor * ps.max(axis=(-2, -1), keepdims=True))


def _padded_power_spectrum_bartlett(power_spectrum, shape, fft_shape, max_dynamic_range=1e12):
    # Exact form of the padded power spectrum, P(k) = 1/N sum_lag L(lag) xi(lag) exp(-i k lag), where
    # L(lag) = (H - |lag_y|)(W - |lag_x|) counts the pixel pairs of the data box at that lag. This is a
    # Fejer/Bartlett smoothing of `power_spectrum`, and it takes a single transform instead of the many
    # realizations that `_padded_power_spectrum_mc` needs.
    # The lag sum has terms of order N * xi(0) while its result in the tail is many orders of magnitude
    # smaller, so it cancels catastrophically once the dynamic range of `power_spectrum` approaches the
    # floating point precision. Hence float64, plus the two guards below; `None` asks for the fallback.
    height, width = shape[-2:]
    m_0, m_1 = fft_shape[-2:]
    ps = np.asarray(power_spectrum, dtype=np.float64)
    if ps.min() <= 0 or ps.max() / ps.min() > max_dynamic_range:
        return None

    xi = np.fft.irfft2(ps, s=(height, width), axes=(-2, -1))
    # smallest lag grid that holds every lag of the data box and whose frequencies are a superset of
    # those of `fft_shape`, so that the result can be subsampled onto the latter
    g_0 = m_0 * int(np.ceil((2 * height - 1) / m_0))
    g_1 = m_1 * int(np.ceil((2 * width - 1) / m_1))

    def _lags(g, n):
        lag = np.arange(g)
        lag = np.where(lag <= g // 2, lag, lag - g)
        return np.where(np.abs(lag) < n, n - np.abs(lag), 0).astype(np.float64), lag % n

    pairs_y, index_y = _lags(g_0, height)
    pairs_x, index_x = _lags(g_1, width)
    xi = xi[..., index_y[:, None], index_x[None, :]] * (pairs_y[:, None] * pairs_x[None, :])
    ps_padded = np.fft.rfft2(xi, axes=(-2, -1)).real[..., :: g_0 // m_0, :: g_1 // m_1]
    ps_padded /= height * width

    if ps_padded.min() <= 0:  # residual cancellation that the dynamic range guard did not catch
        return None
    return jnp.asarray(ps_padded, dtype=power_spectrum.dtype)


def _padded_power_spectrum_mc(power_spectrum, shape, fft_shape, n_realizations=64, seed=0):
    # Measure the padded power spectrum from noise realizations drawn with `power_spectrum` and
    # zero-padded exactly like the residual. Slower and noisier than the Bartlett form, but immune to its
    # cancellation: a periodogram is a sum of squares, and it carries the same numerical floor as the
    # residual it will be divided into, so their ratio stays sane even where both are meaningless.
    height, width = shape[-2:]
    amplitude = jnp.sqrt(power_spectrum)

    def _periodogram(key):
        # filtering white noise with sqrt(power_spectrum) yields a field with exactly that spectrum
        white = jnp.fft.rfft2(jax.random.normal(key, shape=shape), axes=(-2, -1))
        noise = jnp.fft.irfft2(white * amplitude, s=(height, width), axes=(-2, -1))
        return jnp.abs(transform(noise, fft_shape[-2:], axes=(-2, -1))) ** 2

    keys = jax.random.split(jax.random.key(seed), n_realizations)
    # lax.map is sequential, so memory stays at one noise field regardless of n_realizations
    return jax.lax.map(_periodogram, keys).mean(axis=0) / (height * width)


def _padded_power_spectrum(power_spectrum, shape, fft_shape, n_realizations=64, seed=0):
    """Noise power spectrum for residuals that are zero-padded from `shape` to `fft_shape`

    Zero-padding is not a stationary operation: the modes of the padded residual are correlated, and
    their variance is not the power spectrum evaluated on the padded grid. Using the latter overweights
    the leakage into high-k modes by many orders of magnitude, because those are exactly the modes where
    the power spectrum is smallest.

    The normalization is by the number of data pixels rather than padded pixels, which accounts for the
    fact that only those carry data.

    Parameters
    ----------
    power_spectrum: array
        Power spectrum on the data grid, in `rfft2` layout
    shape: tuple
        Shape of the data
    fft_shape: tuple
        Shape of the padded grid the residual is transformed on
    n_realizations: int
        Number of noise realizations for the Monte Carlo fallback. Has no effect unless `power_spectrum`
        has too large a dynamic range for the exact form to be computed.
    seed: int
        Seed of the noise realizations of the Monte Carlo fallback

    Returns
    -------
    array
        Power spectrum in `rfft2` layout of `fft_shape`
    """
    ps = _padded_power_spectrum_bartlett(power_spectrum, shape, fft_shape)
    if ps is not None:
        return ps
    return _padded_power_spectrum_mc(power_spectrum, shape, fft_shape, n_realizations, seed)


def _informative_mode_mask(power_spectrum, n_keep):
    """Boolean `rfft2` mask keeping the `n_keep` highest-power modes.

    A power spectrum measured from resampled noise falls by orders of magnitude past the original
    Nyquist frequency. Those modes carry only interpolation leakage, no independent noise, so summing
    chi^2 over them only adds unmodeled variance. Because every leakage mode sits below every genuine
    mode, keeping the `n_keep` strongest ones selects exactly the informative band, whatever its
    (possibly rotated) shape. `n_keep` is set by the caller so the retained modes carry ~`n_eff`
    degrees of freedom.
    """
    ps = np.asarray(power_spectrum)
    n_keep = int(np.clip(n_keep, 1, ps.size))
    keep = np.argpartition(ps.reshape(-1), ps.size - n_keep)[ps.size - n_keep :]
    mask = np.zeros(ps.size, dtype=bool)
    mask[keep] = True
    return jnp.asarray(mask.reshape(ps.shape))


class CorrelatedObservation(Observation):
    """Content and definition of an observation with pixel correlations

    The noise model is still assumed to be Gaussian, but with correlations between pixels.
    The implementation computes the goodness of fit in Fourier space from the noise power spectrum to avoid
    the expensive computation of/with an inverse banded matrix in configuration space.
    """

    power_spectrum: jnp.ndarray
    """Noise power spectrum on the data grid, in `rfft2` layout

    Has shape `(C, H, W // 2 + 1)` for data of shape `(C, H, W)`.
    """
    mask: jnp.ndarray
    """Mask for invalid pixels"""
    n_eff: int = eqx.field(static=True)
    """Effective number of degrees of freedom of the noise model, see :py:attr:`N`"""
    _data_fft: jnp.ndarray = eqx.field(repr=False)
    """:py:attr:`data`, transformed onto the padded grid. `None` whenever `_power_spectrum_padded` is."""
    _power_spectrum_padded: jnp.ndarray = eqx.field(repr=False)
    """Power spectrum on the padded grid of a trailing :py:class:`~scarlet2.ConvolutionTransformation`

    `None` unless :py:meth:`_match_power_spectrum` found that the faster Fourier-space path applies.
    """
    _mode_mask: jnp.ndarray = eqx.field(repr=False)
    """`rfft2` mask of the modes that enter chi^2, on the grid `_chisquare` evaluates on.

    `None` unless the noise model is rank-deficient (resampled onto a finer grid), in which case the
    modes above the original Nyquist hold only resampling leakage and are excluded from the sum.
    """

    def __init__(
        self,
        data,
        psf=None,
        wcs=None,
        channels=None,
        renderer=None,
        name="",
        power_spectrum=None,
        correlation_function=None,
        mask=None,
        native_scale=None,
    ):
        """Create an observation with a correlated-noise Gaussian likelihood.

        Provide the noise model either as `power_spectrum` (on the data grid) or as a measured
        `correlation_function`; exactly one is required. In most cases it is easier to use
        :py:meth:`from_observation` (for a coadd already on the desired grid) or
        :py:meth:`from_resampling` (to resample the data here).

        Parameters
        ----------
        data: array
            Observed data, 2D or 3D `(C, H, W)`.
        psf, wcs, channels, renderer, name:
            As for :py:class:`Observation`.
        power_spectrum: array, optional
            Noise power spectrum on the data grid, in `rfft2` layout, shape `(C, H, W // 2 + 1)`.
        correlation_function: dict, optional
            2D pixel correlation function keyed by integer `(dy, dx)` lag, as returned by
            :py:func:`scarlet2.measure.correlation_function`.
        mask: array, optional
            Boolean array marking invalid pixels. Defaults to all-valid.
        native_scale: :py:class:`astropy.units.Quantity`, optional
            Angular size of a native, pre-resampling pixel, e.g. ``0.17 * u.arcsec``. Set this if the
            data sit on a grid finer than their true resolution (resampled by an upstream pipeline or by
            :py:meth:`from_resampling`). Only ``1 / (native_scale / pixel_scale) ** 2`` of the pixels are
            then independent, and chi^2 is restricted to the spatial frequencies the native sampling
            supports. Requires `wcs`.
        """
        data = jnp.asarray(data, dtype=float)
        if data.ndim == 2:
            data = data[None, ...]

        assert (power_spectrum is None) != (correlation_function is None), (
            "Provide either power_spectrum or correlation_function"
        )
        # weights ignore pixel covariance: per-pixel variance only, i.e. the zero-lag correlation xi(0,0).
        # Prefer the measured value when the correlation function is given: it is the direct estimate,
        # whereas inverting the power spectrum picks up the distortion that the taper introduces there.
        if power_spectrum is None:
            variance = correlation_function[0, 0]
            power_spectrum = _power_spectrum_from(correlation_function, data.shape)
        else:
            variance = jnp.fft.irfft2(power_spectrum, s=data.shape[-2:], axes=(-2, -1))[..., 0, 0]
        self.power_spectrum = jnp.asarray(power_spectrum, dtype=float)
        assert self.power_spectrum.shape == data.shape[:-1] + (data.shape[-1] // 2 + 1,), (
            f"power_spectrum shape {self.power_spectrum.shape} does not match data shape {data.shape}"
        )

        self.mask = mask if mask is not None else jnp.zeros(data.shape, dtype=bool)
        weights = jnp.ones(data.shape) / variance[:, None, None] * ~self.mask

        # effective degrees of freedom: every unmasked pixel, unless the data sit on a grid finer than
        # their native resolution (`native_scale`). Then the noise covariance is rank-deficient and only
        # 1 / oversampling^2 of the pixels carry independent information.
        n_unmasked = int(data.size - jnp.sum(self.mask))
        self.n_eff = n_unmasked
        if native_scale is not None:
            assert wcs is not None, "native_scale requires a wcs"
            assert u.get_physical_type(native_scale) == "angle", (
                "native_scale must be an astropy angle Quantity, e.g. 0.17 * u.arcsec"
            )
            oversampling = float((native_scale / get_pixel_size(wcs)).to_value(u.dimensionless_unscaled))
            if oversampling > 1.01:  # ignore a native scale within rounding of the delivered one
                self.n_eff = max(1, round(n_unmasked / oversampling**2))

        self._power_spectrum_padded = None
        self._data_fft = None
        self._mode_mask = None
        super().__init__(data, weights, psf=psf, wcs=wcs, channels=channels, renderer=renderer, name=name)
        if self.renderer is not None:
            self._match_power_spectrum()

    @property
    def N(self):  # noqa: N802
        """Effective number of degrees of freedom of the noise model

        For an observation without resampling this is the number of unmasked pixels, as in
        :py:attr:`Observation.N`. When the correlations come from resampling onto a finer grid, the noise
        covariance is rank-deficient: there are only as many independent noise values as unmasked pixels
        in the *original* observation. Using the pixel count there would dilute :py:meth:`goodness_of_fit`
        and the likelihood normalization by the surplus modes, which carry no information.
        """
        return self.n_eff

    def match(self, frame, renderer=None):
        """Construct the mapping between `frame` (from the model) and this observation frame

        See :py:meth:`Observation.match`. In addition, this method determines whether chi^2 can be
        evaluated on the padded grid of the renderer.
        """
        super().match(frame, renderer=renderer)
        self._match_power_spectrum()

    def _match_power_spectrum(self):
        # If the last transformation is a convolution, it can hand us the model in Fourier space on its
        # padded grid, which saves the inverse transform and the transform back for every likelihood
        # evaluation. chi^2 is then evaluated there, against a power spectrum that accounts for the data
        # being zero-padded onto the same grid.
        # Masked pixels rule this out: masking is a real-space operation, so the model cannot be masked in
        # Fourier space, and masked pixels would score the model against zero data.
        _renderer = self.renderer[-1]
        power_spectrum, data_fft = None, None
        if isinstance(_renderer, ConvolutionTransformation) and not self.mask.any():
            fft_shape = tuple(_renderer._fft_shape)
            power_spectrum = _padded_power_spectrum(self.power_spectrum, self.data.shape, fft_shape)
            data_fft = transform(self.data, fft_shape, axes=(-2, -1))
        object.__setattr__(self, "_power_spectrum_padded", power_spectrum)
        object.__setattr__(self, "_data_fft", data_fft)

        # A rank-deficient (resampled) noise model has independent noise only in the modes below the
        # original Nyquist; the rest is resampling leakage and must not enter chi^2. Build the mask on
        # whichever grid `_chisquare` evaluates on, keeping just enough modes to carry `n_eff` degrees
        # of freedom so that `goodness_of_fit` stays ~1 for a calibrated fit. Every retained mode
        # contributes on average `2` to chi^2 on the data grid, and `2 * n_pix / n_pad` on the padded
        # grid (where the padded power spectrum is normalized by the data pixel count, not `n_pad`).
        mode_mask = None
        if self.n_eff < int(self.data.size - jnp.sum(self.mask)):
            n_pix = int(np.prod(self.data.shape[-2:]))
            if power_spectrum is not None:
                n_keep = round(self.n_eff * int(np.prod(fft_shape[-2:])) / (2 * n_pix))
                mode_mask = _informative_mode_mask(power_spectrum, n_keep)
            else:
                mode_mask = _informative_mode_mask(self.power_spectrum, round(self.n_eff / 2))
        object.__setattr__(self, "_mode_mask", mode_mask)

    def _chisquare(self, model):
        # NOTE: when `power_spectrum` was built from resampled noise (see `from_observation`), the modes
        # above the original Nyquist carry no independent information. `_mode_mask` drops them from the
        # sum below. It also sidesteps the fact that convolution and resampling do not commute in that
        # band, so the render and the noise model would otherwise disagree there.
        if self._power_spectrum_padded is not None:
            # take the model straight from the convolution in Fourier space and zero-pad the data onto the
            # same grid, so that the residual there is the zero-padded data-grid residual that
            # `_padded_power_spectrum` is matched to
            model_fft = self.render(model, return_fft=True)
            n_pad = jnp.prod(jnp.asarray(self.renderer[-1]._fft_shape))
            res_fft = (model_fft - self._data_fft) / jnp.sqrt(n_pad / 2)
            chi2_modes = (res_fft * jnp.conjugate(res_fft)).real / self._power_spectrum_padded
            if self._mode_mask is not None:
                chi2_modes = chi2_modes * self._mode_mask
            return jnp.sum(chi2_modes)

        # The Fourier-space chi^2 diagonalizes the noise covariance only if that covariance is circulant
        # on the transformed grid, which holds on the data grid alone. Renderers work on a padded
        # `_fft_shape` to suppress convolution wrap-around; transforming the residual on that grid breaks
        # stationarity (the padding zeros are not data) and mismatches the sqrt(N/2) Parseval factor.
        # The residual therefore has to be brought back to the data grid before the FFT.
        res = ~self.mask * (self.render(model) - self.data)
        # normalization sqrt(n_pix / 2) added because it's missing in numpy/jax forward fft.
        # This is the *spatial* pixel count: rfft2 transforms every channel separately, so Parseval
        # applies per channel. Using self.N here would scale chi^2 by 1 / n_channels. Masked pixels need
        # no accounting either, they carry zero residual and drop out of the transform on their own.
        n_pix = jnp.prod(jnp.asarray(self.data.shape[-2:]))
        res_fft = jnp.fft.rfft2(res, axes=(-2, -1)) / jnp.sqrt(n_pix / 2)
        chi2_modes = (res_fft * jnp.conjugate(res_fft)).real / self.power_spectrum
        if self._mode_mask is not None:
            chi2_modes = chi2_modes * self._mode_mask
        return jnp.sum(chi2_modes)

    @classmethod
    def from_observation(
        cls,
        obs,
        patch_size=50,
        maxlength=12,
        native_scale=None,
    ):
        """Create a :py:class:`CorrelatedObservation` from data that already sit on the desired grid.

        Use this for a coadd (drizzled, warped, stacked, ...) whose pixel grid matches the model frame.
        The method finds a patch of size `patch_size` with as few sources as possible, measures the 2D
        pixel correlation function in it (out to `maxlength`), and turns that into the noise power
        spectrum used by the likelihood. Truncating the correlation function at `maxlength` biases the
        power spectrum, severely so once the correlation length approaches `maxlength`.

        To resample the data onto the model grid here instead, use :py:meth:`from_resampling`.

        Parameters
        ----------
        obs: :py:class:`Observation`
            Observation containing the data and weight map. The weights are used only to locate bad and
            bright pixels for the correlation measurement.
        patch_size: int
            Linear size (in pixels) of the source-free patch used to measure the correlation function.
            Clamped to the image size; if it reaches the image size the whole image is used.
        maxlength: int
            Maximum lag (in pixels) of the 2D correlation function. Large enough to cover the extent of
            the correlations, and small compared to `patch_size` so every lag is averaged over many
            pixel pairs (the assertion enforces ``4 * maxlength <= patch_size``).
        native_scale: :py:class:`astropy.units.Quantity`, optional
            Angular size of a native, pre-resampling pixel, e.g. ``0.17 * u.arcsec``. Set this only if
            an upstream pipeline already resampled the data onto a grid finer than their true
            resolution: the pixels are then correlated *and* not all independent, so chi^2 is restricted
            to the spatial frequencies the native sampling supports. See
            :py:meth:`CorrelatedObservation.__init__`.

        Returns
        -------
        :py:class:`CorrelatedObservation`
        """
        # compute the pixel correlations in a noisy patch (without correlations from sources)
        # 1) mask pixels with bright pixels or zero weights
        data = obs.data
        mask = obs.weights == 0
        mask = mask.at[~mask].set(data[~mask] > 3 * jnp.sqrt(1 / obs.weights[~mask]))
        # extend the mask to remove most of the outskirts of detected galaxies
        kernel = jnp.ones((9, 9))
        _correlate2d = lambda x, kernel: jax.scipy.signal.correlate2d(x, kernel, mode="same")
        correlate3d = jax.vmap(_correlate2d, in_axes=(0, None), out_axes=0)
        mask = correlate3d(mask, kernel) > 0

        # 2) find patch of size length (at most image size) with the largest number of unmasked pixels
        patch_size = min(patch_size, min(data.shape[-2:]))
        assert 4 * maxlength <= patch_size, (
            f"maxlength={maxlength} is too large for patch_size={patch_size}: the longest lags would "
            "be averaged over too few pixel pairs"
        )
        if patch_size >= min(data.shape[-2:]):
            # the patch already covers the whole image: skip the search, `correlation_function`
            # accounts for the masked pixels through its pair count
            patch, patch_mask = data, mask
        else:
            kernel = jnp.ones((patch_size, patch_size))
            # correlated with tophat = sliding count of unmasked pixels
            gaps = correlate3d(~mask, kernel)

            # location of lower-left pixel of the patch with the fewest masked pixels
            def best_patch(img, msk, gaps):
                # trim off patch_size // 2 so the patch center stays away from the image border
                trimmed_shape = tuple(s - patch_size for s in gaps.shape[-2:])
                y, x = jnp.unravel_index(
                    jnp.argmax(gaps[patch_size // 2 : -patch_size // 2, patch_size // 2 : -patch_size // 2]),
                    trimmed_shape,
                )
                slice_shape = (patch_size, patch_size)
                return (
                    jax.lax.dynamic_slice(img, (y, x), slice_shape),
                    jax.lax.dynamic_slice(msk, (y, x), slice_shape),
                )

            patch, patch_mask = jax.vmap(best_patch, in_axes=(0, 0, 0), out_axes=0)(data, mask, gaps)

        # 3) measure correlation function in the patch
        # there is no generative noise model here, so the power spectrum cannot be averaged over
        # realizations and has to be estimated from the correlation function. Its pair count excludes
        # every lag pair that touches a masked pixel, which a periodogram of the patch would not.
        xi = correlation_function(patch, maxlength=maxlength, mask=patch_mask)

        # define the remaining items
        psf = obs.frame.psf
        wcs = obs.frame.wcs
        renderer = obs.renderer if native_scale is None else None
        mask = obs.weights == 0

        return CorrelatedObservation(
            data,
            mask=mask,
            psf=psf,
            wcs=wcs,
            renderer=renderer,
            correlation_function=xi,
            channels=obs.frame.channels,
            name=obs.name,
            native_scale=native_scale,
        )

    @classmethod
    def from_resampling(
        cls,
        obs,
        to_frame,
        lanczos_order=9,
        resample_psf=True,
        compute_power_spectrum=False,
        patch_size=50,
        maxlength=12,
        n_realizations=64,
        batch_size=8,
    ):
        """Create a :py:class:`CorrelatedObservation` by resampling `obs` onto the grid of `to_frame`.

        The Lanczos resampling correlates the pixels, so the result needs a correlated-noise
        likelihood. Two ways to get the noise power spectrum:

        * ``compute_power_spectrum=False`` (default): measure the correlation function of the resampled
          data via :py:meth:`from_observation`. This is the general route -- it also works if `obs` was
          itself a coadd with its own (typically shorter-range) correlations.
        * ``compute_power_spectrum=True``: assume the *original* pixels were uncorrelated and build the
          power spectrum generatively, by averaging periodograms of `n_realizations` noise fields drawn
          from `obs.weights` and pushed through the same resampling. Tighter and cheaper to tune, but
          only valid for genuinely uncorrelated input.

        Either way, `to_frame` is finer than `obs` in the typical (upsampling) case, so the resampled
        noise is rank-deficient: chi^2 is automatically restricted to the frequencies the native
        (`obs`) sampling supports.

        Parameters
        ----------
        obs: :py:class:`Observation`
            Observation to resample. Its pixel scale is taken as the native resolution.
        to_frame: :py:class:`~scarlet2.Frame`
            Frame describing the desired spatial sampling, typically the model frame.
        lanczos_order: int
            Order of the Lanczos resampling kernel.
        resample_psf: bool, optional
            Whether to resample `obs.psf` to `to_frame`. Set to False only if the PSF is already
            sampled at the resolution of `to_frame`.
        compute_power_spectrum: bool, optional
            Use the generative periodogram route (see above) instead of the correlation function.
        patch_size, maxlength: int
            Passed to :py:meth:`from_observation`. No effect if `compute_power_spectrum` is True.
        n_realizations, batch_size: int
            Number of noise fields averaged into the generative power spectrum, and how many are
            resampled at once. No effect unless `compute_power_spectrum` is True.

        Returns
        -------
        :py:class:`CorrelatedObservation`
        """
        # reverse renderer without PSF corrections or channel filtering: obs grid -> to_frame grid
        _obs_frame = Frame(obs.frame.bbox, psf=None, wcs=obs.frame.wcs, channels=obs.frame.channels)
        _new_box = obs.frame.bbox[:-2] @ to_frame.bbox.spatial
        _model_frame = Frame(_new_box, psf=None, wcs=to_frame.wcs, channels=obs.frame.channels)
        trafo = LanczosResamplingTranformation(_obs_frame, _model_frame, lanczos_order=lanczos_order)
        wcs = to_frame.wcs
        # the data's native resolution is the pre-resampling pixel scale
        native_scale = get_pixel_size(obs.frame.wcs)

        # resample data, mask plane, PSF, and (on the default noise-model path) one noise field in a
        # single pass. They all sit on obs's grid and `trafo` resamples each channel independently, so
        # concatenating them along the channel axis and resampling once is equivalent to -- and several
        # times faster than -- the separate calls it replaces.
        psf_image = obs.frame.psf()
        if psf_image.ndim == 2:
            # a single-band PSF (e.g. GaussianPSF) needs the channel dimension to line up with the data
            psf_image = jnp.tile(psf_image, (obs.data.shape[0], 1, 1))

        # per-pixel noise sigma on the original grid, masked pixels filled with the median so the noise
        # field stays stationary under resampling
        sigma = jnp.where(obs.weights > 0, 1 / jnp.sqrt(jnp.where(obs.weights > 0, obs.weights, 1)), 0)
        sigma = jnp.where(sigma > 0, sigma, jnp.median(sigma))
        key = jax.random.key(hash(obs.frame))

        n_c = obs.data.shape[0]
        to_resample = [obs.data, jnp.asarray(obs.weights == 0, dtype=float)]
        if resample_psf:
            # insert the PSF into the middle of an obs-sized image before resampling
            full_psf_image = jnp.zeros(obs.data.shape)
            full_box = Box(full_psf_image.shape)
            shift = tuple(full_psf_image.shape[d] // 2 - psf_image.shape[d] // 2 for d in range(full_box.D))
            psf_box = Box(psf_image.shape) + shift
            to_resample.append(insert_into(full_psf_image, psf_image, psf_box))
        if not compute_power_spectrum:
            to_resample.append(jax.random.normal(key, obs.data.shape) * sigma)

        resampled = trafo(jnp.concatenate(to_resample, axis=0))
        data = resampled[:n_c]
        mask = resampled[n_c : 2 * n_c] > 0.3  # mask edges blur under resampling, keep fractional
        next_c = 2 * n_c
        if resample_psf:
            psf = resampled[next_c : next_c + n_c]
            next_c += n_c
        else:
            psf = psf_image

        if not compute_power_spectrum:
            # from_observation locates bad and bright (i.e. source) pixels from the weight map, and needs
            # the *resampled* noise level to do so. That is not trafo(obs.weights) -- resampling does not
            # propagate inverse variance -- so estimate it per channel from one resampled noise field.
            sigma_resampled = jnp.std(resampled[next_c : next_c + n_c], axis=(-2, -1), keepdims=True)
            weights = jnp.where(mask, 0.0, 1 / sigma_resampled**2)
            obs_ = Observation(
                data,
                weights,
                psf=psf,
                wcs=wcs,
                channels=obs.frame.channels,
                name=obs.name,
                renderer=None,
            )
            return CorrelatedObservation.from_observation(
                obs_, patch_size=patch_size, maxlength=maxlength, native_scale=native_scale
            )

        # measure the noise power spectrum directly:
        # the resampling is what creates the pixel correlations, so noise instances drawn from the
        # original (uncorrelated) weights and pushed through `trafo` have the correct correlation
        # structure. This avoids the detour via a correlation function truncated at `maxlength`,
        # which is not positive semi-definite and therefore yields negative power spectrum modes.
        # Averaging is essential: a single periodogram has 100% scatter per mode.
        #
        # CAVEAT: the noise here is resampled (`trafo(noise)`), but during the fit the model is
        # rendered by convolving on the resampled grid with the resampled PSF (`match()` builds a
        # ConvolutionTransformation, not a resampler). Convolution and resampling commute only for
        # band-limited signals. Above the original (coarser) Nyquist frequency the resampled data
        # carry no independent information (`n_eff` counts the modes below it), and there the
        # resampled power spectrum, the render, and `trafo(model_coarse * psf)` all disagree at the
        # Lanczos side-lobe level. `_chisquare` therefore drops those modes from the sum entirely
        # via `_mode_mask`, which also removes this ambiguity.

        def _periodogram(key):
            noise_field = jax.random.normal(key, shape=obs.data.shape) * sigma
            return jnp.abs(jnp.fft.rfft2(trafo(noise_field), axes=(-2, -1))) ** 2

        # the resampling dominates this loop, and it vectorizes well, so realizations are processed in
        # batches. lax.map keeps memory at `batch_size` noise fields instead of `n_realizations`
        keys = jax.random.split(key, n_realizations)
        power_spectrum = jax.lax.map(_periodogram, keys, batch_size=batch_size).mean(axis=0)
        power_spectrum /= jnp.prod(jnp.asarray(data.shape[-2:]))

        # `native_scale` (the pre-resampling pixel scale) tells the constructor how many of the modes
        # carry independent noise; nothing else to do about the rank deficiency here.
        return CorrelatedObservation(
            data,
            mask=mask,
            psf=psf,
            wcs=wcs,
            channels=obs.frame.channels,
            name=obs.name,
            renderer=None,
            power_spectrum=power_spectrum,
            native_scale=native_scale,
        )


def chi_square_in_box_and_border(residuals, weights, bbox, border_width):
    """
    helper function for :py:meth:`eval_chi_square_in_box_and_border`

    Parameters
    ----------
    residuals: array
        residual image
    weights: array
        observation weights (inverse variance)
    bbox: :py:class:`~scarlet2.Box
        source box`
    border_width: int
        width of the border around the source box
    """
    bbox_out = bbox.grow([0, border_width, border_width])

    sub_res_in = jax.lax.dynamic_slice(residuals, bbox.start, bbox.shape)
    sub_res_out = jax.lax.dynamic_slice(residuals, bbox_out.start, bbox_out.shape)
    weights_in = jax.lax.dynamic_slice(weights, bbox.start, bbox.shape)
    weights_out = jax.lax.dynamic_slice(weights, bbox_out.start, bbox_out.shape)

    border = jax.lax.dynamic_update_slice(
        jnp.ones_like(sub_res_out), jnp.zeros_like(sub_res_in), (0, 3, 3)
    ).astype("bool")

    chi_square_box = (weights_in * (sub_res_in**2)).mean()
    chi_square_border = (weights_out * (sub_res_out**2))[border].mean()

    return chi_square_box, chi_square_border


class ObservationValidator(metaclass=ValidationMethodCollector):
    """A class containing all of the validation checks for Observation objects.
    Note that the metaclass is defined as `MethodCollector`, which collects all
    validation methods in this class into a single class attribute list called
    `validation_checks`. This allows for easy iteration over all checks."""

    def __init__(self, observation: Observation):
        self.observation = observation

    def check_num_channels_matches_data(self) -> ValidationResult:
        """Check that the number of channels in the observation matches the data.

        NOTE: It is unlikely that this check will ever fail because there are many assertions
        in place around Frame and BBox that will raise an error if the number of channels
        does not match the data shape.

        Returns
        -------
        ValidationResult
            A subclass of ValidationResult indicating the result of the check.
        """
        num_channels = len(self.observation.frame.channels)
        if num_channels != self.observation.data.shape[0]:
            return ValidationError(
                message="Number of channels in the observation does not match the data.",
                check=self.__class__.__name__,
                context={
                    "observation.frame.channels": self.observation.frame.channels,
                    "observation.data.shape": self.observation.data.shape,
                },
            )
        else:
            return ValidationInfo(
                message="Number of channels in the observation matches the data.",
                check=self.__class__.__name__,
            )

    def check_data_and_weights_shape(self) -> ValidationResult:
        """Check that the data and weights exist and have the same shape.

        Returns
        -------
        ValidationResult
            A subclass of ValidationResult indicating the result of the check.
        """
        if self.observation.data is None:
            return ValidationError(message="Observation.data are not set.", check=self.__class__.__name__)
        if self.observation.weights is None:
            return ValidationError(message="Observation.weights are not set.", check=self.__class__.__name__)
        if self.observation.data.shape != self.observation.weights.shape:
            return ValidationError(
                message="Observation data and weights must have the same shape.",
                check=self.__class__.__name__,
                context={
                    "observation.data.shape": self.observation.data.shape,
                    "observation.weights.shape": self.observation.weights.shape,
                },
            )
        else:
            return ValidationInfo(
                message="Data and weights have the same shape.",
                check=self.__class__.__name__,
            )

    def check_weights_finite_and_non_negative(self) -> ValidationResult:
        """Check that the weights in the observation are finite and non-negative.

        Returns
        -------
        ValidationResult
            A subclass of ValidationResult indicating the result of the check.
        """
        if self.observation.weights is not None:
            if not jnp.isfinite(self.observation.weights).all():
                return ValidationError(
                    message="Observation.weights must be finite.",
                    check=self.__class__.__name__,
                )
            elif (self.observation.weights < 0).any():
                return ValidationError(
                    message="Observation.weights must be non-negative.",
                    check=self.__class__.__name__,
                )
            else:
                return ValidationInfo(
                    message="Observation.weights in the observation are finite and non-negative.",
                    check=self.__class__.__name__,
                )

    def check_data_finite_for_non_zero_weights(self) -> ValidationResult:
        """Check that the data in the observation is finite where weights are greater
        than zero.

        Returns
        -------
        ValidationResult
            A subclass of ValidationResult indicating the result of the check.
        """
        if self.observation.weights is not None and self.observation.data is not None:
            # Mask self.observation.data where self.observation.weights is 0
            if not jnp.isfinite(self.observation.data[self.observation.weights > 0]).all():
                return ValidationError(
                    message="Data in the observation must be finite where weights are greater than zero.",
                    check=self.__class__.__name__,
                )
            else:
                return ValidationInfo(
                    message="Data in the observation are finite where weights are greater than zero.",
                    check=self.__class__.__name__,
                )

    def check_psf_has_3_dimensions(self) -> ValidationResult:
        """Check that the PSF in the observation is 3-dimensional.

        Returns
        -------
        ValidationResult
            A subclass of ValidationResult indicating the result of the check.
        """
        if self.observation.frame.psf is not None:
            psf = self.observation.frame.psf()
            if psf.ndim != 3:
                return ValidationError(
                    message="PSF must be 3-dimensional.",
                    check=self.__class__.__name__,
                    context={"observation.frame.psf.shape": psf.shape},
                )
            else:
                return ValidationInfo(
                    message="PSF is 3-dimensional.",
                    check=self.__class__.__name__,
                )
        else:
            return ValidationWarning(
                message="Observation.psf is not defined.",
                check=self.__class__.__name__,
            )

    def check_number_of_psf_channels(self) -> ValidationResult:
        """Check that the number of PSF channels matches the number of data channels and
        that the PSF and data have the same number of dimensions. The PSF should be
        3-dimensional, and number of channels should match the data.

        Returns
        -------
        ValidationResult
            A subclass of ValidationResult indicating the result of the check.
        """
        if self.observation.frame.psf is not None:
            num_psf_channels = self.observation.frame.psf().shape[0]
            num_data_channels = self.observation.data.shape[0]

            # The number of bands is different between the PSF and data
            if num_psf_channels == num_data_channels:
                return ValidationInfo(
                    message="Number of PSF channels matches the number of data channels.",
                    check=self.__class__.__name__,
                )
            else:
                return ValidationError(
                    message="Number of PSF channels does not match the number of data channels.",
                    check=self.__class__.__name__,
                    context={
                        "observation.frame.psf.shape": self.observation.frame.psf().shape,
                        "observation.data.shape": self.observation.data.shape,
                    },
                )

    def check_psf_centroid_consistent(self) -> ValidationResult:
        """Check that the pixel location of the PSF centroid is consistent across
        channels.

        Returns
        -------
        ValidationResult
            A subclass of ValidationResult indicating the result of the check.
        """
        if self.observation.frame.psf is not None:
            from .measure import Moments

            psf_shape = self.observation.frame.psf().shape
            psf_center_y = psf_shape[-2] // 2
            psf_center_x = psf_shape[-1] // 2
            moments = Moments(self.observation.frame.psf(), N=1, center=[psf_center_y, psf_center_x])
            psf_centroid = moments.centroid

            psf_centroid_y, psf_centroid_x = psf_centroid

            tolerance = 1e-3
            if jnp.allclose(psf_centroid_y, psf_centroid_y[0], atol=tolerance) and jnp.allclose(
                psf_centroid_x, psf_centroid_x[0], atol=tolerance
            ):
                return ValidationInfo(
                    message="PSF centroid is consistent across channels.",
                    check=self.__class__.__name__,
                )
            else:
                return ValidationWarning(
                    message="PSF centroid is not the same in all channels.",
                    check=self.__class__.__name__,
                    context={
                        "psf_centroid": psf_centroid,
                    },
                )
