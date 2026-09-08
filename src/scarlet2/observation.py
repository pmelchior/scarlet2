import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from .bbox import Box, insert_into, overlap_slices
from .fft import transform
from .frame import Frame, get_affine
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

        # (re)-import `VALIDATION_SWITCH` at runtime to avoid using a static/old value
        from .validation_utils import VALIDATION_SWITCH

        if VALIDATION_SWITCH:
            from .validation import check_observation

            validation_results = check_observation(self)
            print_validation_results("Observation validation results", validation_results)

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


# TODO: sampled power spectrum computations are very slow, and might be done twice (when resampling and when padding)
# Option is to compute it only when the padded PS is needed as a lazy init when calling match()
# Requires that the original observation is stored in the CorrelatedObservation.
# TODO: Performance testing and checking for edge sources
# TODO: More through testing on the correlation_function -> PS path, or a mechanism to avoid it entirely
# TODO: Move all PS/correlation_function handling to fft module
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
    _data_fft: jnp.ndarray = eqx.field(repr=False)
    """:py:attr:`data`, transformed onto the padded grid. `None` whenever `_power_spectrum_padded` is."""
    _power_spectrum_padded: jnp.ndarray = eqx.field(repr=False)
    """Power spectrum on the padded grid of a trailing :py:class:`~scarlet2.ConvolutionTransformation`

    `None` unless :py:meth:`_match_power_spectrum` found that the faster Fourier-space path applies.
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
    ):
        data = jnp.asarray(data, dtype=float)
        if data.ndim == 2:
            data = data[None, ...]

        assert (power_spectrum is None) != (correlation_function is None), (
            "Provide either power_spectrum or correlation_function"
        )
        # weights ignore pixel covariance: per-pixel variance only, i.e. the zero-lag correlation xi(0,0).
        # Prefer the measured value when the correlation function is given: it is the direct estimate,
        # whereas inverting the power spectrum picks up the distortion that abs() introduces there
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
        self._power_spectrum_padded = None
        self._data_fft = None
        super().__init__(data, weights, psf=psf, wcs=wcs, channels=channels, renderer=renderer, name=name)
        if self.renderer is not None:
            self._match_power_spectrum()

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

    def _chisquare(self, model):
        if self._power_spectrum_padded is not None:
            # take the model straight from the convolution in Fourier space and zero-pad the data onto the
            # same grid, so that the residual there is the zero-padded data-grid residual that
            # `_padded_power_spectrum` is matched to
            model_fft = self.render(model, return_fft=True)
            n_pad = jnp.prod(jnp.asarray(self.renderer[-1]._fft_shape))
            res_fft = (model_fft - self._data_fft) / jnp.sqrt(n_pad / 2)
            return jnp.sum((res_fft * jnp.conjugate(res_fft)).real / self._power_spectrum_padded)

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
        return jnp.sum((res_fft * jnp.conjugate(res_fft)).real / self.power_spectrum)

    @classmethod
    def from_observation(
        cls,
        obs,
        patch_size=50,
        maxlength=12,
        resample_to_frame=None,
        lanczos_order=9,
        resample_psf=True,
        n_realizations=64,
        batch_size=8,
    ):
        """Create a :py:class:`CorrelatedObservation` from :py:class:`Observation`

        The method will construct a new Observation instance with a modified likelihood that takes into
        account the pixel correlation. How the noise power spectrum is obtained depends on
        `resample_to_frame`:

        * If it is set, the resampling itself creates the correlations, so the power spectrum is measured
          directly by averaging periodograms of `n_realizations` resampled noise fields.
        * If it is `None`, there is no generative noise model. The method then finds a patch of size
          `patch_size` with as few sources as possible, measures the pixel correlations in that patch, and
          converts them to a power spectrum. Note that truncating the correlation function at `maxlength`
          biases the resulting power spectrum, severely so if the correlation length approaches `maxlength`.

        Parameters
        ----------
        obs: :py:class:`Observation`
            Observation containing the data and original weight map
        patch_size: int
            Linear size of the patch for measuring the correlation function.
            The argument has no effect if `resample_to_frame` is set.
        maxlength: int
            Maximum distance (in pixels) for the 2D correlation function. It needs to be large enough to
            cover the extent of the correlations, and small compared to `patch_size` so that every lag is
            averaged over many pixel pairs.
            The argument has no effect if `resample_to_frame` is set.
        resample_to_frame: None, :py:class:`~scarlet2.Frame`
            Frame describing the desired spatial sampling. Is assumed to be a model frame.
        lanczos_order: int
            Lanczos order used by the resampling operation
            The argument has no effect if `resample_to_frame` is `None`.
        resample_psf: bool, optional
            Whether to resample `obs.psf` to `resample_to_frame`.
            Should be set to False only if PSF is already sampled with the resolution of `resample_to_frame`.
            The argument has no effect if `resample_to_frame` is `None`.
        n_realizations: int
            Number of noise realizations averaged into the power spectrum estimate. The relative scatter
            per mode is `1 / sqrt(n_realizations)`, so values below ~16 bias the likelihood badly.
            The argument has no effect if `resample_to_frame` is `None`.
        batch_size: int
            Number of noise realizations resampled at once. Larger values are faster but hold that many
            noise fields in memory at a time.
            The argument has no effect if `resample_to_frame` is `None`.

        Returns
        -------
        :py:class:`CorrelatedObservation`
        """
        if resample_to_frame is not None:
            # create a reverse renderer without PSF corrections or channel filtering
            _obs_frame = Frame(obs.frame.bbox, psf=None, wcs=obs.frame.wcs, channels=obs.frame.channels)
            _new_box = obs.frame.bbox[:-2] @ resample_to_frame.bbox.spatial
            _model_frame = Frame(_new_box, psf=None, wcs=resample_to_frame.wcs, channels=obs.frame.channels)
            trafo = LanczosResamplingTranformation(_obs_frame, _model_frame, lanczos_order=lanczos_order)
            wcs = resample_to_frame.wcs

            # resample data
            data = trafo(obs.data)

            # resample PSF: first insert PSF into middle of image with same size of obs
            psf_image = obs.frame.psf()
            if psf_image.ndim == 2:
                # a single-band PSF (e.g. GaussianPSF) needs the channel dimension to line up with the data
                psf_image = jnp.tile(psf_image, (obs.data.shape[0], 1, 1))
            if resample_psf:
                full_psf_image = jnp.zeros(obs.data.shape)
                full_box = Box(full_psf_image.shape)
                shift = tuple(
                    full_psf_image.shape[d] // 2 - psf_image.shape[d] // 2 for d in range(full_box.D)
                )
                psf_box = Box(psf_image.shape) + shift
                full_psf_image = insert_into(full_psf_image, psf_image, psf_box)
                psf = trafo(full_psf_image)
            else:
                psf = psf_image

            # resample mask plane (weights themselves are not needed)
            mask = jnp.asarray(obs.weights == 0, dtype=jnp.float32)
            mask = trafo(mask) > 0.3  # edge of mask gets blurry, include fractional masking

            # measure the noise power spectrum directly:
            # the resampling is what creates the pixel correlations, so noise instances drawn from the
            # original (uncorrelated) weights and pushed through `trafo` have the correct correlation
            # structure. This avoids the detour via a correlation function truncated at `maxlength`,
            # which is not positive semi-definite and therefore yields negative power spectrum modes.
            # Averaging is essential: a single periodogram has 100% scatter per mode.
            key = jax.random.key(hash(obs.frame))
            # masked pixels are given the median noise level rather than 0, so that the noise field stays
            # stationary and the periodogram does not pick up leakage from the mask boundaries
            sigma = jnp.where(obs.weights > 0, 1 / jnp.sqrt(jnp.where(obs.weights > 0, obs.weights, 1)), 0)
            sigma = jnp.where(sigma > 0, sigma, jnp.median(sigma))

            def _periodogram(key):
                noise_field = jax.random.normal(key, shape=obs.data.shape) * sigma
                return jnp.abs(jnp.fft.rfft2(trafo(noise_field), axes=(-2, -1))) ** 2

            # the resampling dominates this loop, and it vectorizes well, so realizations are processed in
            # batches. lax.map keeps memory at `batch_size` noise fields instead of `n_realizations`
            keys = jax.random.split(key, n_realizations)
            power_spectrum = jax.lax.map(_periodogram, keys, batch_size=batch_size).mean(axis=0)
            power_spectrum /= jnp.prod(jnp.asarray(data.shape[-2:]))
            xi = None

            # we need a new renderer for this resampled observation
            renderer = None

        else:
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
                        jnp.argmax(
                            gaps[patch_size // 2 : -patch_size // 2, patch_size // 2 : -patch_size // 2]
                        ),
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
            power_spectrum = None

            # define the remaining items
            psf = obs.frame.psf
            wcs = obs.frame.wcs
            renderer = obs.renderer
            mask = obs.weights == 0

        return CorrelatedObservation(
            data,
            mask=mask,
            psf=psf,
            wcs=wcs,
            renderer=renderer,
            power_spectrum=power_spectrum,
            correlation_function=xi,
            channels=obs.frame.channels,
            name=obs.name,
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
