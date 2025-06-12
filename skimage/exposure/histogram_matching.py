import numpy as np

from .._shared import utils


def _match_cumulative_cdf(source, template):
    """
    Return modified source array so that the cumulative density function of
    its values matches the cumulative density function of the template.
    """

    source_flat = source.ravel()  # ravel faster than reshape(-1)
    template_flat = template.ravel()

    if source.dtype.kind == 'u':
        src_lookup = source_flat
        src_counts = np.bincount(src_lookup)
        tmpl_counts = np.bincount(template_flat)
        tmpl_values = np.nonzero(tmpl_counts)[0]
        tmpl_counts = tmpl_counts[tmpl_values]

        # Quantiles
        src_quantiles = np.cumsum(src_counts) / source_flat.size
        tmpl_quantiles = np.cumsum(tmpl_counts) / template_flat.size

        interp_a_values = np.interp(src_quantiles, tmpl_quantiles, tmpl_values)
        result = interp_a_values[src_lookup].reshape(source.shape)
        return result
    else:
        # Get unique, sorted values and their counts
        src_values, src_counts = np.unique(source_flat, return_counts=True)
        tmpl_values, tmpl_counts = np.unique(template_flat, return_counts=True)

        # Quantiles
        src_quantiles = np.cumsum(src_counts) / source_flat.size
        tmpl_quantiles = np.cumsum(tmpl_counts) / template_flat.size

        interp_a_values = np.interp(src_quantiles, tmpl_quantiles, tmpl_values)
        # Use searchsorted for fast lookup, much faster than np.unique(..., return_inverse=True)
        src_lookup = np.searchsorted(src_values, source_flat)
        result = interp_a_values[src_lookup].reshape(source.shape)
        return result


@utils.channel_as_last_axis(channel_arg_positions=(0, 1))
def match_histograms(image, reference, *, channel_axis=None):
    """Adjust an image so that its cumulative histogram matches that of another.

    The adjustment is applied separately for each channel.

    Parameters
    ----------
    image : ndarray
        Input image. Can be gray-scale or in color.
    reference : ndarray
        Image to match histogram of. Must have the same number of channels as
        image.
    channel_axis : int or None, optional
        If None, the image is assumed to be a grayscale (single channel) image.
        Otherwise, this parameter indicates which axis of the array corresponds
        to channels.

    Returns
    -------
    matched : ndarray
        Transformed input image.

    Raises
    ------
    ValueError
        Thrown when the number of channels in the input image and the reference
        differ.

    References
    ----------
    .. [1] http://paulbourke.net/miscellaneous/equalisation/

    """
    if image.ndim != reference.ndim:
        raise ValueError('Image and reference must have the same number of channels.')

    if channel_axis is not None:
        if image.shape[-1] != reference.shape[-1]:
            raise ValueError(
                'Number of channels in the input image and reference image must match!'
            )

        matched = np.empty(image.shape, dtype=image.dtype)
        for channel in range(image.shape[-1]):
            matched_channel = _match_cumulative_cdf(
                image[..., channel], reference[..., channel]
            )
            matched[..., channel] = matched_channel
    else:
        # _match_cumulative_cdf will always return float64 due to np.interp
        matched = _match_cumulative_cdf(image, reference)

    if matched.dtype.kind == 'f':
        # output a float32 result when the input is float16 or float32
        out_dtype = utils._supported_float_type(image.dtype)
        matched = matched.astype(out_dtype, copy=False)
    return matched
