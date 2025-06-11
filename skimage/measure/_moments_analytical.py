"""Analytical transformations from raw image moments to central moments.

The expressions for the 2D central moments of order <=2 are often given in
textbooks. Expressions for higher orders and dimensions were generated in SymPy
using ``tools/precompute/moments_sympy.py`` in the GitHub repository.

"""

import itertools
import math

import numpy as np


def _moments_raw_to_central_fast(moments_raw):
    """Analytical formulae for 2D and 3D central moments of order < 4.

    `moments_raw_to_central` will automatically call this function when
    ndim < 4 and order < 4.

    Parameters
    ----------
    moments_raw : ndarray
        The raw moments.

    Returns
    -------
    moments_central : ndarray
        The central moments.
    """
    ndim = moments_raw.ndim
    order = moments_raw.shape[0] - 1
    float_dtype = moments_raw.dtype
    # Already float64? If not, convert, but only for computation.
    m = (
        moments_raw
        if moments_raw.dtype == np.float64
        else moments_raw.astype(np.float64, copy=False)
    )
    moments_central = np.zeros_like(m, dtype=np.float64)
    if order >= 4 or ndim not in [2, 3]:
        raise ValueError("This function only supports 2D or 3D moments of order < 4.")

    if ndim == 2:
        cx = m[1, 0] / m[0, 0]
        cy = m[0, 1] / m[0, 0]
        moments_central[0, 0] = m[0, 0]
        if order > 1:
            # Vectorized assignment of 2nd order moments
            moments_central[1, 1] = m[1, 1] - cx * m[0, 1]
            moments_central[2, 0] = m[2, 0] - cx * m[1, 0]
            moments_central[0, 2] = m[0, 2] - cy * m[0, 1]
        if order > 2:
            # Avoid repeated computation by precomputing terms
            cx2, cy2 = cx * cx, cy * cy
            cxcy = cx * cy
            # 3rd order moments
            moments_central[2, 1] = (
                m[2, 1]
                - 2 * cx * m[1, 1]
                - cy * m[2, 0]
                + cx2 * m[0, 1]
                + cxcy * m[1, 0]
            )
            moments_central[1, 2] = (
                m[1, 2] - 2 * cy * m[1, 1] - cx * m[0, 2] + 2 * cxcy * m[0, 1]
            )
            moments_central[3, 0] = m[3, 0] - 3 * cx * m[2, 0] + 2 * cx2 * m[1, 0]
            moments_central[0, 3] = m[0, 3] - 3 * cy * m[0, 2] + 2 * cy2 * m[0, 1]
    else:
        # 3D case
        cx = m[1, 0, 0] / m[0, 0, 0]
        cy = m[0, 1, 0] / m[0, 0, 0]
        cz = m[0, 0, 1] / m[0, 0, 0]
        moments_central[0, 0, 0] = m[0, 0, 0]
        if order > 1:
            # 2nd order moments
            moments_central[0, 0, 2] = -cz * m[0, 0, 1] + m[0, 0, 2]
            moments_central[0, 1, 1] = -cy * m[0, 0, 1] + m[0, 1, 1]
            moments_central[0, 2, 0] = -cy * m[0, 1, 0] + m[0, 2, 0]
            moments_central[1, 0, 1] = -cx * m[0, 0, 1] + m[1, 0, 1]
            moments_central[1, 1, 0] = -cx * m[0, 1, 0] + m[1, 1, 0]
            moments_central[2, 0, 0] = -cx * m[1, 0, 0] + m[2, 0, 0]
        if order > 2:
            # 3rd order, precompute powers and products
            cx2, cy2, cz2 = cx * cx, cy * cy, cz * cz
            cxcy, cxcz, cycz = cx * cy, cx * cz, cy * cz
            # 3rd order moments
            moments_central[0, 0, 3] = (
                2 * cz2 * m[0, 0, 1] - 3 * cz * m[0, 0, 2] + m[0, 0, 3]
            )
            moments_central[0, 1, 2] = (
                -cy * m[0, 0, 2] + 2 * cz * (cy * m[0, 0, 1] - m[0, 1, 1]) + m[0, 1, 2]
            )
            moments_central[0, 2, 1] = (
                cy2 * m[0, 0, 1]
                - 2 * cy * m[0, 1, 1]
                + cz * (cy * m[0, 1, 0] - m[0, 2, 0])
                + m[0, 2, 1]
            )
            moments_central[0, 3, 0] = (
                2 * cy2 * m[0, 1, 0] - 3 * cy * m[0, 2, 0] + m[0, 3, 0]
            )
            moments_central[1, 0, 2] = (
                -cx * m[0, 0, 2] + 2 * cz * (cx * m[0, 0, 1] - m[1, 0, 1]) + m[1, 0, 2]
            )
            moments_central[1, 1, 1] = (
                -cx * m[0, 1, 1]
                + cy * (cx * m[0, 0, 1] - m[1, 0, 1])
                + cz * (cx * m[0, 1, 0] - m[1, 1, 0])
                + m[1, 1, 1]
            )
            moments_central[1, 2, 0] = (
                -cx * m[0, 2, 0] - 2 * cy * (-cx * m[0, 1, 0] + m[1, 1, 0]) + m[1, 2, 0]
            )
            moments_central[2, 0, 1] = (
                cx2 * m[0, 0, 1]
                - 2 * cx * m[1, 0, 1]
                + cz * (cx * m[1, 0, 0] - m[2, 0, 0])
                + m[2, 0, 1]
            )
            moments_central[2, 1, 0] = (
                cx2 * m[0, 1, 0]
                - 2 * cx * m[1, 1, 0]
                + cy * (cx * m[1, 0, 0] - m[2, 0, 0])
                + m[2, 1, 0]
            )
            moments_central[3, 0, 0] = (
                2 * cx2 * m[1, 0, 0] - 3 * cx * m[2, 0, 0] + m[3, 0, 0]
            )

    return (
        moments_central.astype(float_dtype, copy=False)
        if moments_central.dtype != float_dtype
        else moments_central
    )


def moments_raw_to_central(moments_raw):
    ndim = moments_raw.ndim
    order = moments_raw.shape[0] - 1
    if ndim in [2, 3] and order < 4:
        return _moments_raw_to_central_fast(moments_raw)

    m = moments_raw
    centers = tuple(m[tuple(np.eye(ndim, dtype=int))] / m[(0,) * ndim])

    if ndim == 2:
        # Explicit fast 2D version
        return _moments2d_to_central_general(m, order, centers)
    else:
        return _moments_nd_to_central_general(m, order, centers)


def _binom_pows_cache(order, c):
    # Returns a 2D array b[p, i] = comb(p, i) * (-c) ** (p-i) for all p in [0,order], i in [0,p]
    binom = np.zeros((order + 1, order + 1), dtype=np.float64)
    for p in range(order + 1):
        binom[p, : p + 1] = [math.comb(p, i) * ((-c) ** (p - i)) for i in range(p + 1)]
    return binom


def _moments2d_to_central_general(m, order, centers):
    # Cache binomial multipliers for both axes
    b0 = _binom_pows_cache(order, centers[0])
    b1 = _binom_pows_cache(order, centers[1])
    moments_central = np.zeros_like(m)
    for p in range(order + 1):
        for q in range(order + 1 - p):
            v = 0.0
            # Vectorize over all (i, j)
            for i in range(p + 1):
                # Use precomputed binomial for (p, i)
                b0_pi = b0[p, i]
                arr = m[i, : q + 1] * b0_pi * b1[q, : q + 1]
                v += arr.sum()
            moments_central[p, q] = v
    return moments_central


def _moments_nd_to_central_general(m, order, centers):
    """
    Optimized (still explicit loops, but with some caching) general nD formula for central moments.
    """
    ndim = m.ndim
    moments_central = np.zeros_like(m)

    # Precompute binomial and power tables for all axes
    binoms = [_binom_pows_cache(order, c) for c in centers]

    # Indices to assign to
    def valid_indices(order, ndim):
        """Yield all multi-indices where sum(idx) <= order."""
        if ndim == 1:
            for i in range(order + 1):
                yield (i,)
        else:
            for i in range(order + 1):
                for t in valid_indices(order - i, ndim - 1):
                    yield (i,) + t

    idx_shape = m.shape
    for out_idx in valid_indices(order, ndim):
        # Prepare for inner sum, for all idxs <= out_idx on each axis
        ranges = [range(o + 1) for o in out_idx]
        sub_binoms = [binoms[d][out_idx[d], : out_idx[d] + 1] for d in range(ndim)]
        s = 0.0
        for idx in itertools.product(*ranges):
            prod = m[idx]
            for d in range(ndim):
                prod *= sub_binoms[d][idx[d]]
            s += prod
        moments_central[out_idx] = s
    return moments_central
