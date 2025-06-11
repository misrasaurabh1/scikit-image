# See "Writing benchmarks" in the asv docs for more information.
# https://asv.readthedocs.io/en/latest/writing_benchmarks.html
import numpy as np
import pytest
from skimage import transform


class InterpolationResize:
    """Benchmark for filter routines in scikit-image."""

    def _setup_interpolation(self, new_shape, dtype):
        ndim = len(new_shape)
        if ndim == 2:
            image = np.random.random((1000, 1000))
        else:
            image = np.random.random((100, 100, 100))
        self.image = image.astype(dtype, copy=False)

    @pytest.mark.parametrize('new_shape,order,mode,dtype,anti_aliasing', [
        (new_shape, order, mode, dtype, anti_aliasing)
        for new_shape in [(500, 800), (2000, 4000), (80, 80, 80), (150, 150, 150)]
        for order in [0, 1, 3, 5]
        for mode in ['symmetric']
        for dtype in [np.float64]
        for anti_aliasing in [True]
    ])
    def test_resize(self, new_shape, order, mode, dtype, anti_aliasing):
        self._setup_interpolation(new_shape, dtype)
        transform.resize(
            self.image, new_shape, order=order, mode=mode, anti_aliasing=anti_aliasing
        )

    @pytest.mark.parametrize('new_shape,order,mode,dtype,anti_aliasing', [
        (new_shape, order, mode, dtype, anti_aliasing)
        for new_shape in [(500, 800), (2000, 4000), (80, 80, 80), (150, 150, 150)]
        for order in [0, 1, 3, 5]
        for mode in ['symmetric']
        for dtype in [np.float64]
        for anti_aliasing in [True]
    ])
    def test_rescale(self, new_shape, order, mode, dtype, anti_aliasing):
        self._setup_interpolation(new_shape, dtype)
        scale = tuple(s2 / s1 for s2, s1 in zip(new_shape, self.image.shape))
        transform.rescale(
            self.image, scale, order=order, mode=mode, anti_aliasing=anti_aliasing
        )

    @pytest.mark.parametrize('new_shape,order,mode,dtype,anti_aliasing', [
        (new_shape, order, mode, dtype, anti_aliasing)
        for new_shape in [(500, 800), (2000, 4000), (80, 80, 80), (150, 150, 150)]
        for order in [0, 1, 3, 5]
        for mode in ['symmetric']
        for dtype in [np.float64]
        for anti_aliasing in [True]
    ])
    def test_peakmem_resize(self, new_shape, order, mode, dtype, anti_aliasing):
        self._setup_interpolation(new_shape, dtype)
        transform.resize(
            self.image, new_shape, order=order, mode=mode, anti_aliasing=anti_aliasing
        )

    @pytest.mark.parametrize('new_shape,order,mode,dtype,anti_aliasing', [
        (new_shape, order, mode, dtype, anti_aliasing)
        for new_shape in [(500, 800), (2000, 4000), (80, 80, 80), (150, 150, 150)]
        for order in [0, 1, 3, 5]
        for mode in ['symmetric']
        for dtype in [np.float64]
        for anti_aliasing in [True]
    ])
    def test_peakmem_reference(self, new_shape, order, mode, dtype, anti_aliasing):
        """Provide reference for memory measurement with empty benchmark.

        Peakmem benchmarks measure the maximum amount of RAM used by a
        function. However, this maximum also includes the memory used
        during the setup routine (as of asv 0.2.1; see [1]_).
        Measuring an empty peakmem function might allow us to disambiguate
        between the memory used by setup and the memory used by target (see
        other ``peakmem_`` functions below).

        References
        ----------
        .. [1]: https://asv.readthedocs.io/en/stable/writing_benchmarks.html#peak-memory
        """
        pass
