import numpy as np
import pytest

from skimage import data, filters, measure

try:
    from skimage.measure._regionprops import PROP_VALS
except ImportError:
    PROP_VALS = []


def init_regionprops_data():
    image = filters.gaussian(data.coins().astype(float), sigma=3)
    # increase size to (2048, 2048) by tiling
    image = np.tile(image, (4, 4))
    label_image = measure.label(image > 130, connectivity=image.ndim)
    intensity_image = image
    return label_image, intensity_image


class RegionpropsTableIndividual:
    def setup_method(self):
        self.label_image, self.intensity_image = init_regionprops_data()

    @pytest.mark.parametrize('prop', sorted(list(PROP_VALS)))
    def test_single_region_property(self, prop):
        measure.regionprops_table(
            self.label_image, self.intensity_image, properties=[prop], cache=True
        )

    # omit peakmem tests to save time (memory usage was minimal)


class RegionpropsTableAll:
    def setup_method(self):
        self.label_image, self.intensity_image = init_regionprops_data()

    @pytest.mark.parametrize('cache', (False, True))
    def test_regionprops_table_all(self, cache):
        measure.regionprops_table(
            self.label_image, self.intensity_image, properties=PROP_VALS, cache=cache
        )

    # omit peakmem tests to save time (memory usage was minimal)


class MomentsSuite:
    """Benchmark for filter routines in scikit-image."""

    def _setup_moments(self, shape, dtype):
        rng = np.random.default_rng(1234)
        if np.dtype(dtype).kind in 'iu':
            self.image = rng.integers(0, 256, shape, dtype=dtype)
        else:
            self.image = rng.standard_normal(shape, dtype=dtype)

    @pytest.mark.parametrize('shape,dtype,order', [
        (shape, dtype, order)
        for shape in [(64, 64), (4096, 2048), (32, 32, 32), (256, 256, 192)]
        for dtype in [np.uint8, np.float32, np.float64]
        for order in [1, 2, 3]
    ])
    def test_moments_raw(self, shape, dtype, order):
        self._setup_moments(shape, dtype)
        measure.moments(self.image)

    @pytest.mark.parametrize('shape,dtype,order', [
        (shape, dtype, order)
        for shape in [(64, 64), (4096, 2048), (32, 32, 32), (256, 256, 192)]
        for dtype in [np.uint8, np.float32, np.float64]
        for order in [1, 2, 3]
    ])
    def test_moments_central(self, shape, dtype, order):
        self._setup_moments(shape, dtype)
        measure.moments_central(self.image)

    @pytest.mark.parametrize('shape,dtype,order', [
        (shape, dtype, order)
        for shape in [(64, 64), (4096, 2048), (32, 32, 32), (256, 256, 192)]
        for dtype in [np.uint8, np.float32, np.float64]
        for order in [1, 2, 3]
    ])
    def test_peakmem_reference(self, shape, dtype, order):
        pass

    @pytest.mark.parametrize('shape,dtype,order', [
        (shape, dtype, order)
        for shape in [(64, 64), (4096, 2048), (32, 32, 32), (256, 256, 192)]
        for dtype in [np.uint8, np.float32, np.float64]
        for order in [1, 2, 3]
    ])
    def test_peakmem_moments_central(self, shape, dtype, order):
        self._setup_moments(shape, dtype)
        measure.moments_central(self.image)
