# See "Writing benchmarks" in the asv docs for more information.
# https://asv.readthedocs.io/en/latest/writing_benchmarks.html
import numpy as np
import pytest

from skimage import data, filters, color
from skimage.filters.thresholding import threshold_li


class FiltersSuite:
    """Benchmark for filter routines in scikit-image."""

    def setup_method(self):
        self.image = np.random.random((4000, 4000))
        self.image[:2000, :2000] += 1
        self.image[3000:, 3000] += 0.5

    def test_sobel(self):
        filters.sobel(self.image)


class FiltersSobel3D:
    """Benchmark for 3d sobel filters."""

    def setup_method(self):
        try:
            filters.sobel(np.ones((8, 8, 8)))
        except ValueError:
            raise NotImplementedError("3d sobel unavailable")
        self.image3d = data.binary_blobs(length=256, n_dim=3).astype(float)

    def test_sobel_3d(self):
        _ = filters.sobel(self.image3d)


class MultiOtsu:
    """Benchmarks for MultiOtsu threshold."""

    def setup_method(self):
        self.image = data.camera()

    @pytest.mark.parametrize('classes', [3, 4, 5])
    def test_threshold_multiotsu(self, classes):
        filters.threshold_multiotsu(self.image, classes=classes)

    @pytest.mark.parametrize('classes', [3, 4, 5])
    def test_peakmem_reference(self, classes):
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

    @pytest.mark.parametrize('classes', [3, 4, 5])
    def test_peakmem_threshold_multiotsu(self, classes):
        filters.threshold_multiotsu(self.image, classes=classes)


class ThresholdSauvolaSuite:
    """Benchmark for transform routines in scikit-image."""

    def setup_method(self):
        self.image = np.zeros((2000, 2000), dtype=np.uint8)
        self.image3D = np.zeros((30, 300, 300), dtype=np.uint8)

        idx = np.arange(500, 700)
        idx3D = np.arange(10, 200)

        self.image[idx[::-1], idx] = 255
        self.image[idx, idx] = 255

        self.image3D[:, idx3D[::-1], idx3D] = 255
        self.image3D[:, idx3D, idx3D] = 255

    def test_sauvola(self):
        filters.threshold_sauvola(self.image, window_size=51)

    def test_sauvola_3d(self):
        filters.threshold_sauvola(self.image3D, window_size=51)


class ThresholdLi:
    """Benchmark for threshold_li in scikit-image."""

    def setup_method(self):
        try:
            self.image = data.eagle()
        except ValueError:
            raise NotImplementedError("eagle data unavailable")
        self.image_float32 = self.image.astype(np.float32)

    def test_integer_image(self):
        threshold_li(self.image)

    def test_float32_image(self):
        threshold_li(self.image_float32)


class RidgeFilters:
    """Benchmark ridge filters in scikit-image."""

    def setup_method(self):
        # Ensure memory footprint of lazy import is included in reference
        self._ = filters.meijering, filters.sato, filters.frangi, filters.hessian
        self.image = color.rgb2gray(data.retina())

    def test_peakmem_setup(self):
        """peakmem includes the memory used by setup.
        Peakmem benchmarks measure the maximum amount of RAM used by a
        function. However, this maximum also includes the memory used
        by ``setup`` (as of asv 0.2.1; see [1]_)
        Measuring an empty peakmem function might allow us to disambiguate
        between the memory used by setup and the memory used by slic (see
        ``peakmem_slic_basic``, below).
        References
        ----------
        .. [1]: https://asv.readthedocs.io/en/stable/writing_benchmarks.html#peak-memory
        """
        pass

    def test_meijering(self):
        filters.meijering(self.image)

    def test_peakmem_meijering(self):
        filters.meijering(self.image)

    def test_sato(self):
        filters.sato(self.image)

    def test_peakmem_sato(self):
        filters.sato(self.image)

    def test_frangi(self):
        filters.frangi(self.image)

    def test_peakmem_frangi(self):
        filters.frangi(self.image)

    def test_hessian(self):
        filters.hessian(self.image)

    def test_peakmem_hessian(self):
        filters.hessian(self.image)
