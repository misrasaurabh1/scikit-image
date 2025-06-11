"""Benchmarks for `skimage.morphology`.

See "Writing benchmarks" in the asv docs for more information.
"""

import numpy as np
from numpy.lib import NumpyVersion as Version
import scipy.ndimage
import pytest

import skimage
from skimage import color, data, morphology, util


class Skeletonize3d:
    def setup_method(self, *args):
        try:
            # use a separate skeletonize_3d function on older scikit-image
            if Version(skimage.__version__) < Version('0.16.0'):
                self.skeletonize = morphology.skeletonize_3d
            else:
                self.skeletonize = morphology.skeletonize
        except AttributeError:
            raise NotImplementedError("3d skeletonize unavailable")

        # we stack the horse data 5 times to get an example volume
        self.image = np.stack(5 * [util.invert(data.horse())])

    def test_skeletonize(self):
        self.skeletonize(self.image)

    def test_peakmem_reference(self, *args):
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

    def test_peakmem_skeletonize(self):
        self.skeletonize(self.image)


class IsotropicMorphology2D:
    # skip rectangle as roughly equivalent to square
    def setup_method(self, shape):
        rng = np.random.default_rng(123)
        # Create an image that is mostly True, with random isolated False areas
        # (so it will not become fully False for any of the footprints).
        self.image = rng.standard_normal(shape) < 3.5

    @pytest.mark.parametrize('shape,radius', [
        (shape, radius)
        for shape in [((512, 512),)]
        for radius in [1, 3, 5, 15, 25, 40]
    ])
    def test_erosion(self, shape, radius):
        self.setup_method(shape[0])
        morphology.isotropic_erosion(self.image, radius)


# Repeat the same footprint tests for grayscale morphology


class GrayMorphology2D:
    def setup_method(self, shape, footprint, radius, decomposition):
        rng = np.random.default_rng(123)
        # Make an image that is mostly True, with random isolated False areas
        # (so it will not become fully False for any of the footprints).
        self.image = rng.standard_normal(shape) < 3.5
        fp_func = getattr(morphology, footprint)
        allow_sequence = ("rectangle", "square", "diamond", "octagon", "disk")
        allow_separable = ("rectangle", "square")
        allow_crosses = ("disk", "ellipse")
        allow_decomp = tuple(
            set(allow_sequence) | set(allow_separable) | set(allow_crosses)
        )
        footprint_kwargs = {}
        if decomposition == "sequence" and footprint not in allow_sequence:
            raise NotImplementedError("decomposition unimplemented")
        elif decomposition == "separable" and footprint not in allow_separable:
            raise NotImplementedError("separable decomposition unavailable")
        elif decomposition == "crosses" and footprint not in allow_crosses:
            raise NotImplementedError("separable decomposition unavailable")
        if footprint in allow_decomp:
            footprint_kwargs["decomposition"] = decomposition
        if footprint in ["rectangle", "square"]:
            size = 2 * radius + 1
            self.footprint = fp_func(size, **footprint_kwargs)
        elif footprint in ["diamond", "disk"]:
            self.footprint = fp_func(radius, **footprint_kwargs)
        elif footprint == "star":
            # set a so bounding box size is approximately 2*radius + 1
            # size will be 2*a + 1 + 2*floor(a / 2)
            a = max((2 * radius) // 3, 1)
            self.footprint = fp_func(a, **footprint_kwargs)
        elif footprint == "octagon":
            # overall size is m + 2 * n
            # so choose m = n so that overall size is ~ 2*radius + 1
            m = n = max((2 * radius) // 3, 1)
            self.footprint = fp_func(m, n, **footprint_kwargs)
        elif footprint == "ellipse":
            if radius > 1:
                # make somewhat elliptical
                self.footprint = fp_func(radius - 1, radius + 1, **footprint_kwargs)
            else:
                self.footprint = fp_func(radius, radius, **footprint_kwargs)

    @pytest.mark.parametrize('shape,footprint,radius,decomposition', [
        (shape, footprint, radius, decomposition)
        for shape in [((512, 512),)]
        for footprint in ["square", "diamond", "octagon", "disk", "ellipse", "star"]
        for radius in [1, 3, 5, 15, 25, 40]
        for decomposition in [None, "sequence", "separable", "crosses"]
    ])
    def test_erosion(self, shape, footprint, radius, decomposition):
        self.setup_method(shape[0], footprint, radius, decomposition)
        morphology.erosion(self.image, self.footprint)


class GrayMorphology3D:
    # skip rectangle as roughly equivalent to square
    def setup_method(self, shape, footprint, radius, decomposition):
        rng = np.random.default_rng(123)
        # make an image that is mostly True, with a few isolated False areas
        self.image = rng.standard_normal(shape) > -3
        fp_func = getattr(morphology, footprint)
        allow_decomp = ("cube", "octahedron", "ball")
        allow_separable = ("cube",)
        if decomposition == "separable" and footprint != "cube":
            raise NotImplementedError("separable unavailable")
        footprint_kwargs = {}
        if decomposition is not None and footprint not in allow_decomp:
            raise NotImplementedError("decomposition unimplemented")
        elif decomposition == "separable" and footprint not in allow_separable:
            raise NotImplementedError("separable decomposition unavailable")
        if footprint in allow_decomp:
            footprint_kwargs["decomposition"] = decomposition
        if footprint == "cube":
            size = 2 * radius + 1
            self.footprint = fp_func(size, **footprint_kwargs)
        elif footprint in ["ball", "octahedron"]:
            self.footprint = fp_func(radius, **footprint_kwargs)

    @pytest.mark.parametrize('shape,footprint,radius,decomposition', [
        (shape, footprint, radius, decomposition)
        for shape in [((128, 128, 128),)]
        for footprint in ["ball", "cube", "octahedron"]
        for radius in [1, 3, 5, 10]
        for decomposition in [None, "sequence", "separable"]
    ])
    def test_erosion(self, shape, footprint, radius, decomposition):
        self.setup_method(shape[0], footprint, radius, decomposition)
        morphology.erosion(self.image, self.footprint)


class GrayReconstruction:
    # skip rectangle as roughly equivalent to square
    def setup_method(self, shape, dtype):
        rng = np.random.default_rng(123)
        # make an image that is mostly True, with a few isolated False areas
        rvals = rng.integers(1, 255, size=shape).astype(dtype=dtype)

        roi1 = tuple(slice(s // 4, s // 2) for s in rvals.shape)
        roi2 = tuple(slice(s // 2 + 1, (3 * s) // 4) for s in rvals.shape)
        seed = np.full(rvals.shape, 1, dtype=dtype)
        seed[roi1] = rvals[roi1]
        seed[roi2] = rvals[roi2]

        # create a mask with a couple of square regions set to seed maximum
        mask = np.full(seed.shape, 1, dtype=dtype)
        mask[roi1] = 255
        mask[roi2] = 255

        self.seed = seed
        self.mask = mask

    @pytest.mark.parametrize('shape,dtype', [
        (shape, dtype)
        for shape in [(10, 10), (64, 64), (1200, 1200), (96, 96, 96)]
        for dtype in [np.uint8, np.float32, np.float64]
    ])
    def test_reconstruction(self, shape, dtype):
        self.setup_method(shape, dtype)
        morphology.reconstruction(self.seed, self.mask)

    @pytest.mark.parametrize('shape,dtype', [
        (shape, dtype)
        for shape in [(10, 10), (64, 64), (1200, 1200), (96, 96, 96)]
        for dtype in [np.uint8, np.float32, np.float64]
    ])
    def test_peakmem_reference(self, shape, dtype):
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

    @pytest.mark.parametrize('shape,dtype', [
        (shape, dtype)
        for shape in [(10, 10), (64, 64), (1200, 1200), (96, 96, 96)]
        for dtype in [np.uint8, np.float32, np.float64]
    ])
    def test_peakmem_reconstruction(self, shape, dtype):
        self.setup_method(shape, dtype)
        morphology.reconstruction(self.seed, self.mask)


class LocalMaxima:
    def setup_method(self):
        # Natural image with small extrema
        self.image = data.moon()

    @pytest.mark.parametrize('connectivity,allow_borders', [
        (connectivity, allow_borders)
        for connectivity in [1, 2]
        for allow_borders in [False, True]
    ])
    def test_2d(self, connectivity, allow_borders):
        morphology.local_maxima(
            self.image, connectivity=connectivity, allow_borders=allow_borders
        )

    @pytest.mark.parametrize('connectivity,allow_borders', [
        (connectivity, allow_borders)
        for connectivity in [1, 2]
        for allow_borders in [False, True]
    ])
    def test_peakmem_reference(self, connectivity, allow_borders):
        """Provide reference for memory measurement with empty benchmark.

        .. [1] https://asv.readthedocs.io/en/stable/writing_benchmarks.html#peak-memory
        """
        pass

    @pytest.mark.parametrize('connectivity,allow_borders', [
        (connectivity, allow_borders)
        for connectivity in [1, 2]
        for allow_borders in [False, True]
    ])
    def test_peakmem_2d(self, connectivity, allow_borders):
        morphology.local_maxima(
            self.image, connectivity=connectivity, allow_borders=allow_borders
        )


class RemoveObjectsByDistance:
    def setup_method(self):
        image = data.hubble_deep_field()
        image = color.rgb2gray(image)
        objects = image > 0.18  # Chosen with threshold_li
        self.labels, _ = scipy.ndimage.label(objects)

    @pytest.mark.parametrize('min_distance', [5, 100])
    def test_remove_near_objects(self, min_distance):
        morphology.remove_objects_by_distance(self.labels, min_distance=min_distance)

    @pytest.mark.parametrize('min_distance', [5, 100])
    def test_peakmem_reference(self, min_distance):
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

    @pytest.mark.parametrize('min_distance', [5, 100])
    def test_peakmem_remove_near_objects(self, min_distance):
        morphology.remove_objects_by_distance(
            self.labels,
            min_distance=min_distance,
        )
