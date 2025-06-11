import numpy as np
from skimage.transform import SimilarityTransform, warp, resize_local_mean
import warnings
import functools
import inspect
import pytest

try:
    from skimage.util.dtype import _convert as convert
except ImportError:
    from skimage.util.dtype import convert


class WarpSuite:
    @pytest.mark.parametrize('dtype_in,N,order', [
        (dtype_in, N, order)
        for dtype_in in [np.uint8, np.uint16, np.float32, np.float64]
        for N in [128, 1024, 4096]
        for order in [0, 1, 3]
    ])
    def setup_method(self):
        pass

    def _setup_warp(self, dtype_in, N, order):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "Possible precision loss")
            self.image = convert(np.random.random((N, N)), dtype=dtype_in)
        self.tform = SimilarityTransform(
            scale=1, rotation=np.pi / 10, translation=(0, 4)
        )
        self.tform.params = self.tform.params.astype('float32')
        self.order = order

        if 'dtype' in inspect.signature(warp).parameters:
            self.warp = functools.partial(warp, dtype=self.image.dtype)
        else:
            # Keep a call to functools to have the same number of python
            # function calls
            self.warp = functools.partial(warp)

    @pytest.mark.parametrize('dtype_in,N,order', [
        (dtype_in, N, order)
        for dtype_in in [np.uint8, np.uint16, np.float32, np.float64]
        for N in [128, 1024, 4096]
        for order in [0, 1, 3]
    ])
    def test_same_type(self, dtype_in, N, order):
        """Test the case where the users wants to preserve their same low
        precision data type."""
        self._setup_warp(dtype_in, N, order)
        result = self.warp(
            self.image, self.tform, order=self.order, preserve_range=True
        )

        # convert back to input type, no-op if same type
        result = result.astype(dtype_in, copy=False)

    @pytest.mark.parametrize('dtype_in,N,order', [
        (dtype_in, N, order)
        for dtype_in in [np.uint8, np.uint16, np.float32, np.float64]
        for N in [128, 1024, 4096]
        for order in [0, 1, 3]
    ])
    def test_to_float64(self, dtype_in, N, order):
        """Test the case where want to upvert to float64 for continued
        transformations."""
        self._setup_warp(dtype_in, N, order)
        warp(self.image, self.tform, order=self.order, preserve_range=True)


class ResizeLocalMeanSuite:
    timeout = 180

    def setup_method(self):
        pass

    def _setup_resize(self, dtype, shape_in, shape_out):
        if len(shape_in) != len(shape_out):
            raise NotImplementedError("shape_in, shape_out must have same dimension")
        self.image = np.zeros(shape_in, dtype=dtype)

    @pytest.mark.parametrize('dtype,shape_in,shape_out', [
        (dtype, shape_in, shape_out)
        for dtype in [np.float32, np.float64]
        for shape_in in [(512, 512), (2048, 2048), (48, 48, 48), (192, 192, 192)]
        for shape_out in [(512, 512), (2048, 2048), (48, 48, 48), (192, 192, 192)]
    ])
    def test_resize_local_mean(self, dtype, shape_in, shape_out):
        self._setup_resize(dtype, shape_in, shape_out)
        resize_local_mean(self.image, shape_out)
