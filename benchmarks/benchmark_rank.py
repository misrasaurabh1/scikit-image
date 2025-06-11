import numpy as np
from skimage.filters import rank
from skimage.filters.rank import __all__ as all_rank_filters
from skimage.filters.rank import __3Dfilters as all_3d_rank_filters
from skimage.morphology import disk, ball
import pytest


class RankSuite:
    def setup_method(self):
        pass

    def _setup_rank(self, filter_func, shape):
        self.image = np.random.randint(0, 255, size=shape, dtype=np.uint8)
        self.footprint = disk(1)

    @pytest.mark.parametrize('filter_func,shape', [
        (filter_func, shape)
        for filter_func in sorted(all_rank_filters)
        for shape in [(32, 32), (256, 256)]
    ])
    def test_filter(self, filter_func, shape):
        self._setup_rank(filter_func, shape)
        getattr(rank, filter_func)(self.image, self.footprint)


class Rank3DSuite:
    def setup_method(self):
        pass

    def _setup_rank3d(self, filter3d, shape3d):
        self.volume = np.random.randint(0, 255, size=shape3d, dtype=np.uint8)
        self.footprint_3d = ball(1)

    @pytest.mark.parametrize('filter3d,shape3d', [
        (filter3d, shape3d)
        for filter3d in sorted(all_3d_rank_filters)
        for shape3d in [(32, 32, 32), (128, 128, 128)]
    ])
    def test_3d_filters(self, filter3d, shape3d):
        self._setup_rank3d(filter3d, shape3d)
        getattr(rank, filter3d)(self.volume, self.footprint_3d)
