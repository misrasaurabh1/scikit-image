import numpy as np

# guard against import of a non-existent metrics module in older skimage
try:
    from skimage import metrics
except ImportError:
    pass


class SetMetricsSuite:
    shape = (6, 6)
    coords_a = np.zeros(shape, dtype=bool)
    coords_b = np.zeros(shape, dtype=bool)

    def setup_method(self):
        points_a = (1, 0)
        points_b = (5, 2)
        self.coords_a[points_a] = True
        self.coords_b[points_b] = True

    def test_hausdorff_distance(self):
        metrics.hausdorff_distance(self.coords_a, self.coords_b)

    def test_modified_hausdorff_distance(self):
        metrics.hausdorff_distance(self.coords_a, self.coords_b, method="modified")

    def test_hausdorff_pair(self):
        metrics.hausdorff_pair(self.coords_a, self.coords_b)
