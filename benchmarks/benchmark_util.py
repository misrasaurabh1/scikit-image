# See "Writing benchmarks" in the asv docs for more information.
# https://asv.readthedocs.io/en/latest/writing_benchmarks.html
import numpy as np
from skimage import util
import pytest


class NoiseSuite:
    """Benchmark for noise routines in scikit-image."""

    def setup_method(self):
        self.image = np.zeros((5000, 5000))

    @pytest.mark.parametrize('amount,salt_vs_pepper', [
        (amount, salt_vs_pepper)
        for amount in [0.0, 0.50, 1.0]
        for salt_vs_pepper in [0.0, 0.50, 1.0]
    ])
    def test_peakmem_salt_and_pepper(self, amount, salt_vs_pepper):
        self._make_salt_and_pepper_noise(amount, salt_vs_pepper)

    @pytest.mark.parametrize('amount,salt_vs_pepper', [
        (amount, salt_vs_pepper)
        for amount in [0.0, 0.50, 1.0]
        for salt_vs_pepper in [0.0, 0.50, 1.0]
    ])
    def test_salt_and_pepper(self, amount, salt_vs_pepper):
        self._make_salt_and_pepper_noise(amount, salt_vs_pepper)

    def _make_salt_and_pepper_noise(self, amount, salt_vs_pepper):
        util.random_noise(
            self.image,
            mode="s&p",
            amount=amount,
            salt_vs_pepper=salt_vs_pepper,
        )
