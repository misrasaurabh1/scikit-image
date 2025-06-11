from subprocess import run, PIPE
from sys import executable
import pytest


class ImportSuite:
    """Benchmark the time it takes to import various modules"""

    def setup_method(self):
        pass

    @pytest.mark.parametrize('package_name', [
        'numpy',
        'skimage',
        'skimage.feature',
        'skimage.morphology',
        'skimage.color',
        'skimage.io',
    ])
    def test_import(self, package_name):
        run(
            executable + ' -c "import ' + package_name + '"',
            capture_output=True,
            stdin=PIPE,
            shell=True,
        )
