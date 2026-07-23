from importlib.metadata import version

import catan


def test_version_matches_package_metadata():
    assert catan.__version__ == version("catan")