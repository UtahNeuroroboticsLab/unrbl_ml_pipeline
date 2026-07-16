import unrbl_ml_pipeline


def test_version_present():
    """
    Just tests that a version actually exists for this package
    """
    assert isinstance(unrbl_ml_pipeline.__version__, str)
    assert unrbl_ml_pipeline.__version__
