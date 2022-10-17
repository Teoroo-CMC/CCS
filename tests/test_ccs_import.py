import ccs

def test_import():
    assert ccs.__version__=="0.1.0", "CCS did not import, or version has changed!"