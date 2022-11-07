import ccs

def test_import():
    print(dir(ccs))
    assert ccs.__version__, "CCS not successfully imported!"