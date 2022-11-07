import ccs

def test_import():
    print(dir(ccs))
    assert ccs.__name__, "CCS not successfully imported!"