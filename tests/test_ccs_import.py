import ccs

def test_import():
    print(ccs.__path__)
    print(dir(ccs))
    assert ccs.__name__, "CCS not successfully imported!"