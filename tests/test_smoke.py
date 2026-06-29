"""
tests/test_smoke.py

Import smoke test. boenet shipped a misspelled `__iniit__.py` and no test caught it;
this is that missing test — the public API must import the documented way.
"""


def test_public_api_imports():
    import arcus
    from arcus import (  # noqa: F401
        mod_select,
        ArcusMoDE,
        ModelConfig,
        get_config,
        get_tokenizer,
        pack_kept,
    )

    assert arcus.__version__
    assert callable(mod_select) and callable(get_config)
