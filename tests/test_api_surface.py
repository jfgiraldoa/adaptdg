import importlib

import pytest

import adaptdg


def test_public_names():
    assert set(adaptdg.__all__) == {"FlowState", "__version__"}


def test_flowstate_fields():
    fs = adaptdg.FlowState(mesh=None, theta_old=None, theta_new=None, q=None)
    assert fs.q is None


@pytest.mark.parametrize("module", ["fields", "estimators"])
def test_removed_stub_modules(module):
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(f"adaptdg.{module}")
