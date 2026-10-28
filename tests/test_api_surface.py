import importlib

import pytest

import adaptdg


def test_public_names():
    for name in ("FlowState", "AdaptiveTransport", "register_field"):
        assert hasattr(adaptdg, name)


def test_flowstate_fields():
    fs = adaptdg.FlowState(mesh=None, theta_old=None, theta_new=None, q=None)
    assert fs.q is None


def test_stubs_raise():
    with pytest.raises(NotImplementedError):
        adaptdg.AdaptiveTransport(None)
    with pytest.raises(NotImplementedError):
        adaptdg.register_field("c", "dg1")
    calls = {
        "transport": ("transport_step", (None, None, 1.0)),
        "estimators": ("cell_indicators", (None, None)),
        "mesh": ("adapt_mesh", (None, None)),
        "transfer": ("transfer_field", (None, None, None)),
    }
    for mod, (fn, args) in calls.items():
        m = importlib.import_module(f"adaptdg.{mod}")
        with pytest.raises(NotImplementedError):
            getattr(m, fn)(*args)
