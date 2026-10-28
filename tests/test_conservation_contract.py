import pytest

dolfinx = pytest.importorskip("dolfinx")


@pytest.mark.skip(reason="TODO: implement once conservative transfer is extracted")
def test_transfer_preserves_integral():
    """Contract: transfer_field preserves the integral of a field to round-off."""


@pytest.mark.skip(reason="TODO: implement with the ADAP loop")
def test_water_volume_conserved_across_adaptation():
    """Contract: AdaptiveTransport.water_volume() is unchanged by mesh adaptation."""
