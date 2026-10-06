import math

import pytest

from conftest import direction
from gnss_service.dop import dop_from_directions


def test_reference_geometry_gdop():
    dops = dop_from_directions(
        [
            (0.0, 0.0, 1.0),  # zenith
            direction(0, 30),
            direction(120, 30),
            direction(240, 30),
        ]
    )
    assert dops.visible
    assert dops.gdop == pytest.approx(3.07318510155, abs=1e-9)


def test_dop_square_sum_identities():
    dops = dop_from_directions(
        [
            (0, 0, 1),
            direction(10, 20),
            direction(130, 35),
            direction(220, 45),
            direction(310, 15),
        ]
    )
    assert dops.gdop**2 == pytest.approx(dops.pdop**2 + dops.tdop**2)
    assert dops.pdop**2 == pytest.approx(dops.hdop**2 + dops.vdop**2)


def test_adding_visible_satellite_does_not_increase_gdop():
    initial = [
        (0, 0, 1),
        direction(10, 20),
        direction(130, 35),
        direction(220, 45),
    ]
    before = dop_from_directions(initial)
    after = dop_from_directions(initial + [direction(310, 15)])
    assert after.gdop <= before.gdop + 1e-12


def test_fewer_than_four_satellites_is_unavailable():
    assert dop_from_directions([(1, 0, 0), (0, 1, 0), (0, 0, 1)]).visible is False
