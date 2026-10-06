from collections import OrderedDict
from dataclasses import dataclass
from threading import Lock

import numpy as np

from .coordinates import ecef_to_enu, enu_azimuth_elevation, geodetic_to_ecef
from .dop import DopResult, dop_from_directions
from .orbit import satellite_ecef
from .timeutils import parse_time
from .visibility import MaskProfile, is_above_mask


@dataclass(frozen=True)
class SatelliteObservation:
    satellite_id: str
    azimuth: float
    elevation: float
    enu: np.ndarray


class ObservationEngine:
    """Thread-safe calculator sharing satellite, geometry and DOP values.

    The cache is intentionally keyed by immutable version identifiers. A point
    mask edit creates a new point version, so it can never reuse a visibility
    result computed with the previous mask.
    """

    def __init__(self, position_size: int = 200_000, evaluation_size: int = 100_000):
        self.position_size = position_size
        self.evaluation_size = evaluation_size
        self._position_cache: OrderedDict[tuple[str, str, float], np.ndarray] = OrderedDict()
        self._point_cache: OrderedDict[tuple[str, float, float, float], np.ndarray] = OrderedDict()
        self._evaluation_cache: OrderedDict[tuple[str, str, float], DopResult] = OrderedDict()
        self._lock = Lock()
        self.position_calls = 0
        self.dop_calls = 0

    @staticmethod
    def _time_key(at) -> float:
        return parse_time(at).timestamp()

    def _cached_position(self, almanac_version: str, satellite: dict, tkey: float) -> np.ndarray:
        key = (almanac_version, satellite["id"], tkey)
        with self._lock:
            cached = self._position_cache.get(key)
            if cached is not None:
                self._position_cache.move_to_end(key)
                return cached
        # Orbit propagation is the expensive operation; do it outside the lock.
        self.position_calls += 1
        value = satellite_ecef(satellite, tkey)
        with self._lock:
            self._position_cache[key] = value
            self._position_cache.move_to_end(key)
            while len(self._position_cache) > self.position_size:
                self._position_cache.popitem(last=False)
        return value

    def _cached_receiver(self, point_version: str, point: dict) -> np.ndarray:
        key = (point_version, point["latitude"], point["longitude"], point["height"])
        with self._lock:
            cached = self._point_cache.get(key)
            if cached is not None:
                return cached
        value = geodetic_to_ecef(point["latitude"], point["longitude"], point["height"])
        with self._lock:
            self._point_cache[key] = value
        return value

    def observe(self, almanac: dict, point: dict, at, almanac_version: str = "none", point_version: str = "none"):
        tkey = self._time_key(at)
        receiver = self._cached_receiver(point_version, point)
        mask = MaskProfile.from_point(point)
        observations: list[SatelliteObservation] = []
        for satellite in almanac["satellites"]:
            sat_ecef = self._cached_position(almanac_version, satellite, tkey)
            enu = ecef_to_enu(receiver, point["latitude"], point["longitude"], sat_ecef)
            azimuth, elevation = enu_azimuth_elevation(enu)
            if is_above_mask(mask, azimuth, elevation):
                observations.append(
                    SatelliteObservation(satellite["id"], azimuth, elevation, tuple(enu / np.linalg.norm(enu)))
                )
        return observations

    def evaluate(self, almanac: dict, point: dict, at, almanac_version: str = "none", point_version: str = "none") -> DopResult:
        tkey = self._time_key(at)
        key = (almanac_version, point_version, tkey)
        with self._lock:
            cached = self._evaluation_cache.get(key)
            if cached is not None:
                self._evaluation_cache.move_to_end(key)
                return cached
        observations = self.observe(almanac, point, tkey, almanac_version, point_version)
        visible_ids = tuple(item.satellite_id for item in observations)
        if len(observations) < 4:
            result = DopResult(False, len(observations), visible_satellites=visible_ids)
        else:
            partial = dop_from_directions([item.enu for item in observations])
            result = DopResult(
                partial.visible,
                partial.satellite_count,
                partial.gdop,
                partial.pdop,
                partial.hdop,
                partial.vdop,
                partial.tdop,
                visible_ids,
            )
            self.dop_calls += 1
        with self._lock:
            self._evaluation_cache[key] = result
            self._evaluation_cache.move_to_end(key)
            while len(self._evaluation_cache) > self.evaluation_size:
                self._evaluation_cache.popitem(last=False)
        return result
