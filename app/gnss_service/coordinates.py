import math

import numpy as np

from .constants import WGS84_A, WGS84_E2


def geodetic_to_ecef(latitude_deg: float, longitude_deg: float, height: float) -> np.ndarray:
    lat = math.radians(latitude_deg)
    lon = math.radians(longitude_deg)
    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)
    prime_radius = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
    return np.array(
        (
            (prime_radius + height) * cos_lat * math.cos(lon),
            (prime_radius + height) * cos_lat * math.sin(lon),
            (prime_radius * (1.0 - WGS84_E2) + height) * sin_lat,
        ),
        dtype=float,
    )


def ecef_to_enu(receiver_ecef: np.ndarray, latitude_deg: float, longitude_deg: float, target_ecef: np.ndarray) -> np.ndarray:
    lat = math.radians(latitude_deg)
    lon = math.radians(longitude_deg)
    delta = np.asarray(target_ecef, dtype=float) - np.asarray(receiver_ecef, dtype=float)
    sin_lat, cos_lat = math.sin(lat), math.cos(lat)
    sin_lon, cos_lon = math.sin(lon), math.cos(lon)
    east = -sin_lon * delta[0] + cos_lon * delta[1]
    north = -sin_lat * cos_lon * delta[0] - sin_lat * sin_lon * delta[1] + cos_lat * delta[2]
    up = cos_lat * cos_lon * delta[0] + cos_lat * sin_lon * delta[1] + sin_lat * delta[2]
    return np.array((east, north, up), dtype=float)


def enu_azimuth_elevation(enu: np.ndarray) -> tuple[float, float]:
    east, north, up = [float(value) for value in np.asarray(enu, dtype=float).reshape(3)]
    horizontal = math.hypot(east, north)
    elevation = math.degrees(math.atan2(up, horizontal))
    azimuth = math.degrees(math.atan2(east, north)) % 360.0
    return azimuth, elevation
