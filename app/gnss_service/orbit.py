import math

import numpy as np

from .constants import J2000, SECONDS_PER_DAY, WGS84_EARTH_ROTATION, WGS84_MU
from .timeutils import parse_time


def earth_rotation_angle(at: object) -> float:
    """Greenwich rotation angle (radians), equivalent to ERA for this simple model."""
    dt = parse_time(at)
    # Julian date based on the POSIX timestamp avoids date arithmetic edge cases.
    julian_ut1 = dt.timestamp() / SECONDS_PER_DAY + 2440587.5
    tu = julian_ut1 - J2000
    # IERS Earth rotation angle, reduced modulo 2*pi.
    fraction = tu - math.floor(tu)
    angle = 2.0 * math.pi * (fraction + 0.7790572732640 + 0.00273781191135448 * tu)
    return angle % (2.0 * math.pi)


def solve_kepler(mean_anomaly: float, eccentricity: float, iterations: int = 30) -> float:
    m = mean_anomaly % (2.0 * math.pi)
    eccentric_anomaly = m if eccentricity < 0.8 else math.pi
    for _ in range(iterations):
        next_value = eccentric_anomaly - (
            eccentric_anomaly - eccentricity * math.sin(eccentric_anomaly) - m
        ) / (1.0 - eccentricity * math.cos(eccentric_anomaly))
        if abs(next_value - eccentric_anomaly) < 1e-14:
            return next_value
        eccentric_anomaly = next_value
    return eccentric_anomaly


def satellite_ecef(satellite: dict, at: object) -> np.ndarray:
    """Evaluate an ideal Keplerian almanac satellite in an ECEF, metres frame.

    RAAN is interpreted in the Earth-fixed inertial direction at the satellite's
    reference epoch; Earth rotation is then applied with the IERS ERA formula.
    The supplied linear RAAN rate is interpreted as radians/second after model
    validation converts degrees/day.
    """
    t = parse_time(at)
    reference = parse_time(satellite["reference_time"])
    dt = (t - reference).total_seconds()

    a = float(satellite["semi_major_axis"])
    e = float(satellite["eccentricity"])
    inclination = float(satellite["inclination"])
    omega_raan = float(satellite["raan"]) + float(satellite["raan_rate"]) * dt
    omega_perigee = float(satellite["argument_of_perigee"])
    mean_anomaly = float(satellite["mean_anomaly"])

    mean_motion = math.sqrt(WGS84_MU / (a**3))
    eccentric_anomaly = solve_kepler(mean_anomaly + mean_motion * dt, e)

    x_orb = a * (math.cos(eccentric_anomaly) - e)
    y_orb = a * math.sqrt(1.0 - e * e) * math.sin(eccentric_anomaly)

    raan = omega_raan
    argp = omega_perigee
    incl = inclination
    cos_o, sin_o = math.cos(raan), math.sin(raan)
    cos_w, sin_w = math.cos(argp), math.sin(argp)
    cos_i, sin_i = math.cos(incl), math.sin(incl)

    # Combined Rz(RAAN) Rx(inclination) Rz(argument of perigee).
    r11 = cos_o * cos_w - sin_o * sin_w * cos_i
    r12 = -cos_o * sin_w - sin_o * cos_w * cos_i
    r21 = sin_o * cos_w + cos_o * sin_w * cos_i
    r22 = -sin_o * sin_w + cos_o * cos_w * cos_i
    r31 = sin_w * sin_i
    r32 = cos_w * sin_i
    x_eci = r11 * x_orb + r12 * y_orb
    y_eci = r21 * x_orb + r22 * y_orb
    z_eci = r31 * x_orb + r32 * y_orb

    theta = earth_rotation_angle(t)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    return np.array(
        (
            cos_t * x_eci + sin_t * y_eci,
            -sin_t * x_eci + cos_t * y_eci,
            z_eci,
        ),
        dtype=float,
    )
