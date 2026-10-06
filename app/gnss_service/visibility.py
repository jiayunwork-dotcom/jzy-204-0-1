from dataclasses import dataclass
import math


@dataclass(frozen=True)
class MaskProfile:
    default_elevation: float
    knots: tuple[tuple[float, float], ...]

    @classmethod
    def from_point(cls, point: dict) -> "MaskProfile":
        knots = tuple((float(a), float(e)) for a, e in point.get("mask_profile", ()))
        return cls(float(point.get("default_elevation_mask", 5.0)), knots)

    def elevation_at(self, azimuth_deg: float) -> float:
        if not self.knots:
            return self.default_elevation
        az = azimuth_deg % 360.0
        knots = self.knots
        if az < knots[0][0] or az >= knots[-1][0]:
            first_a, first_e = knots[0]
            last_a, last_e = knots[-1]
            span = first_a + 360.0 - last_a
            x = az if az < first_a else az - 360.0
            fraction = (x - last_a) / span
            return last_e + fraction * (first_e - last_e)
        for (a1, e1), (a2, e2) in zip(knots, knots[1:]):
            if a1 <= az < a2:
                fraction = (az - a1) / (a2 - a1)
                return e1 + fraction * (e2 - e1)
        return self.default_elevation

    def raised(self, delta: float) -> "MaskProfile":
        return MaskProfile(
            self.default_elevation + delta,
            tuple((a, e + delta) for a, e in self.knots),
        )


def is_above_mask(mask: MaskProfile, azimuth_deg: float, elevation_deg: float) -> bool:
    return elevation_deg >= mask.elevation_at(azimuth_deg) - 1e-12
