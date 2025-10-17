from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Tuple


MAJOR_HUBS = {"DEN", "IAD", "ATL", "LAX"}


def normalize_flight_key(op_carrier: str, fl_num: str | int) -> str:
    carrier = (op_carrier or "").strip().upper()
    num = str(fl_num).strip()
    return f"{carrier}{num}"


FLIGHT_NUM_RE = re.compile(r"^\s*([A-Za-z]{2,3})\s*-?\s*(\d{1,4})\s*$")


def parse_flight_number(user_input: str) -> Optional[Tuple[str, str]]:
    match = FLIGHT_NUM_RE.match(user_input or "")
    if not match:
        return None
    carrier, number = match.group(1).upper(), match.group(2)
    return carrier, number


@dataclass
class ConfidenceBand:
    point_minutes: float
    half_width_minutes: float

    @property
    def lower(self) -> float:
        return max(0.0, self.point_minutes - self.half_width_minutes)

    @property
    def upper(self) -> float:
        return self.point_minutes + self.half_width_minutes


def confidence_badge_level(mae: float) -> str:
    if mae <= 20:
        return "High"
    if mae <= 35:
        return "Medium"
    return "Low"

