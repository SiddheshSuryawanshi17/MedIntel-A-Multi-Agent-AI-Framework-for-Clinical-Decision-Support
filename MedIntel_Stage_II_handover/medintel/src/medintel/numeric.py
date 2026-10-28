import math
from decimal import ROUND_HALF_UP, Decimal


def round3(value: float) -> float:
    """Round half up to 3 decimals, the reporting precision used throughout the pipeline."""
    return math.floor(value * 1000 + 0.5) / 1000


def number_text(value: float | int) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def fixed2(value: float) -> str:
    return str(Decimal(value).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
