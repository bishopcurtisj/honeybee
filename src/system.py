from typing import NamedTuple


class Asset(NamedTuple):
    symbol: str
    price: float
    dividend: float
    history: list