
"""
Data models and classes for the Enhanced Trading Bot
"""

from dataclasses import dataclass
from datetime import datetime
from typing import List

@dataclass
class Candle:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int

@dataclass
class TrendLine:
    start_point: tuple
    end_point: tuple
    type: str
    direction: str
    strength: float
    touches: int

@dataclass
class SupportResistanceLevel:
    price: float
    type: str
    touches: int
    strength: float
    first_touch: datetime
    last_touch: datetime

@dataclass
class TradingSignal:
    timestamp: datetime
    signal_type: str
    price: float
    confidence: float
    components: List[str]
