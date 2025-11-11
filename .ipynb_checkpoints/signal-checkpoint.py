from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple, Optional

@dataclass 
class Signal:
    '''Represents a trading signal with temporal properties'''
    signal_type: str    # 'technical', 'news', 'market_structure'
    direction: str      # 'bullish', 'bearish', 'neutral'
    strength: float     # 0.0 to 1.0
    timestamp: datetime
    details: Dict

    def get_decayed_strength(self, halflife_hours: float = 6.0) -> float:
        '''Calculates the strength of the signal considering temporal decay'''
        age_hours = (datetime.now() - self.timestamp).total_seconds() / 3600
        decay_factor = 0.5 ** (age_hours / halflife_hours)     #TO-DO: The decay_factor could be optimized 
        return self.strength * decay_factor