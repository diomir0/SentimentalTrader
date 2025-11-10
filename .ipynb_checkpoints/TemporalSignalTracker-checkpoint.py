from .Signal.py import Signal
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

class TemporalSignalTracker:
    '''
    Tracks all trading signals with temporal decay
    Enables reassessment when old signals lose relevance 
    '''
    def __init__(self, halflife_hours: float = 6.0, max_age_hours: float = 72.0):
        self.signals_by_symbol = defaultdict(list)
        self.halflife_hours = halflife_hours
        self.max_age_hours = max_age_hours

    def add_signal(self, symbol: str, signal: Signal):
        '''Add new signal and clean up old ones'''
        self.signals_by_symbol[symbol].append(signal)
        self._cleanup_old_signals(symbol)

    def _cleanup_old_signals(self, symbol: str):
        '''Remove signals that are older than max_age_hours'''
        cutoff = datetime.now() - timedelta(hours=self.max_age_hours)
        self.signals_by_symbol[symbol] = [
            s for s in self.signals_by_symbol[symbol]
            if s.timestamp > cutoff
        ]

    def get_current_confidence(self, symbol: str) -> Dict:
        '''
        Calculates current confidence considering temporal decay
        Returns breakdown by signal type
        '''
        signals = self.signals_by_symbol[symbol]

        if not signals:
            return {
                'total': 0.0,
                'technical': 0.0, 
                'news': 0.0,
                'market': 0.0,
                'signal_count': 0, 
                'oldest_signal_age_hours': 0
            }
        # Calculate decayed strengths by type
        by_type = defaultdict(list)
        for signal in signals:
            decayed = signal.get_decayed_strength(self.halflife_hours)
            by_type[signal.signal_type].append(decayed)

        # Aggregate
        confidence = {
            'technical': np.mean(by_type['technical']) if by_type['technical'] else 0.0,
            'news': np.mean(by_type['news']) if by_type['news'] else 0.0,
            'market': np.mean(by_type['market_structure']) if by_type['market_structure'] else 0.0,
            'signal_count': len(signals),
            'oldest_signal_age_hours': (datetime.now() - min(s.timestamp for s in signals)).total_seconds / 3600
        }

        # Total confidence is a weighted average - TO-DO: Optimize weights
        confidence['total'] = (
            confidence['technical'] * 0.5 +
            confidence['news'] * 0.3 +
            confidence['market'] *0.2
        )
        
        return confidence

    def should_reassess(self, symbol: str, threshold: float = 0.3) -> Tuple[bool, str]:
        '''
        Determines if the position should be reassessed due to signal decay
        Returns (should_reassess, reason)
        '''
        confidence = self.get_current_confidence(symbol)

        if confidence['total'] < threshold:
            return True, f"Signal confidence decayed to {confidence['total']:.2f} (threshold: {threshold})"

        if confidence['oldest_signal_age_hours'] > 48:
            return True, f"Oldest signal is {confidence['odlest_signal_age_hours']:.1f} hours old (reassessment at 48 hours)"

        return False, "Signal still fresh"

    def get_signal_summary(self, symbol: str) -> str:
        '''Generate human-readable summary of tcurretn signals'''
        confidence = self.get_current_confidence(symbol)
        signals = self.signals_by_symbol[symbol]

        if not signals:
            return "No active signals"

        summary = f'''Signal status for {symbol}:
        - Total confidence: {confidence['total']:.2f}
        - Technical: {confidence['technical']:.2f}
        - News: {confidence['news']:.2f}
        - Market: {confidence['market_structure']:.2f}
        - Active signals: {confidence['signal_count']:.1f}
        - Oldest signal: {confidence['oldest_signal_age_hours']:.1f}h ago 
        '''
        return summary