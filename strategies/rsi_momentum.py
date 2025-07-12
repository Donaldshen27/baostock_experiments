#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RSI Momentum Strategy
Trade based on RSI oversold/overbought conditions
"""

from typing import List, Dict
import pandas as pd
import numpy as np

from .base import BaseStrategy, Signal, SignalType
from core.indicators import TechnicalIndicators


class RSIMomentumStrategy(BaseStrategy):
    """
    RSI Momentum Strategy
    
    Rules:
    - Buy when RSI crosses above oversold level (momentum reversal)
    - Sell when RSI crosses below overbought level
    - Optional: Confirm with price action or volume
    """
    
    def __init__(self, params: Dict = None):
        """
        Initialize strategy
        
        Default params:
        - rsi_period: 14
        - oversold_level: 30
        - overbought_level: 70
        - use_divergence: True (look for price/RSI divergence)
        - volume_confirm: True (require volume confirmation)
        - hold_min_days: 2 (minimum holding period)
        """
        default_params = {
            'rsi_period': 14,
            'oversold_level': 30,
            'overbought_level': 70,
            'use_divergence': True,
            'volume_confirm': True,
            'hold_min_days': 2,
            'stop_loss_pct': 0.03,  # Tighter stop for momentum
            'take_profit_pct': 0.08,
            'position_size_pct': 0.15,
            'max_positions': 8
        }
        
        if params:
            default_params.update(params)
        
        super().__init__(default_params)
        
        # Track entry dates for minimum holding period
        self.entry_dates = {}
    
    def update_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate required indicators"""
        # Add RSI
        data = TechnicalIndicators.add_rsi(data, period=self.params['rsi_period'])
        
        # Add moving averages for trend context
        data = TechnicalIndicators.add_moving_averages(data, [20, 50])
        
        # Add volume indicators
        data['Volume_MA'] = data['volume'].rolling(window=20).mean()
        data['Volume_Ratio'] = data['volume'] / data['Volume_MA']
        
        # Add price rate of change for divergence
        data['ROC'] = data['close'].pct_change(periods=5) * 100
        
        # Add Bollinger Bands for volatility context
        data = TechnicalIndicators.add_bollinger_bands(data)
        
        return data
    
    def generate_signals(self, data: pd.DataFrame, stock_code: str) -> List[Signal]:
        """Generate trading signals"""
        signals = []
        
        # Need enough data for indicators
        min_period = max(self.params['rsi_period'], 50) + 5
        if len(data) < min_period:
            return signals
        
        # Check each bar for signals
        for i in range(min_period, len(data)):
            current = data.iloc[i]
            previous = data.iloc[i-1]
            
            # Skip if RSI is not available
            if pd.isna(current['RSI']) or pd.isna(previous['RSI']):
                continue
            
            # Check minimum holding period for existing positions
            if stock_code in self.entry_dates:
                days_held = (current.name - self.entry_dates[stock_code]).days
                if days_held < self.params['hold_min_days']:
                    continue
            
            # Buy Signal: RSI crosses above oversold
            if (current['RSI'] > self.params['oversold_level'] and 
                previous['RSI'] <= self.params['oversold_level']):
                
                # Volume confirmation
                if self.params['volume_confirm']:
                    if current['Volume_Ratio'] < 1.2:  # Need 20% above average volume
                        continue
                
                # Check for bullish divergence if enabled
                divergence_strength = 1.0
                if self.params['use_divergence']:
                    divergence_strength = self._check_divergence(data, i, 'bullish')
                    if divergence_strength < 0.5:
                        continue
                
                # Trend filter - prefer oversold in uptrend
                trend_score = self._assess_trend(current)
                
                # Calculate signal strength
                rsi_distance = self.params['oversold_level'] - previous['RSI']
                strength = min(1.0, (rsi_distance / 10) * divergence_strength * trend_score)
                
                signal = Signal(
                    timestamp=current.name,
                    stock_code=stock_code,
                    signal_type=SignalType.BUY,
                    price=current['close'],
                    strength=strength,
                    reason=f"RSI Oversold Reversal ({previous['RSI']:.1f} -> {current['RSI']:.1f})",
                    indicators={
                        'rsi': current['RSI'],
                        'volume_ratio': current['Volume_Ratio'],
                        'divergence': divergence_strength,
                        'trend_score': trend_score,
                        'bb_position': (current['close'] - current['BB_Lower']) / (current['BB_Upper'] - current['BB_Lower']),
                        'stop_loss_pct': self.params['stop_loss_pct'],
                        'take_profit_pct': self.params['take_profit_pct']
                    }
                )
                signals.append(signal)
                self.entry_dates[stock_code] = current.name
            
            # Sell Signal: RSI crosses below overbought
            elif (current['RSI'] < self.params['overbought_level'] and 
                  previous['RSI'] >= self.params['overbought_level']):
                
                # Check for bearish divergence
                divergence_strength = 1.0
                if self.params['use_divergence']:
                    divergence_strength = self._check_divergence(data, i, 'bearish')
                
                signal = Signal(
                    timestamp=current.name,
                    stock_code=stock_code,
                    signal_type=SignalType.SELL,
                    price=current['close'],
                    strength=divergence_strength,
                    reason=f"RSI Overbought Reversal ({previous['RSI']:.1f} -> {current['RSI']:.1f})",
                    indicators={
                        'rsi': current['RSI'],
                        'divergence': divergence_strength
                    }
                )
                signals.append(signal)
                if stock_code in self.entry_dates:
                    del self.entry_dates[stock_code]
            
            # Alternative exit: RSI extreme levels
            elif stock_code in self.positions:
                # Exit if RSI reaches extreme overbought
                if current['RSI'] > 80:
                    signal = Signal(
                        timestamp=current.name,
                        stock_code=stock_code,
                        signal_type=SignalType.SELL,
                        price=current['close'],
                        strength=1.0,
                        reason=f"RSI Extreme Overbought ({current['RSI']:.1f})"
                    )
                    signals.append(signal)
                    if stock_code in self.entry_dates:
                        del self.entry_dates[stock_code]
        
        return signals
    
    def _check_divergence(self, data: pd.DataFrame, current_idx: int, 
                         divergence_type: str = 'bullish') -> float:
        """
        Check for price/RSI divergence
        Returns strength of divergence (0-1)
        """
        lookback = 10
        if current_idx < lookback:
            return 0.5
        
        current = data.iloc[current_idx]
        
        if divergence_type == 'bullish':
            # Look for lower lows in price but higher lows in RSI
            price_lows = []
            rsi_lows = []
            
            for i in range(current_idx - lookback, current_idx):
                if (data.iloc[i]['close'] < data.iloc[i-1]['close'] and 
                    data.iloc[i]['close'] < data.iloc[i+1]['close']):
                    price_lows.append((i, data.iloc[i]['close']))
                    rsi_lows.append((i, data.iloc[i]['RSI']))
            
            if len(price_lows) >= 2:
                # Check if price made lower low but RSI made higher low
                if (price_lows[-1][1] < price_lows[-2][1] and 
                    rsi_lows[-1][1] > rsi_lows[-2][1]):
                    # Calculate divergence strength
                    price_diff = (price_lows[-2][1] - price_lows[-1][1]) / price_lows[-2][1]
                    rsi_diff = (rsi_lows[-1][1] - rsi_lows[-2][1]) / rsi_lows[-2][1]
                    return min(1.0, (price_diff + rsi_diff) * 2)
        
        else:  # bearish divergence
            # Look for higher highs in price but lower highs in RSI
            price_highs = []
            rsi_highs = []
            
            for i in range(current_idx - lookback, current_idx):
                if (data.iloc[i]['close'] > data.iloc[i-1]['close'] and 
                    data.iloc[i]['close'] > data.iloc[i+1]['close']):
                    price_highs.append((i, data.iloc[i]['close']))
                    rsi_highs.append((i, data.iloc[i]['RSI']))
            
            if len(price_highs) >= 2:
                if (price_highs[-1][1] > price_highs[-2][1] and 
                    rsi_highs[-1][1] < rsi_highs[-2][1]):
                    price_diff = (price_highs[-1][1] - price_highs[-2][1]) / price_highs[-2][1]
                    rsi_diff = (rsi_highs[-2][1] - rsi_highs[-1][1]) / rsi_highs[-2][1]
                    return min(1.0, (price_diff + rsi_diff) * 2)
        
        return 0.5  # No clear divergence
    
    def _assess_trend(self, current_data: pd.Series) -> float:
        """
        Assess trend strength (0-1)
        Higher score means stronger uptrend
        """
        score = 0.5  # Neutral
        
        # Check MA positions
        if 'MA20' in current_data and not pd.isna(current_data['MA20']):
            if current_data['close'] > current_data['MA20']:
                score += 0.25
        
        if 'MA50' in current_data and not pd.isna(current_data['MA50']):
            if current_data['close'] > current_data['MA50']:
                score += 0.25
            
            # Bonus for MA20 > MA50
            if current_data['MA20'] > current_data['MA50']:
                score += 0.25
        
        # Normalize to 0-1
        return min(1.0, score)
    
    def calculate_position_size(self, signal: Signal, portfolio_value: float, 
                              current_positions: int) -> int:
        """Calculate position size based on signal strength"""
        if current_positions >= self.params['max_positions']:
            return 0
        
        # Base position size
        base_size = portfolio_value * self.params['position_size_pct']
        
        # Adjust by signal strength
        adjusted_size = base_size * (0.5 + 0.5 * signal.strength)
        
        # Further adjust based on RSI level
        if 'rsi' in signal.indicators:
            rsi = signal.indicators['rsi']
            if rsi < 20:  # Extremely oversold
                adjusted_size *= 1.3
            elif rsi > 80:  # Extremely overbought (for shorts)
                adjusted_size *= 0.7
        
        # Calculate shares
        shares = int(adjusted_size / signal.price)
        
        return max(shares, 100) if shares > 0 else 0
    
    def get_strategy_description(self) -> str:
        """Get strategy description"""
        return f"""
        RSI Momentum Strategy
        
        Entry Rules:
        - Buy when RSI crosses above {self.params['oversold_level']}
        {'- Requires volume 20% above average' if self.params['volume_confirm'] else ''}
        {'- Checks for bullish divergence' if self.params['use_divergence'] else ''}
        - Minimum holding period: {self.params['hold_min_days']} days
        
        Exit Rules:
        - Sell when RSI crosses below {self.params['overbought_level']}
        - Sell when RSI > 80 (extreme overbought)
        - Stop loss at {self.params['stop_loss_pct']*100:.1f}%
        - Take profit at {self.params['take_profit_pct']*100:.1f}%
        
        Position Sizing:
        - {self.params['position_size_pct']*100:.0f}% base position size
        - Adjusted by signal strength and RSI level
        - Maximum {self.params['max_positions']} positions
        """