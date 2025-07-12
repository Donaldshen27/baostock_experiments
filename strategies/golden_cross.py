#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Golden Cross Strategy
A simple moving average crossover strategy
"""

from typing import List, Dict
import pandas as pd
import numpy as np

from .base import BaseStrategy, Signal, SignalType
from core.indicators import TechnicalIndicators


class GoldenCrossStrategy(BaseStrategy):
    """
    Golden Cross Strategy
    
    Rules:
    - Buy when short MA crosses above long MA (golden cross)
    - Sell when short MA crosses below long MA (death cross)
    - Optional: Only trade when price is above/below trend filter
    """
    
    def __init__(self, params: Dict = None):
        """
        Initialize strategy
        
        Default params:
        - short_period: 5
        - long_period: 20
        - trend_filter_period: 60 (0 to disable)
        - stop_loss_pct: 0.05 (5%)
        - take_profit_pct: 0.10 (10%)
        """
        default_params = {
            'short_period': 5,
            'long_period': 20,
            'trend_filter_period': 60,
            'stop_loss_pct': 0.05,
            'take_profit_pct': 0.10,
            'position_size_pct': 0.20,  # 20% per position
            'max_positions': 5
        }
        
        if params:
            default_params.update(params)
        
        super().__init__(default_params)
        
        # Validate parameters
        if self.params['short_period'] >= self.params['long_period']:
            raise ValueError("Short period must be less than long period")
    
    def update_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate required indicators"""
        # Add moving averages
        short_period = self.params['short_period']
        long_period = self.params['long_period']
        trend_period = self.params['trend_filter_period']
        
        data[f'MA{short_period}'] = data['close'].rolling(window=short_period).mean()
        data[f'MA{long_period}'] = data['close'].rolling(window=long_period).mean()
        
        if trend_period > 0:
            data[f'MA{trend_period}'] = data['close'].rolling(window=trend_period).mean()
        
        # Add ATR for volatility-based stops
        data = TechnicalIndicators.add_atr(data)
        
        # Add volume MA for filtering
        data['Volume_MA'] = data['volume'].rolling(window=20).mean()
        
        return data
    
    def generate_signals(self, data: pd.DataFrame, stock_code: str) -> List[Signal]:
        """Generate trading signals"""
        signals = []
        
        # Get parameters
        short_period = self.params['short_period']
        long_period = self.params['long_period']
        trend_period = self.params['trend_filter_period']
        
        short_ma_col = f'MA{short_period}'
        long_ma_col = f'MA{long_period}'
        trend_ma_col = f'MA{trend_period}' if trend_period > 0 else None
        
        # Need enough data
        min_period = max(short_period, long_period, trend_period if trend_period > 0 else 0)
        if len(data) < min_period + 1:
            return signals
        
        # Check for crossovers
        for i in range(min_period, len(data)):
            current = data.iloc[i]
            previous = data.iloc[i-1]
            
            # Skip if any required data is missing
            if pd.isna(current[short_ma_col]) or pd.isna(current[long_ma_col]):
                continue
            
            # Golden Cross (Buy Signal)
            if (current[short_ma_col] > current[long_ma_col] and 
                previous[short_ma_col] <= previous[long_ma_col]):
                
                # Apply trend filter if enabled
                if trend_ma_col and not pd.isna(current[trend_ma_col]):
                    if current['close'] < current[trend_ma_col]:
                        continue  # Skip if price below trend
                
                # Volume filter - only trade on above average volume
                if current['volume'] < current['Volume_MA'] * 0.8:
                    continue
                
                # Calculate signal strength based on separation
                separation = (current[short_ma_col] - current[long_ma_col]) / current[long_ma_col]
                strength = min(1.0, separation * 10)  # Scale to 0-1
                
                signal = Signal(
                    timestamp=current.name,
                    stock_code=stock_code,
                    signal_type=SignalType.BUY,
                    price=current['close'],
                    strength=strength,
                    reason=f"Golden Cross: MA{short_period} > MA{long_period}",
                    indicators={
                        'short_ma': current[short_ma_col],
                        'long_ma': current[long_ma_col],
                        'atr': current.get('ATR', 0),
                        'volume_ratio': current['volume'] / current['Volume_MA'],
                        'stop_loss_pct': self.params['stop_loss_pct'],
                        'take_profit_pct': self.params['take_profit_pct']
                    }
                )
                signals.append(signal)
            
            # Death Cross (Sell Signal)
            elif (current[short_ma_col] < current[long_ma_col] and 
                  previous[short_ma_col] >= previous[long_ma_col]):
                
                signal = Signal(
                    timestamp=current.name,
                    stock_code=stock_code,
                    signal_type=SignalType.SELL,
                    price=current['close'],
                    strength=1.0,
                    reason=f"Death Cross: MA{short_period} < MA{long_period}",
                    indicators={
                        'short_ma': current[short_ma_col],
                        'long_ma': current[long_ma_col]
                    }
                )
                signals.append(signal)
        
        return signals
    
    def calculate_position_size(self, signal: Signal, portfolio_value: float, 
                              current_positions: int) -> int:
        """
        Calculate position size based on signal strength and volatility
        """
        # Check if we can open new positions
        if current_positions >= self.params['max_positions']:
            return 0
        
        # Base position size
        base_size = portfolio_value * self.params['position_size_pct']
        
        # Adjust by signal strength
        adjusted_size = base_size * signal.strength
        
        # Volatility adjustment using ATR if available
        if 'atr' in signal.indicators and signal.indicators['atr'] > 0:
            # Reduce position size for high volatility
            current_price = signal.price
            atr = signal.indicators['atr']
            volatility_factor = atr / current_price
            
            if volatility_factor > 0.03:  # High volatility
                adjusted_size *= 0.7
            elif volatility_factor < 0.01:  # Low volatility
                adjusted_size *= 1.2
        
        # Calculate shares
        shares = int(adjusted_size / signal.price)
        
        # Ensure minimum position size
        min_shares = max(100, int(5000 / signal.price))  # At least 100 shares or 5000 yuan
        
        return max(shares, min_shares) if shares > 0 else 0
    
    def get_strategy_description(self) -> str:
        """Get strategy description"""
        return f"""
        Golden Cross Strategy ({self.params['short_period']}/{self.params['long_period']})
        
        Entry Rules:
        - Buy when {self.params['short_period']}-day MA crosses above {self.params['long_period']}-day MA
        - Volume must be above 80% of 20-day average
        {'- Price must be above ' + str(self.params['trend_filter_period']) + '-day MA' if self.params['trend_filter_period'] > 0 else ''}
        
        Exit Rules:
        - Sell when {self.params['short_period']}-day MA crosses below {self.params['long_period']}-day MA
        - Stop loss at {self.params['stop_loss_pct']*100:.1f}%
        - Take profit at {self.params['take_profit_pct']*100:.1f}%
        
        Position Sizing:
        - {self.params['position_size_pct']*100:.0f}% of portfolio per position
        - Maximum {self.params['max_positions']} positions
        - Volatility adjusted using ATR
        """