#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Base Strategy Framework
Abstract base class for all trading strategies
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class SignalType(Enum):
    """Trading signal types"""
    BUY = 1
    SELL = -1
    HOLD = 0


class PositionSizing(Enum):
    """Position sizing methods"""
    FIXED = "fixed"
    PERCENT_EQUITY = "percent_equity"
    KELLY = "kelly"
    VOLATILITY_BASED = "volatility_based"
    RISK_PARITY = "risk_parity"


@dataclass
class Signal:
    """Trading signal with metadata"""
    timestamp: pd.Timestamp
    stock_code: str
    signal_type: SignalType
    price: float
    strength: float = 1.0  # Signal strength (0-1)
    reason: str = ""
    indicators: Dict = None
    
    def __post_init__(self):
        if self.indicators is None:
            self.indicators = {}


@dataclass
class Position:
    """Position information"""
    stock_code: str
    entry_date: pd.Timestamp
    entry_price: float
    shares: int
    position_size: float  # Total value
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    
    @property
    def current_value(self, current_price: float) -> float:
        """Calculate current position value"""
        return self.shares * current_price
    
    @property
    def pnl(self, current_price: float) -> float:
        """Calculate profit/loss"""
        return (current_price - self.entry_price) * self.shares
    
    @property
    def pnl_percent(self, current_price: float) -> float:
        """Calculate profit/loss percentage"""
        return (current_price - self.entry_price) / self.entry_price * 100


class BaseStrategy(ABC):
    """
    Abstract base class for trading strategies
    
    All strategies must implement:
    - generate_signals: Generate buy/sell signals
    - calculate_position_size: Determine position sizing
    """
    
    def __init__(self, params: Dict = None):
        """
        Initialize strategy
        
        Args:
            params: Strategy parameters
        """
        self.params = params or {}
        self.name = self.__class__.__name__
        self.positions: Dict[str, Position] = {}
        self.signals: List[Signal] = []
        self.equity_curve = []
        
        # Default parameters
        self.initial_capital = self.params.get('initial_capital', 100000)
        self.commission_rate = self.params.get('commission_rate', 0.0003)  # 0.03%
        self.slippage = self.params.get('slippage', 0.001)  # 0.1%
        self.max_positions = self.params.get('max_positions', 10)
        self.position_sizing = self.params.get('position_sizing', PositionSizing.PERCENT_EQUITY)
        self.position_size_pct = self.params.get('position_size_pct', 0.1)  # 10% per position
        
        # Risk management
        self.stop_loss_pct = self.params.get('stop_loss_pct', None)  # e.g., 0.05 for 5%
        self.take_profit_pct = self.params.get('take_profit_pct', None)  # e.g., 0.10 for 10%
        self.max_drawdown_pct = self.params.get('max_drawdown_pct', 0.20)  # 20% max drawdown
        
        # A-share specific
        self.t_plus_1 = self.params.get('t_plus_1', True)  # T+1 trading rule
        self.price_limit = self.params.get('price_limit', 0.10)  # 10% daily limit
        
        logger.info(f"Initialized {self.name} strategy with params: {self.params}")
    
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame, stock_code: str) -> List[Signal]:
        """
        Generate trading signals based on data
        
        Args:
            data: DataFrame with OHLCV and indicators
            stock_code: Stock code
            
        Returns:
            List of trading signals
        """
        pass
    
    @abstractmethod
    def calculate_position_size(self, signal: Signal, portfolio_value: float, 
                              current_positions: int) -> int:
        """
        Calculate position size for a signal
        
        Args:
            signal: Trading signal
            portfolio_value: Current portfolio value
            current_positions: Number of current positions
            
        Returns:
            Number of shares to buy/sell
        """
        pass
    
    def apply_risk_management(self, data: pd.DataFrame, current_positions: Dict[str, Position], 
                            current_date: pd.Timestamp = None) -> List[Signal]:
        """
        Apply risk management rules (stop loss, take profit, etc.)
        
        Args:
            data: Current market data
            current_positions: Current open positions
            current_date: Current date/timestamp
            
        Returns:
            List of risk management signals
        """
        risk_signals = []
        
        # Get current date if not provided
        if current_date is None:
            # Try to get from data index if it's a DatetimeIndex
            if isinstance(data.index, pd.DatetimeIndex) and len(data) > 0:
                current_date = data.index[-1]
            else:
                current_date = pd.Timestamp.now()
        
        for stock_code, position in current_positions.items():
            if stock_code not in data.index:
                continue
                
            current_price = data.loc[stock_code, 'close']
            
            # Check stop loss
            if position.stop_loss and current_price <= position.stop_loss:
                signal = Signal(
                    timestamp=current_date,
                    stock_code=stock_code,
                    signal_type=SignalType.SELL,
                    price=current_price,
                    reason="Stop loss triggered",
                    strength=1.0
                )
                risk_signals.append(signal)
                logger.info(f"Stop loss triggered for {stock_code} at {current_price}")
            
            # Check take profit
            elif position.take_profit and current_price >= position.take_profit:
                signal = Signal(
                    timestamp=current_date,
                    stock_code=stock_code,
                    signal_type=SignalType.SELL,
                    price=current_price,
                    reason="Take profit triggered",
                    strength=1.0
                )
                risk_signals.append(signal)
                logger.info(f"Take profit triggered for {stock_code} at {current_price}")
        
        return risk_signals
    
    def validate_signal(self, signal: Signal, data: pd.DataFrame) -> bool:
        """
        Validate if signal can be executed
        
        Args:
            signal: Trading signal
            data: Market data
            
        Returns:
            True if signal is valid
        """
        # Check if stock is trading (not suspended)
        if 'tradestatus' in data.columns:
            if data.loc[data.index[-1], 'tradestatus'] != '1':
                logger.warning(f"Stock {signal.stock_code} is suspended")
                return False
        
        # Check price limits (A-share specific)
        if self.price_limit and len(data) > 1:
            prev_close = data.iloc[-2]['close']
            current_price = signal.price
            price_change = (current_price - prev_close) / prev_close
            
            if abs(price_change) >= self.price_limit:
                logger.warning(f"Stock {signal.stock_code} hit price limit: {price_change:.2%}")
                return False
        
        # Check if we can open new positions
        if signal.signal_type == SignalType.BUY:
            if len(self.positions) >= self.max_positions:
                logger.warning(f"Maximum positions ({self.max_positions}) reached")
                return False
        
        return True
    
    def calculate_commission(self, trade_value: float) -> float:
        """Calculate trading commission"""
        commission = trade_value * self.commission_rate
        # A-share minimum commission is usually 5 yuan
        return max(commission, 5.0)
    
    def calculate_slippage(self, price: float, signal_type: SignalType) -> float:
        """Calculate price after slippage"""
        if signal_type == SignalType.BUY:
            return price * (1 + self.slippage)
        else:
            return price * (1 - self.slippage)
    
    def get_strategy_stats(self) -> Dict:
        """Get strategy performance statistics"""
        if not self.signals:
            return {}
        
        buy_signals = [s for s in self.signals if s.signal_type == SignalType.BUY]
        sell_signals = [s for s in self.signals if s.signal_type == SignalType.SELL]
        
        stats = {
            'total_signals': len(self.signals),
            'buy_signals': len(buy_signals),
            'sell_signals': len(sell_signals),
            'win_rate': 0,  # To be calculated by backtester
            'avg_return': 0,  # To be calculated by backtester
            'max_drawdown': 0,  # To be calculated by backtester
            'sharpe_ratio': 0,  # To be calculated by backtester
        }
        
        return stats
    
    def reset(self):
        """Reset strategy state"""
        self.positions.clear()
        self.signals.clear()
        self.equity_curve.clear()
        logger.info(f"Strategy {self.name} reset")
    
    def update_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Update/calculate required indicators for the strategy
        Override in child classes to add specific indicators
        
        Args:
            data: Raw OHLCV data
            
        Returns:
            DataFrame with indicators
        """
        return data
    
    def get_position_size_kelly(self, win_probability: float, avg_win: float, avg_loss: float) -> float:
        """
        Calculate position size using Kelly Criterion
        
        Args:
            win_probability: Probability of winning
            avg_win: Average win amount
            avg_loss: Average loss amount (positive number)
            
        Returns:
            Fraction of capital to bet
        """
        if avg_loss == 0:
            return 0
        
        b = avg_win / avg_loss
        p = win_probability
        q = 1 - p
        
        kelly_fraction = (b * p - q) / b
        
        # Use fractional Kelly (25%) for safety
        return max(0, min(kelly_fraction * 0.25, 0.25))
    
    def get_position_size_volatility(self, data: pd.DataFrame, risk_per_trade: float = 0.02) -> int:
        """
        Calculate position size based on volatility (ATR)
        
        Args:
            data: DataFrame with ATR
            risk_per_trade: Risk per trade as fraction of capital
            
        Returns:
            Number of shares
        """
        if 'ATR' not in data.columns:
            return self.calculate_fixed_position_size(data.iloc[-1]['close'])
        
        atr = data.iloc[-1]['ATR']
        current_price = data.iloc[-1]['close']
        
        # Risk amount in currency
        risk_amount = self.initial_capital * risk_per_trade
        
        # Shares based on ATR stop
        shares = int(risk_amount / (2 * atr))
        
        return shares