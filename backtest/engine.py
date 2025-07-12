#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simple Backtesting Engine
Handles strategy execution and performance tracking
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from collections import defaultdict

from strategies.base import BaseStrategy, Signal, SignalType, Position

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Completed trade record"""
    stock_code: str
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float
    exit_price: float
    shares: int
    pnl: float
    pnl_percent: float
    commission: float
    hold_days: int
    exit_reason: str


@dataclass 
class BacktestResult:
    """Backtesting results container"""
    trades: List[Trade] = field(default_factory=list)
    equity_curve: pd.DataFrame = field(default_factory=pd.DataFrame)
    positions: pd.DataFrame = field(default_factory=pd.DataFrame)
    signals: List[Signal] = field(default_factory=list)
    metrics: Dict = field(default_factory=dict)
    
    @property
    def total_trades(self) -> int:
        return len(self.trades)
    
    @property
    def winning_trades(self) -> List[Trade]:
        return [t for t in self.trades if t.pnl > 0]
    
    @property
    def losing_trades(self) -> List[Trade]:
        return [t for t in self.trades if t.pnl <= 0]
    
    @property
    def win_rate(self) -> float:
        if self.total_trades == 0:
            return 0
        return len(self.winning_trades) / self.total_trades


class SimpleBacktester:
    """
    Simple backtesting engine for A-share market
    Features:
    - T+1 trading rules
    - Commission and slippage
    - Position tracking
    - Performance metrics
    """
    
    def __init__(self, 
                 initial_capital: float = 100000,
                 commission_rate: float = 0.0003,
                 slippage: float = 0.001,
                 min_commission: float = 5.0):
        """
        Initialize backtester
        
        Args:
            initial_capital: Starting capital
            commission_rate: Commission rate (0.03% default)
            slippage: Slippage percentage
            min_commission: Minimum commission per trade
        """
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage = slippage
        self.min_commission = min_commission
        
        # State tracking
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.pending_orders: Dict[str, Signal] = {}  # For T+1
        self.equity_history = []
        self.position_history = []
        
        logger.info(f"Initialized backtester with capital: {initial_capital}")
    
    def reset(self):
        """Reset backtester state"""
        self.cash = self.initial_capital
        self.positions.clear()
        self.trades.clear()
        self.pending_orders.clear()
        self.equity_history.clear()
        self.position_history.clear()
    
    def run(self, 
            strategy: BaseStrategy,
            data: Dict[str, pd.DataFrame],
            start_date: Optional[str] = None,
            end_date: Optional[str] = None) -> BacktestResult:
        """
        Run backtest for a strategy
        
        Args:
            strategy: Strategy instance
            data: Dictionary mapping stock codes to DataFrames
            start_date: Backtest start date
            end_date: Backtest end date
            
        Returns:
            BacktestResult object
        """
        logger.info(f"Starting backtest for {strategy.name}")
        
        # Reset state
        self.reset()
        strategy.reset()
        
        # Get all unique dates
        all_dates = set()
        for df in data.values():
            all_dates.update(df.index)
        
        all_dates = sorted(list(all_dates))
        
        # Filter dates if specified
        if start_date:
            all_dates = [d for d in all_dates if d >= pd.to_datetime(start_date)]
        if end_date:
            all_dates = [d for d in all_dates if d <= pd.to_datetime(end_date)]
        
        # Main backtest loop
        for i, current_date in enumerate(all_dates):
            # Process pending orders (T+1)
            self._process_pending_orders(current_date, data)
            
            # Update current market data
            current_data = self._get_current_data(current_date, data)
            
            if current_data.empty:
                continue
            
            # Generate signals for each stock
            all_signals = []
            for stock_code, stock_data in data.items():
                if current_date not in stock_data.index:
                    continue
                
                # Get historical data up to current date
                hist_data = stock_data[stock_data.index <= current_date]
                
                if len(hist_data) < 20:  # Need minimum history
                    continue
                
                # Update indicators
                hist_data = strategy.update_indicators(hist_data)
                
                # Generate signals
                signals = strategy.generate_signals(hist_data, stock_code)
                
                # Only consider the latest signal
                if signals:
                    latest_signal = signals[-1]
                    if latest_signal.timestamp == current_date:
                        all_signals.append(latest_signal)
            
            # Apply risk management
            risk_signals = strategy.apply_risk_management(current_data, self.positions, current_date)
            all_signals.extend(risk_signals)
            
            # Execute signals
            for signal in all_signals:
                self._execute_signal(signal, strategy, current_data)
            
            # Update positions value
            self._update_positions(current_data)
            
            # Record equity
            total_value = self._calculate_portfolio_value(current_data)
            self.equity_history.append({
                'date': current_date,
                'cash': self.cash,
                'positions_value': total_value - self.cash,
                'total_value': total_value,
                'returns': (total_value - self.initial_capital) / self.initial_capital
            })
            
            # Record positions
            position_snapshot = {
                'date': current_date,
                'positions': len(self.positions),
                'stocks': list(self.positions.keys())
            }
            self.position_history.append(position_snapshot)
        
        # Close all remaining positions
        self._close_all_positions(all_dates[-1], data)
        
        # Calculate metrics
        result = self._create_result(strategy)
        
        logger.info(f"Backtest completed. Total trades: {len(self.trades)}")
        
        return result
    
    def _get_current_data(self, date: pd.Timestamp, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Get current market data for all stocks"""
        current_data = []
        
        for stock_code, df in data.items():
            if date in df.index:
                row = df.loc[date].copy()
                row['stock_code'] = stock_code
                current_data.append(row)
        
        if not current_data:
            return pd.DataFrame()
        
        return pd.DataFrame(current_data).set_index('stock_code')
    
    def _execute_signal(self, signal: Signal, strategy: BaseStrategy, current_data: pd.DataFrame):
        """Execute a trading signal"""
        # Validate signal
        if signal.stock_code not in current_data.index:
            logger.warning(f"Stock {signal.stock_code} not in current data")
            return
        
        stock_data = current_data.loc[signal.stock_code]
        
        # Apply slippage
        execution_price = strategy.calculate_slippage(signal.price, signal.signal_type)
        
        if signal.signal_type == SignalType.BUY:
            # Check if we already have a position
            if signal.stock_code in self.positions:
                logger.info(f"Already have position in {signal.stock_code}")
                return
            
            # Calculate position size
            portfolio_value = self._calculate_portfolio_value(current_data)
            shares = strategy.calculate_position_size(
                signal, portfolio_value, len(self.positions)
            )
            
            if shares <= 0:
                return
            
            # Calculate cost
            cost = shares * execution_price
            commission = self._calculate_commission(cost)
            total_cost = cost + commission
            
            # Check if we have enough cash
            if total_cost > self.cash:
                # Adjust shares
                available_cash = self.cash - commission
                shares = int(available_cash / execution_price)
                if shares <= 0:
                    logger.warning(f"Insufficient cash for {signal.stock_code}")
                    return
                
                cost = shares * execution_price
                commission = self._calculate_commission(cost)
                total_cost = cost + commission
            
            # T+1 rule - add to pending orders
            self.pending_orders[signal.stock_code] = signal
            logger.info(f"Buy order placed for {signal.stock_code}: {shares} shares at {execution_price}")
            
        elif signal.signal_type == SignalType.SELL:
            # Check if we have a position
            if signal.stock_code not in self.positions:
                logger.warning(f"No position to sell in {signal.stock_code}")
                return
            
            position = self.positions[signal.stock_code]
            
            # Calculate proceeds
            proceeds = position.shares * execution_price
            commission = self._calculate_commission(proceeds)
            net_proceeds = proceeds - commission
            
            # Calculate P&L
            cost_basis = position.entry_price * position.shares
            pnl = net_proceeds - cost_basis - self._calculate_commission(cost_basis)
            pnl_percent = pnl / cost_basis
            
            # Record trade
            trade = Trade(
                stock_code=signal.stock_code,
                entry_date=position.entry_date,
                exit_date=signal.timestamp,
                entry_price=position.entry_price,
                exit_price=execution_price,
                shares=position.shares,
                pnl=pnl,
                pnl_percent=pnl_percent,
                commission=commission * 2,  # Entry + exit commission
                hold_days=(signal.timestamp - position.entry_date).days,
                exit_reason=signal.reason
            )
            self.trades.append(trade)
            
            # Update cash and remove position
            self.cash += net_proceeds
            del self.positions[signal.stock_code]
            
            logger.info(f"Sold {signal.stock_code}: P&L={pnl:.2f} ({pnl_percent:.2%})")
    
    def _process_pending_orders(self, date: pd.Timestamp, data: Dict[str, pd.DataFrame]):
        """Process pending orders (T+1)"""
        executed = []
        
        for stock_code, signal in self.pending_orders.items():
            if stock_code not in data or date not in data[stock_code].index:
                continue
            
            stock_data = data[stock_code].loc[date]
            execution_price = stock_data['open']  # Execute at open
            
            # Calculate position size again (market may have moved)
            shares = int(self.cash * 0.95 / execution_price)  # Use 95% of cash to leave room for commission
            
            if shares <= 0:
                continue
            
            # Calculate actual cost
            cost = shares * execution_price
            commission = self._calculate_commission(cost)
            total_cost = cost + commission
            
            if total_cost > self.cash:
                shares = int((self.cash - commission) / execution_price)
                if shares <= 0:
                    continue
                cost = shares * execution_price
                commission = self._calculate_commission(cost)
                total_cost = cost + commission
            
            # Create position
            position = Position(
                stock_code=stock_code,
                entry_date=date,
                entry_price=execution_price,
                shares=shares,
                position_size=cost,
                stop_loss=execution_price * (1 - signal.indicators.get('stop_loss_pct', 0.05)),
                take_profit=execution_price * (1 + signal.indicators.get('take_profit_pct', 0.10))
            )
            
            self.positions[stock_code] = position
            self.cash -= total_cost
            executed.append(stock_code)
            
            logger.info(f"Executed buy order for {stock_code}: {shares} shares at {execution_price}")
        
        # Remove executed orders
        for stock_code in executed:
            del self.pending_orders[stock_code]
    
    def _update_positions(self, current_data: pd.DataFrame):
        """Update position values with current prices"""
        for stock_code, position in self.positions.items():
            if stock_code in current_data.index:
                current_price = current_data.loc[stock_code, 'close']
                # Update position value (for tracking, not modifying position object)
                # This could be extended to track unrealized P&L
    
    def _calculate_portfolio_value(self, current_data: pd.DataFrame) -> float:
        """Calculate total portfolio value"""
        positions_value = 0
        
        for stock_code, position in self.positions.items():
            if stock_code in current_data.index:
                current_price = current_data.loc[stock_code, 'close']
                positions_value += position.shares * current_price
        
        return self.cash + positions_value
    
    def _calculate_commission(self, trade_value: float) -> float:
        """Calculate trading commission"""
        commission = trade_value * self.commission_rate
        return max(commission, self.min_commission)
    
    def _close_all_positions(self, date: pd.Timestamp, data: Dict[str, pd.DataFrame]):
        """Close all remaining positions at end of backtest"""
        positions_to_close = list(self.positions.keys())
        
        for stock_code in positions_to_close:
            if stock_code not in data:
                continue
                
            df = data[stock_code]
            if date not in df.index:
                # Use last available price
                last_date = df.index[df.index <= date][-1]
                closing_price = df.loc[last_date, 'close']
            else:
                closing_price = df.loc[date, 'close']
            
            signal = Signal(
                timestamp=date,
                stock_code=stock_code,
                signal_type=SignalType.SELL,
                price=closing_price,
                reason="Backtest end"
            )
            
            # Temporarily disable T+1 for final close
            position = self.positions[stock_code]
            proceeds = position.shares * closing_price
            commission = self._calculate_commission(proceeds)
            net_proceeds = proceeds - commission
            
            cost_basis = position.entry_price * position.shares
            pnl = net_proceeds - cost_basis - self._calculate_commission(cost_basis)
            pnl_percent = pnl / cost_basis
            
            trade = Trade(
                stock_code=stock_code,
                entry_date=position.entry_date,
                exit_date=date,
                entry_price=position.entry_price,
                exit_price=closing_price,
                shares=position.shares,
                pnl=pnl,
                pnl_percent=pnl_percent,
                commission=commission * 2,
                hold_days=(date - position.entry_date).days,
                exit_reason="Backtest end"
            )
            self.trades.append(trade)
            
            self.cash += net_proceeds
            del self.positions[stock_code]
    
    def _create_result(self, strategy: BaseStrategy) -> BacktestResult:
        """Create backtest result object"""
        # Create equity curve DataFrame
        equity_curve = pd.DataFrame(self.equity_history)
        if not equity_curve.empty:
            equity_curve.set_index('date', inplace=True)
            
            # Calculate daily returns
            equity_curve['daily_returns'] = equity_curve['total_value'].pct_change()
            
            # Calculate drawdown
            equity_curve['cummax'] = equity_curve['total_value'].cummax()
            equity_curve['drawdown'] = (equity_curve['total_value'] - equity_curve['cummax']) / equity_curve['cummax']
        
        # Create positions DataFrame
        positions_df = pd.DataFrame(self.position_history)
        
        # Calculate metrics
        metrics = self._calculate_metrics(equity_curve)
        
        return BacktestResult(
            trades=self.trades,
            equity_curve=equity_curve,
            positions=positions_df,
            signals=strategy.signals,
            metrics=metrics
        )
    
    def _calculate_metrics(self, equity_curve: pd.DataFrame) -> Dict:
        """Calculate performance metrics"""
        metrics = {}
        
        if equity_curve.empty:
            return metrics
        
        # Basic metrics
        final_value = equity_curve['total_value'].iloc[-1]
        metrics['total_return'] = (final_value - self.initial_capital) / self.initial_capital
        metrics['total_return_pct'] = metrics['total_return'] * 100
        
        # Trade metrics
        metrics['total_trades'] = len(self.trades)
        winning_trades = [t for t in self.trades if t.pnl > 0]
        losing_trades = [t for t in self.trades if t.pnl <= 0]
        
        metrics['winning_trades'] = len(winning_trades)
        metrics['losing_trades'] = len(losing_trades)
        metrics['win_rate'] = len(winning_trades) / len(self.trades) if self.trades else 0
        
        # Average trade metrics
        if winning_trades:
            metrics['avg_win'] = np.mean([t.pnl for t in winning_trades])
            metrics['avg_win_pct'] = np.mean([t.pnl_percent for t in winning_trades]) * 100
        else:
            metrics['avg_win'] = 0
            metrics['avg_win_pct'] = 0
        
        if losing_trades:
            metrics['avg_loss'] = np.mean([t.pnl for t in losing_trades])
            metrics['avg_loss_pct'] = np.mean([t.pnl_percent for t in losing_trades]) * 100
        else:
            metrics['avg_loss'] = 0
            metrics['avg_loss_pct'] = 0
        
        # Profit factor
        total_wins = sum([t.pnl for t in winning_trades]) if winning_trades else 0
        total_losses = abs(sum([t.pnl for t in losing_trades])) if losing_trades else 1
        metrics['profit_factor'] = total_wins / total_losses if total_losses > 0 else 0
        
        # Risk metrics
        if 'daily_returns' in equity_curve.columns:
            returns = equity_curve['daily_returns'].dropna()
            
            if len(returns) > 1:
                metrics['volatility'] = returns.std() * np.sqrt(252)  # Annualized
                metrics['sharpe_ratio'] = (returns.mean() * 252) / (returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
            else:
                metrics['volatility'] = 0
                metrics['sharpe_ratio'] = 0
        
        # Drawdown
        if 'drawdown' in equity_curve.columns:
            metrics['max_drawdown'] = equity_curve['drawdown'].min()
            metrics['max_drawdown_pct'] = metrics['max_drawdown'] * 100
        
        # Trading frequency
        if len(equity_curve) > 0:
            days = (equity_curve.index[-1] - equity_curve.index[0]).days
            metrics['trades_per_month'] = metrics['total_trades'] / (days / 30) if days > 0 else 0
        
        # Average holding period
        if self.trades:
            metrics['avg_hold_days'] = np.mean([t.hold_days for t in self.trades])
        else:
            metrics['avg_hold_days'] = 0
        
        return metrics