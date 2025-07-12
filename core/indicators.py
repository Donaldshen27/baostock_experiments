#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Technical Indicators Module
Comprehensive collection of technical indicators for A-share analysis
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict
import logging

logger = logging.getLogger(__name__)


class TechnicalIndicators:
    """
    Static methods for calculating technical indicators
    All methods modify DataFrame in-place for efficiency
    """
    
    @staticmethod
    def add_moving_averages(df: pd.DataFrame, periods: list = [5, 10, 20, 60, 120, 250]) -> pd.DataFrame:
        """Add simple moving averages"""
        for period in periods:
            if len(df) >= period:
                df[f'MA{period}'] = df['close'].rolling(window=period).mean()
        return df
    
    @staticmethod
    def add_ema(df: pd.DataFrame, periods: list = [12, 26]) -> pd.DataFrame:
        """Add exponential moving averages"""
        for period in periods:
            df[f'EMA{period}'] = df['close'].ewm(span=period, adjust=False).mean()
        return df
    
    @staticmethod
    def add_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """Add MACD indicator"""
        df[f'EMA{fast}'] = df['close'].ewm(span=fast, adjust=False).mean()
        df[f'EMA{slow}'] = df['close'].ewm(span=slow, adjust=False).mean()
        
        df['MACD'] = df[f'EMA{fast}'] - df[f'EMA{slow}']
        df['MACD_Signal'] = df['MACD'].ewm(span=signal, adjust=False).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        return df
    
    @staticmethod
    def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Add Relative Strength Index"""
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Handle division by zero
        df['RSI'] = df['RSI'].fillna(50)
        
        return df
    
    @staticmethod
    def add_bollinger_bands(df: pd.DataFrame, period: int = 20, num_std: float = 2) -> pd.DataFrame:
        """Add Bollinger Bands"""
        df['BB_Middle'] = df['close'].rolling(window=period).mean()
        bb_std = df['close'].rolling(window=period).std()
        
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * num_std)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * num_std)
        df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
        df['BB_Percent'] = (df['close'] - df['BB_Lower']) / df['BB_Width']
        
        return df
    
    @staticmethod
    def add_kdj(df: pd.DataFrame, period: int = 14, k_period: int = 3, d_period: int = 3) -> pd.DataFrame:
        """Add KDJ (Stochastic) indicator"""
        low_min = df['low'].rolling(window=period).min()
        high_max = df['high'].rolling(window=period).max()
        
        rsv = 100 * ((df['close'] - low_min) / (high_max - low_min))
        
        df['K'] = rsv.rolling(window=k_period).mean()
        df['D'] = df['K'].rolling(window=d_period).mean()
        df['J'] = 3 * df['K'] - 2 * df['D']
        
        return df
    
    @staticmethod
    def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Add Average True Range"""
        df['HL'] = df['high'] - df['low']
        df['HC'] = abs(df['high'] - df['close'].shift())
        df['LC'] = abs(df['low'] - df['close'].shift())
        
        df['TR'] = df[['HL', 'HC', 'LC']].max(axis=1)
        df['ATR'] = df['TR'].rolling(window=period).mean()
        
        # Cleanup temporary columns
        df.drop(['HL', 'HC', 'LC'], axis=1, inplace=True)
        
        return df
    
    @staticmethod
    def add_cci(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """Add Commodity Channel Index"""
        df['TP'] = (df['high'] + df['low'] + df['close']) / 3
        df['CCI'] = (df['TP'] - df['TP'].rolling(period).mean()) / (0.015 * df['TP'].rolling(period).std())
        
        return df
    
    @staticmethod
    def add_obv(df: pd.DataFrame) -> pd.DataFrame:
        """Add On Balance Volume"""
        df['OBV'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        return df
    
    @staticmethod
    def add_williams_r(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Add Williams %R"""
        high_max = df['high'].rolling(window=period).max()
        low_min = df['low'].rolling(window=period).min()
        
        df['Williams_R'] = -100 * ((high_max - df['close']) / (high_max - low_min))
        
        return df
    
    @staticmethod
    def add_mfi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Add Money Flow Index"""
        df['TP'] = (df['high'] + df['low'] + df['close']) / 3
        df['MF'] = df['TP'] * df['volume']
        
        df['MF_pos'] = df['MF'].where(df['TP'] > df['TP'].shift(), 0)
        df['MF_neg'] = df['MF'].where(df['TP'] < df['TP'].shift(), 0)
        
        mf_pos_sum = df['MF_pos'].rolling(period).sum()
        mf_neg_sum = df['MF_neg'].rolling(period).sum()
        
        df['MFI'] = 100 - (100 / (1 + mf_pos_sum / mf_neg_sum))
        
        # Cleanup
        df.drop(['MF', 'MF_pos', 'MF_neg'], axis=1, inplace=True)
        
        return df
    
    @staticmethod
    def add_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Add Average Directional Index"""
        # First calculate ATR if not present
        if 'ATR' not in df.columns:
            df = TechnicalIndicators.add_atr(df, period)
        
        # Calculate directional movements
        df['DMplus'] = df['high'].diff()
        df['DMminus'] = -df['low'].diff()
        
        df['DMplus'] = df['DMplus'].where((df['DMplus'] > 0) & (df['DMplus'] > df['DMminus']), 0)
        df['DMminus'] = df['DMminus'].where((df['DMminus'] > 0) & (df['DMminus'] > df['DMplus']), 0)
        
        df['DIplus'] = 100 * (df['DMplus'].rolling(period).mean() / df['ATR'])
        df['DIminus'] = 100 * (df['DMminus'].rolling(period).mean() / df['ATR'])
        
        df['DX'] = 100 * abs(df['DIplus'] - df['DIminus']) / (df['DIplus'] + df['DIminus'])
        df['ADX'] = df['DX'].rolling(period).mean()
        
        # Cleanup
        df.drop(['DMplus', 'DMminus'], axis=1, inplace=True)
        
        return df
    
    @staticmethod
    def add_ichimoku(df: pd.DataFrame) -> pd.DataFrame:
        """Add Ichimoku Cloud indicator"""
        # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
        nine_high = df['high'].rolling(window=9).max()
        nine_low = df['low'].rolling(window=9).min()
        df['Tenkan'] = (nine_high + nine_low) / 2
        
        # Kijun-sen (Base Line): (26-period high + 26-period low)/2
        period26_high = df['high'].rolling(window=26).max()
        period26_low = df['low'].rolling(window=26).min()
        df['Kijun'] = (period26_high + period26_low) / 2
        
        # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods
        df['SpanA'] = ((df['Tenkan'] + df['Kijun']) / 2).shift(26)
        
        # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods
        period52_high = df['high'].rolling(window=52).max()
        period52_low = df['low'].rolling(window=52).min()
        df['SpanB'] = ((period52_high + period52_low) / 2).shift(26)
        
        # Chikou Span (Lagging Span): Close price shifted back 26 periods
        df['Chikou'] = df['close'].shift(-26)
        
        return df
    
    @staticmethod
    def add_vwap(df: pd.DataFrame) -> pd.DataFrame:
        """Add Volume Weighted Average Price"""
        df['VWAP'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
        return df
    
    @staticmethod
    def add_pivots(df: pd.DataFrame) -> pd.DataFrame:
        """Add Pivot Points (Classic)"""
        df['Pivot'] = (df['high'].shift() + df['low'].shift() + df['close'].shift()) / 3
        df['R1'] = 2 * df['Pivot'] - df['low'].shift()
        df['S1'] = 2 * df['Pivot'] - df['high'].shift()
        df['R2'] = df['Pivot'] + (df['high'].shift() - df['low'].shift())
        df['S2'] = df['Pivot'] - (df['high'].shift() - df['low'].shift())
        df['R3'] = df['high'].shift() + 2 * (df['Pivot'] - df['low'].shift())
        df['S3'] = df['low'].shift() - 2 * (df['high'].shift() - df['Pivot'])
        
        return df
    
    @staticmethod
    def add_fibonacci_retracements(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """Add Fibonacci Retracement Levels"""
        rolling_high = df['high'].rolling(window=period).max()
        rolling_low = df['low'].rolling(window=period).min()
        diff = rolling_high - rolling_low
        
        df['Fib_0'] = rolling_high
        df['Fib_236'] = rolling_high - diff * 0.236
        df['Fib_382'] = rolling_high - diff * 0.382
        df['Fib_500'] = rolling_high - diff * 0.500
        df['Fib_618'] = rolling_high - diff * 0.618
        df['Fib_100'] = rolling_low
        
        return df
    
    @staticmethod
    def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Add all common indicators to DataFrame"""
        # Moving averages
        df = TechnicalIndicators.add_moving_averages(df, [5, 10, 20, 60])
        df = TechnicalIndicators.add_ema(df, [12, 26])
        
        # Momentum indicators
        df = TechnicalIndicators.add_rsi(df)
        df = TechnicalIndicators.add_macd(df)
        df = TechnicalIndicators.add_kdj(df)
        
        # Volatility indicators
        df = TechnicalIndicators.add_bollinger_bands(df)
        df = TechnicalIndicators.add_atr(df)
        
        # Volume indicators
        df = TechnicalIndicators.add_obv(df)
        
        # Trend indicators
        df = TechnicalIndicators.add_adx(df)
        
        # Other indicators
        df = TechnicalIndicators.add_cci(df)
        df = TechnicalIndicators.add_williams_r(df)
        
        return df
    
    @staticmethod
    def calculate_signals(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate common trading signals based on indicators"""
        # Golden/Death Cross
        if 'MA5' in df.columns and 'MA20' in df.columns:
            df['GoldenCross'] = (df['MA5'] > df['MA20']) & (df['MA5'].shift() <= df['MA20'].shift())
            df['DeathCross'] = (df['MA5'] < df['MA20']) & (df['MA5'].shift() >= df['MA20'].shift())
        
        # RSI Signals
        if 'RSI' in df.columns:
            df['RSI_Oversold'] = df['RSI'] < 30
            df['RSI_Overbought'] = df['RSI'] > 70
            
        # MACD Signals
        if 'MACD' in df.columns and 'MACD_Signal' in df.columns:
            df['MACD_Buy'] = (df['MACD'] > df['MACD_Signal']) & (df['MACD'].shift() <= df['MACD_Signal'].shift())
            df['MACD_Sell'] = (df['MACD'] < df['MACD_Signal']) & (df['MACD'].shift() >= df['MACD_Signal'].shift())
        
        # Bollinger Band Signals
        if 'BB_Upper' in df.columns and 'BB_Lower' in df.columns:
            df['BB_Squeeze'] = df['BB_Width'] < df['BB_Width'].rolling(20).mean()
            df['BB_BreakUpper'] = (df['close'] > df['BB_Upper']) & (df['close'].shift() <= df['BB_Upper'].shift())
            df['BB_BreakLower'] = (df['close'] < df['BB_Lower']) & (df['close'].shift() >= df['BB_Lower'].shift())
        
        # Volume Spike
        if 'volume' in df.columns:
            df['Volume_Spike'] = df['volume'] > df['volume'].rolling(20).mean() * 2
        
        return df
    
    @staticmethod
    def get_indicator_summary(df: pd.DataFrame) -> Dict[str, str]:
        """Get summary of current indicator values and signals"""
        summary = {}
        latest = df.iloc[-1]
        
        # Price position
        if 'MA20' in df.columns:
            summary['Price vs MA20'] = "Above" if latest['close'] > latest['MA20'] else "Below"
        
        # RSI
        if 'RSI' in df.columns and not pd.isna(latest['RSI']):
            rsi_val = latest['RSI']
            if rsi_val > 70:
                summary['RSI'] = f"{rsi_val:.1f} (Overbought)"
            elif rsi_val < 30:
                summary['RSI'] = f"{rsi_val:.1f} (Oversold)"
            else:
                summary['RSI'] = f"{rsi_val:.1f} (Neutral)"
        
        # MACD
        if 'MACD' in df.columns and 'MACD_Signal' in df.columns:
            if latest['MACD'] > latest['MACD_Signal']:
                summary['MACD'] = "Bullish"
            else:
                summary['MACD'] = "Bearish"
        
        # ADX
        if 'ADX' in df.columns and not pd.isna(latest['ADX']):
            adx_val = latest['ADX']
            if adx_val > 40:
                summary['ADX'] = f"{adx_val:.1f} (Strong Trend)"
            elif adx_val > 25:
                summary['ADX'] = f"{adx_val:.1f} (Trending)"
            else:
                summary['ADX'] = f"{adx_val:.1f} (No Trend)"
        
        return summary