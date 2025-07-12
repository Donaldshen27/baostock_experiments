#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Data Manager Module
Handles all data fetching, caching, and preprocessing
Enhanced with AkShare integration for chip distribution and real-time data
"""

import baostock as bs
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import pickle
from pathlib import Path
import logging
from typing import Optional, Dict, List, Tuple
try:
    import akshare as ak
    AKSHARE_AVAILABLE = True
except ImportError:
    AKSHARE_AVAILABLE = False
    logging.warning("AkShare not installed. Chip distribution features will be unavailable.")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataManager:
    """
    Centralized data management for baostock data
    Features:
    - Automatic login/logout management
    - Data caching to reduce API calls
    - Data validation and cleaning
    - Batch data fetching
    """
    
    def __init__(self, cache_dir: str = "data/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._logged_in = False
        self.cache_expiry_hours = 24  # Cache data for 24 hours
        
    def _ensure_login(self):
        """Ensure we're logged into baostock"""
        if not self._logged_in:
            lg = bs.login()
            if lg.error_code == '0':
                self._logged_in = True
                logger.info("Baostock login successful")
            else:
                raise Exception(f"Baostock login failed: {lg.error_msg}")
    
    def _logout(self):
        """Logout from baostock"""
        if self._logged_in:
            bs.logout()
            self._logged_in = False
            logger.info("Baostock logout successful")
    
    def __del__(self):
        """Cleanup on deletion"""
        self._logout()
    
    def _get_cache_path(self, stock_code: str, start_date: str, end_date: str, 
                       frequency: str = "d") -> Path:
        """Generate cache file path"""
        filename = f"{stock_code}_{start_date}_{end_date}_{frequency}.pkl"
        return self.cache_dir / filename
    
    def _is_cache_valid(self, cache_path: Path) -> bool:
        """Check if cache file is still valid"""
        if not cache_path.exists():
            return False
        
        # Check file age
        file_age = datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)
        return file_age.total_seconds() < self.cache_expiry_hours * 3600
    
    def get_stock_data(self, stock_code: str, start_date: str, end_date: str,
                      frequency: str = "d", adjustflag: str = "3",
                      use_cache: bool = True) -> pd.DataFrame:
        """
        Fetch stock data with caching support
        
        Args:
            stock_code: Stock code (e.g., 'sh.600000')
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            frequency: Data frequency ('d', 'w', 'm', '5', '15', '30', '60')
            adjustflag: Adjust flag ('1': backward, '2': forward, '3': no adjust)
            use_cache: Whether to use cached data
            
        Returns:
            DataFrame with stock data
        """
        # Check cache first
        cache_path = self._get_cache_path(stock_code, start_date, end_date, frequency)
        
        if use_cache and self._is_cache_valid(cache_path):
            logger.info(f"Loading cached data for {stock_code}")
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        
        # Fetch from baostock
        logger.info(f"Fetching data from baostock for {stock_code}")
        self._ensure_login()
        
        # Define fields based on frequency
        if frequency in ['5', '15', '30', '60']:
            fields = "date,time,code,open,high,low,close,volume,amount,adjustflag"
        else:
            fields = ("date,code,open,high,low,close,preclose,volume,amount,"
                     "adjustflag,turn,tradestatus,pctChg,peTTM,psTTM,pcfNcfTTM,pbMRQ,isST")
        
        rs = bs.query_history_k_data_plus(
            stock_code,
            fields,
            start_date=start_date,
            end_date=end_date,
            frequency=frequency,
            adjustflag=adjustflag
        )
        
        if rs.error_code != '0':
            logger.error(f"Failed to fetch data: {rs.error_msg}")
            return pd.DataFrame()
        
        # Convert to DataFrame
        data_list = []
        while (rs.error_code == '0') & rs.next():
            data_list.append(rs.get_row_data())
        
        if not data_list:
            logger.warning(f"No data returned for {stock_code}")
            return pd.DataFrame()
        
        df = pd.DataFrame(data_list, columns=rs.fields)
        
        # Data preprocessing
        df = self._preprocess_data(df, frequency)
        
        # Save to cache
        if use_cache:
            with open(cache_path, 'wb') as f:
                pickle.dump(df, f)
            logger.info(f"Data cached for {stock_code}")
        
        return df
    
    def _preprocess_data(self, df: pd.DataFrame, frequency: str) -> pd.DataFrame:
        """Preprocess raw data from baostock"""
        # Convert date column
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            
        # Handle minute data datetime
        if frequency in ['5', '15', '30', '60'] and 'time' in df.columns:
            df['datetime'] = pd.to_datetime(df['date'].astype(str) + ' ' + 
                                          df['time'].str[:8])
            df = df.set_index('datetime')
        else:
            df = df.set_index('date')
        
        # Convert numeric columns
        numeric_columns = ['open', 'high', 'low', 'close', 'preclose', 
                          'volume', 'amount', 'turn', 'pctChg', 
                          'peTTM', 'psTTM', 'pcfNcfTTM', 'pbMRQ']
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove any rows with all NaN prices
        price_cols = ['open', 'high', 'low', 'close']
        available_price_cols = [col for col in price_cols if col in df.columns]
        if available_price_cols:
            df = df.dropna(subset=available_price_cols, how='all')
        
        # Sort by date
        df = df.sort_index()
        
        return df
    
    def get_stock_list(self, date: Optional[str] = None) -> pd.DataFrame:
        """
        Get list of all stocks on a given date
        
        Args:
            date: Date in 'YYYY-MM-DD' format (default: today)
            
        Returns:
            DataFrame with stock list
        """
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
        
        self._ensure_login()
        
        rs = bs.query_all_stock(day=date)
        data_list = []
        
        while (rs.error_code == '0') & rs.next():
            data_list.append(rs.get_row_data())
        
        if not data_list:
            return pd.DataFrame()
        
        df = pd.DataFrame(data_list, columns=rs.fields)
        
        # Only return tradeable stocks
        df = df[df['tradeStatus'] == '1']
        
        return df
    
    def get_index_components(self, index_code: str, date: Optional[str] = None) -> pd.DataFrame:
        """
        Get index component stocks
        
        Args:
            index_code: Index code ('hs300', 'sz50', 'zz500')
            date: Date in 'YYYY-MM-DD' format
            
        Returns:
            DataFrame with component stocks
        """
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
        
        self._ensure_login()
        
        if index_code == 'hs300':
            rs = bs.query_hs300_stocks(date=date)
        elif index_code == 'sz50':
            rs = bs.query_sz50_stocks(date=date)
        elif index_code == 'zz500':
            rs = bs.query_zz500_stocks(date=date)
        else:
            raise ValueError(f"Unsupported index code: {index_code}")
        
        data_list = []
        while (rs.error_code == '0') & rs.next():
            data_list.append(rs.get_row_data())
        
        if not data_list:
            return pd.DataFrame()
        
        return pd.DataFrame(data_list, columns=rs.fields)
    
    def get_financial_data(self, stock_code: str, year: int, quarter: int,
                          data_type: str = "profit") -> pd.DataFrame:
        """
        Get financial statement data
        
        Args:
            stock_code: Stock code
            year: Year
            quarter: Quarter (1-4)
            data_type: Type of financial data 
                      ('profit', 'operation', 'growth', 'balance', 'cash_flow', 'dupont')
                      
        Returns:
            DataFrame with financial data
        """
        self._ensure_login()
        
        query_functions = {
            'profit': bs.query_profit_data,
            'operation': bs.query_operation_data,
            'growth': bs.query_growth_data,
            'balance': bs.query_balance_data,
            'cash_flow': bs.query_cash_flow_data,
            'dupont': bs.query_dupont_data
        }
        
        if data_type not in query_functions:
            raise ValueError(f"Unsupported data type: {data_type}")
        
        query_func = query_functions[data_type]
        rs = query_func(code=stock_code, year=year, quarter=quarter)
        
        data_list = []
        while (rs.error_code == '0') & rs.next():
            data_list.append(rs.get_row_data())
        
        if not data_list:
            return pd.DataFrame()
        
        df = pd.DataFrame(data_list, columns=rs.fields)
        
        # Convert numeric columns
        for col in df.columns:
            if col not in ['code', 'pubDate', 'statDate']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    
    def get_trading_calendar(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Get trading calendar
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            DataFrame with trading dates
        """
        self._ensure_login()
        
        rs = bs.query_trade_dates(start_date=start_date, end_date=end_date)
        data_list = []
        
        while (rs.error_code == '0') & rs.next():
            data_list.append(rs.get_row_data())
        
        df = pd.DataFrame(data_list, columns=rs.fields)
        df['calendar_date'] = pd.to_datetime(df['calendar_date'])
        df['is_trading_day'] = df['is_trading_day'].astype(int)
        
        return df
    
    def batch_get_stock_data(self, stock_codes: List[str], start_date: str, 
                           end_date: str, **kwargs) -> Dict[str, pd.DataFrame]:
        """
        Batch fetch data for multiple stocks
        
        Args:
            stock_codes: List of stock codes
            start_date: Start date
            end_date: End date
            **kwargs: Additional arguments for get_stock_data
            
        Returns:
            Dictionary mapping stock codes to DataFrames
        """
        results = {}
        
        for stock_code in stock_codes:
            logger.info(f"Fetching data for {stock_code}")
            try:
                df = self.get_stock_data(stock_code, start_date, end_date, **kwargs)
                if not df.empty:
                    results[stock_code] = df
            except Exception as e:
                logger.error(f"Failed to fetch data for {stock_code}: {e}")
                continue
        
        return results
    
    def clear_cache(self, older_than_days: Optional[int] = None):
        """
        Clear cache files
        
        Args:
            older_than_days: Only clear files older than this many days
        """
        cleared_count = 0
        
        for cache_file in self.cache_dir.glob("*.pkl"):
            if older_than_days:
                file_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
                if file_age.days < older_than_days:
                    continue
            
            cache_file.unlink()
            cleared_count += 1
        
        logger.info(f"Cleared {cleared_count} cache files")
        
    def get_industry_stocks(self, industry: str, date: Optional[str] = None) -> pd.DataFrame:
        """
        Get stocks in a specific industry
        
        Args:
            industry: Industry name
            date: Date for the query
            
        Returns:
            DataFrame with stocks in the industry
        """
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
            
        self._ensure_login()
        
        # Get all stocks first
        all_stocks = self.get_stock_list(date)
        
        # Get industry classification for each stock
        industry_stocks = []
        
        for _, stock in all_stocks.iterrows():
            rs = bs.query_stock_industry(code=stock['code'])
            
            while (rs.error_code == '0') & rs.next():
                row_data = rs.get_row_data()
                if industry.lower() in row_data[2].lower():  # industry field
                    industry_stocks.append({
                        'code': stock['code'],
                        'code_name': stock['code_name'],
                        'industry': row_data[2]
                    })
        
        return pd.DataFrame(industry_stocks)
    
    # AkShare Integration Methods
    def get_chip_distribution(self, stock_code: str, adjust: str = "") -> pd.DataFrame:
        """
        Get chip distribution data from AkShare
        
        Args:
            stock_code: Stock code (e.g., 'sh.600519' or '600519')
            adjust: Adjustment type (empty string for no adjustment)
            
        Returns:
            DataFrame with chip distribution data
        """
        if not AKSHARE_AVAILABLE:
            logger.warning("AkShare not available. Cannot fetch chip distribution.")
            return pd.DataFrame()
            
        # Convert baostock format to akshare format
        if '.' in stock_code:
            code = stock_code.split('.')[1]
        else:
            code = stock_code
            
        try:
            df = ak.stock_cyq_em(symbol=code, adjust=adjust)
            return df
        except Exception as e:
            logger.error(f"Error getting chip distribution for {stock_code}: {e}")
            return pd.DataFrame()
    
    def get_stock_info_ak(self, stock_code: str) -> Dict:
        """
        Get comprehensive stock information including outstanding shares from AkShare
        
        Args:
            stock_code: Stock code (e.g., 'sh.600519' or '600519')
            
        Returns:
            Dictionary with stock information
        """
        if not AKSHARE_AVAILABLE:
            logger.warning("AkShare not available. Cannot fetch stock info.")
            return {}
            
        # Convert format
        if '.' in stock_code:
            code = stock_code.split('.')[1]
        else:
            code = stock_code
            
        try:
            df = ak.stock_individual_info_em(symbol=code)
            # Convert to dictionary
            info = {}
            for _, row in df.iterrows():
                info[row['item']] = row['value']
            return info
        except Exception as e:
            logger.error(f"Error getting stock info for {stock_code}: {e}")
            return {}
    
    def get_realtime_data(self, stock_code: str) -> Dict:
        """
        Get real-time stock data from AkShare
        
        Args:
            stock_code: Stock code (e.g., 'sh.600519' or '600519')
            
        Returns:
            Dictionary with real-time data
        """
        if not AKSHARE_AVAILABLE:
            logger.warning("AkShare not available. Cannot fetch real-time data.")
            return {}
            
        # Convert format
        if '.' in stock_code:
            code = stock_code.split('.')[1]
        else:
            code = stock_code
            
        try:
            df = ak.stock_zh_a_spot_em()
            stock_data = df[df['代码'] == code]
            if not stock_data.empty:
                return stock_data.iloc[0].to_dict()
            return {}
        except Exception as e:
            logger.error(f"Error getting real-time data for {stock_code}: {e}")
            return {}
    
    def get_stock_data_with_chip(self, 
                                stock_code: str, 
                                start_date: str, 
                                end_date: str,
                                frequency: str = 'd',
                                adjustflag: str = '3') -> pd.DataFrame:
        """
        Get historical data from Baostock and merge with chip distribution from AkShare
        
        Args:
            stock_code: Stock code (e.g., 'sh.600519')
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            frequency: Data frequency
            adjustflag: Adjustment flag
            
        Returns:
            DataFrame with price data and chip distribution metrics
        """
        # Get historical data from baostock
        df = self.get_stock_data(stock_code, start_date, end_date, frequency, adjustflag)
        
        if df.empty:
            return df
            
        # Get chip distribution data
        chip_df = self.get_chip_distribution(stock_code)
        
        if not chip_df.empty:
            # Merge chip distribution data
            chip_df['date'] = pd.to_datetime(chip_df['日期']).dt.strftime('%Y-%m-%d')
            chip_df = chip_df.rename(columns={
                '获利比例': 'profit_ratio',
                '平均成本': 'avg_cost',
                '90成本-低': 'cost_90_low',
                '90成本-高': 'cost_90_high',
                '90集中度': 'concentration_90',
                '70成本-低': 'cost_70_low',
                '70成本-高': 'cost_70_high',
                '70集中度': 'concentration_70'
            })
            
            # Select relevant columns
            chip_cols = ['date', 'profit_ratio', 'avg_cost', 'cost_90_low', 
                        'cost_90_high', 'concentration_90', 'cost_70_low', 
                        'cost_70_high', 'concentration_70']
            chip_df = chip_df[chip_cols]
            
            # Convert date index back to column for merging
            df_merge = df.reset_index()
            df_merge['date'] = df_merge['date'].dt.strftime('%Y-%m-%d')
            
            # Merge with chip data
            df_merge = pd.merge(df_merge, chip_df, on='date', how='left')
            
            # Set date back as index
            df_merge['date'] = pd.to_datetime(df_merge['date'])
            df_merge.set_index('date', inplace=True)
            
            return df_merge
        
        return df