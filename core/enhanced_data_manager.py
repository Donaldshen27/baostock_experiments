"""
Enhanced Data Manager that combines Baostock and AkShare capabilities
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import baostock as bs
import akshare as ak
from typing import Optional, Dict, List, Tuple
import os
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class EnhancedDataManager:
    """
    Unified data manager that combines:
    - Baostock: Historical price/volume data, financial statements
    - AkShare: Chip distribution, outstanding shares, real-time data
    """
    
    def __init__(self, cache_dir: str = 'data/cache'):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.bs_logged_in = False
        self._login_baostock()
        
    def _login_baostock(self):
        """Login to baostock"""
        if not self.bs_logged_in:
            lg = bs.login()
            if lg.error_code == '0':
                self.bs_logged_in = True
            else:
                raise Exception(f"Baostock login failed: {lg.error_msg}")
    
    def __del__(self):
        """Logout from baostock on deletion"""
        if hasattr(self, 'bs_logged_in') and self.bs_logged_in:
            bs.logout()
    
    def get_chip_distribution(self, stock_code: str, adjust: str = "") -> pd.DataFrame:
        """
        Get chip distribution data from AkShare
        
        Args:
            stock_code: Stock code (e.g., 'sh.600519' or '600519')
            adjust: Adjustment type (empty string for no adjustment)
            
        Returns:
            DataFrame with chip distribution data
        """
        # Convert baostock format to akshare format
        if '.' in stock_code:
            code = stock_code.split('.')[1]
        else:
            code = stock_code
            
        try:
            df = ak.stock_cyq_em(symbol=code, adjust=adjust)
            return df
        except Exception as e:
            print(f"Error getting chip distribution for {stock_code}: {e}")
            return pd.DataFrame()
    
    def get_stock_info(self, stock_code: str) -> Dict:
        """
        Get comprehensive stock information including outstanding shares
        
        Args:
            stock_code: Stock code (e.g., 'sh.600519' or '600519')
            
        Returns:
            Dictionary with stock information
        """
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
            print(f"Error getting stock info for {stock_code}: {e}")
            return {}
    
    def get_realtime_data(self, stock_code: str) -> Dict:
        """
        Get real-time stock data from AkShare
        
        Args:
            stock_code: Stock code (e.g., 'sh.600519' or '600519')
            
        Returns:
            Dictionary with real-time data
        """
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
            print(f"Error getting real-time data for {stock_code}: {e}")
            return {}
    
    def get_historical_with_chip(self, 
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
            frequency: Data frequency ('d', 'w', 'm', '5', '15', '30', '60')
            adjustflag: Adjustment flag ('1': ex-rights, '2': pre-adjustment, '3': no adjustment)
            
        Returns:
            DataFrame with price data and chip distribution metrics
        """
        # Get historical data from baostock
        fields = "date,open,high,low,close,volume,amount,turn,pctChg"
        rs = bs.query_history_k_data_plus(
            stock_code, fields, start_date, end_date, 
            frequency=frequency, adjustflag=adjustflag
        )
        
        data_list = []
        while (rs.error_code == '0') & rs.next():
            data_list.append(rs.get_row_data())
        
        df = pd.DataFrame(data_list, columns=rs.fields)
        
        # Convert numeric columns
        numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'amount', 'turn', 'pctChg']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
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
            
            # Merge with historical data
            df = pd.merge(df, chip_df, on='date', how='left')
        
        # Set date as index
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
        return df
    
    def calculate_turnover_rate(self, stock_code: str, volume: float) -> float:
        """
        Calculate turnover rate using outstanding shares from AkShare
        
        Args:
            stock_code: Stock code
            volume: Trading volume
            
        Returns:
            Turnover rate (percentage)
        """
        stock_info = self.get_stock_info(stock_code)
        
        # Try to get outstanding shares (流通股本)
        outstanding_shares = None
        for key in ['流通股', '流通股本', '流通A股']:
            if key in stock_info:
                try:
                    # Convert to float, handling different units (万股, 亿股)
                    value = stock_info[key]
                    if isinstance(value, str):
                        if '万' in value:
                            outstanding_shares = float(value.replace('万', '').replace(',', '')) * 10000
                        elif '亿' in value:
                            outstanding_shares = float(value.replace('亿', '').replace(',', '')) * 100000000
                        else:
                            outstanding_shares = float(value.replace(',', ''))
                    else:
                        outstanding_shares = float(value)
                    break
                except:
                    continue
        
        if outstanding_shares and outstanding_shares > 0:
            return (volume / outstanding_shares) * 100
        else:
            return 0.0
    
    def get_market_overview(self) -> pd.DataFrame:
        """
        Get real-time market overview for all A-shares
        
        Returns:
            DataFrame with market overview data
        """
        try:
            df = ak.stock_zh_a_spot_em()
            return df
        except Exception as e:
            print(f"Error getting market overview: {e}")
            return pd.DataFrame()
    
    def get_index_realtime(self, index_code: str = "sh000001") -> Dict:
        """
        Get real-time index data
        
        Args:
            index_code: Index code (e.g., 'sh000001' for Shanghai Composite)
            
        Returns:
            Dictionary with index data
        """
        try:
            df = ak.stock_zh_index_spot()
            index_data = df[df['代码'] == index_code]
            if not index_data.empty:
                return index_data.iloc[0].to_dict()
            return {}
        except Exception as e:
            print(f"Error getting index data: {e}")
            return {}

# Example usage
if __name__ == "__main__":
    edm = EnhancedDataManager()
    
    # Get historical data with chip distribution
    df = edm.get_historical_with_chip('sh.600519', '2024-01-01', '2024-12-31')
    print("Historical data with chip distribution:")
    print(df.tail())
    
    # Get stock info including outstanding shares
    info = edm.get_stock_info('sh.600519')
    print("\nStock info:")
    for key, value in info.items():
        print(f"{key}: {value}")
    
    # Get real-time data
    realtime = edm.get_realtime_data('sh.600519')
    print("\nReal-time data:")
    print(realtime)