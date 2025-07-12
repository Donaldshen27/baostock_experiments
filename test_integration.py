#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script for AkShare and Baostock integration
"""

from core.data_manager import DataManager
import pandas as pd

def test_integration():
    print("Testing AkShare + Baostock Integration\n")
    print("="*50)
    
    # Initialize data manager
    dm = DataManager()
    
    # Test stock
    test_stock = "sh.600519"  # 贵州茅台
    
    print(f"\n1. Testing basic historical data (Baostock):")
    df = dm.get_stock_data(test_stock, "2024-12-01", "2024-12-31")
    print(f"   - Retrieved {len(df)} days of data")
    print(f"   - Columns: {list(df.columns)}")
    print(f"   - Latest close: {df['close'].iloc[-1]}")
    
    print(f"\n2. Testing chip distribution (AkShare):")
    chip_df = dm.get_chip_distribution(test_stock)
    if not chip_df.empty:
        print(f"   - Retrieved {len(chip_df)} days of chip data")
        print(f"   - Columns: {list(chip_df.columns)}")
        latest = chip_df.iloc[-1]
        print(f"   - Latest profit ratio: {latest['获利比例']:.1f}%")
        print(f"   - Latest avg cost: {latest['平均成本']:.2f}")
        print(f"   - 90% concentration: {latest['90集中度']:.1f}%")
    else:
        print("   - No chip distribution data available")
    
    print(f"\n3. Testing stock info with outstanding shares (AkShare):")
    info = dm.get_stock_info_ak(test_stock)
    if info:
        print(f"   - Retrieved {len(info)} info fields")
        for key in ['总股本', '流通股', '流通市值', '总市值']:
            if key in info:
                print(f"   - {key}: {info[key]}")
    else:
        print("   - No stock info available")
    
    print(f"\n4. Testing real-time data (AkShare):")
    realtime = dm.get_realtime_data(test_stock)
    if realtime:
        print(f"   - Stock name: {realtime.get('名称', 'N/A')}")
        print(f"   - Current price: {realtime.get('最新价', 'N/A')}")
        print(f"   - Change %: {realtime.get('涨跌幅', 'N/A')}")
    else:
        print("   - No real-time data available")
    
    print(f"\n5. Testing combined data (historical + chip):")
    combined_df = dm.get_stock_data_with_chip(test_stock, "2024-12-01", "2024-12-31")
    print(f"   - Retrieved {len(combined_df)} days of combined data")
    print(f"   - Total columns: {len(combined_df.columns)}")
    chip_cols = [col for col in combined_df.columns if 'cost' in col or 'profit' in col or 'concentration' in col]
    print(f"   - Chip-related columns: {chip_cols}")
    
    print("\n" + "="*50)
    print("Integration test completed successfully!")

if __name__ == "__main__":
    test_integration()