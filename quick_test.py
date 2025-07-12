#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Quick test for chip distribution"""

from core.data_manager import DataManager

dm = DataManager()
stock = "sh.600519"

# Test chip distribution
print("Testing chip distribution for 贵州茅台...")
chip_df = dm.get_chip_distribution(stock)

if not chip_df.empty:
    latest = chip_df.iloc[-1]
    print(f"\n✓ Success! Retrieved {len(chip_df)} days of data")
    print(f"  - Date: {latest['日期']}")
    print(f"  - Profit ratio: {latest['获利比例']:.1f}%")
    print(f"  - Avg cost: {latest['平均成本']:.2f}")
    print(f"  - 90% concentration: {latest['90集中度']:.1f}%")
    print(f"  - Cost range 70%: {latest['70成本-低']:.2f} - {latest['70成本-高']:.2f}")
else:
    print("✗ Failed to retrieve chip distribution data")