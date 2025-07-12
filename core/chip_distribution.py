"""
Chip Distribution Calculation Module
计算筹码分布的核心算法
"""
import pandas as pd
import numpy as np
from typing import Tuple, Dict

class ChipDistribution:
    """
    筹码分布计算类
    基于历史成交量和换手率计算筹码在不同价位的分布
    """
    
    @staticmethod
    def calculate_chip_distribution(df: pd.DataFrame, 
                                  current_price: float,
                                  lookback_days: int = 120,
                                  decay_factor: float = 0.9,
                                  price_bins: int = 100) -> pd.DataFrame:
        """
        计算筹码分布
        
        Args:
            df: 包含 date, open, high, low, close, volume, turn 的数据
            current_price: 当前价格
            lookback_days: 回看天数
            decay_factor: 衰减系数 (基于换手率)
            price_bins: 价格区间数量
            
        Returns:
            DataFrame with columns: price, chips, profit_chips, trapped_chips, percentage
        """
        # 只使用最近的数据
        df_recent = df.tail(lookback_days).copy()
        
        # 计算价格范围
        price_min = df_recent['low'].min() * 0.95
        price_max = df_recent['high'].max() * 1.05
        
        # 创建价格区间
        price_range = np.linspace(price_min, price_max, price_bins)
        
        # 初始化筹码分布
        chip_dist = np.zeros(len(price_range))
        
        # 从最早的一天开始计算
        for idx in range(len(df_recent)):
            row = df_recent.iloc[idx]
            
            # 当天的价格范围和成交量
            day_low = row['low']
            day_high = row['high']
            day_volume = row['volume']
            day_turn = row.get('turn', 1.0)  # 换手率
            
            # 假设当天的成交量在最高价和最低价之间均匀分布
            # 也可以使用三角分布，将更多权重放在平均价格附近
            day_avg = (day_high + day_low + row['close']) / 3
            
            # 使用三角分布
            for i, price in enumerate(price_range):
                if day_low <= price <= day_high:
                    # 三角分布权重，越接近平均价权重越大
                    weight = 1 - abs(price - day_avg) / (day_high - day_low + 0.0001)
                    chip_dist[i] += day_volume * weight
            
            # 应用衰减：根据换手率，旧的筹码逐渐被换掉
            if idx < len(df_recent) - 1:  # 不是最后一天
                decay = 1 - (day_turn / 100 * decay_factor)  # 换手率越高，衰减越快
                chip_dist *= decay
        
        # 归一化筹码分布
        total_chips = chip_dist.sum()
        if total_chips > 0:
            chip_dist_pct = (chip_dist / total_chips) * 100
        else:
            chip_dist_pct = chip_dist
        
        # 创建结果DataFrame
        result_df = pd.DataFrame({
            'price': price_range,
            'chips': chip_dist,
            'chips_pct': chip_dist_pct,
            'profit_chips': np.where(price_range < current_price, chip_dist, 0),
            'trapped_chips': np.where(price_range >= current_price, chip_dist, 0),
            'profit_chips_pct': np.where(price_range < current_price, chip_dist_pct, 0),
            'trapped_chips_pct': np.where(price_range >= current_price, chip_dist_pct, 0)
        })
        
        return result_df
    
    @staticmethod
    def calculate_cost_distribution_stats(chip_df: pd.DataFrame) -> Dict:
        """
        计算筹码分布的统计信息
        
        Args:
            chip_df: 筹码分布数据
            
        Returns:
            包含平均成本、获利比例等统计信息的字典
        """
        total_chips = chip_df['chips'].sum()
        
        if total_chips == 0:
            return {
                'avg_cost': 0,
                'profit_ratio': 0,
                'trapped_ratio': 0,
                'concentration_90': 0,
                'concentration_70': 0
            }
        
        # 平均成本
        avg_cost = (chip_df['price'] * chip_df['chips']).sum() / total_chips
        
        # 获利比例
        profit_ratio = chip_df['profit_chips'].sum() / total_chips * 100
        trapped_ratio = chip_df['trapped_chips'].sum() / total_chips * 100
        
        # 计算集中度
        # 90%筹码的价格区间
        cumsum = chip_df['chips'].cumsum() / total_chips
        idx_5 = np.where(cumsum >= 0.05)[0][0] if any(cumsum >= 0.05) else 0
        idx_95 = np.where(cumsum >= 0.95)[0][0] if any(cumsum >= 0.95) else len(chip_df)-1
        concentration_90 = (chip_df.iloc[idx_95]['price'] - chip_df.iloc[idx_5]['price']) / avg_cost * 100
        
        # 70%筹码的价格区间
        idx_15 = np.where(cumsum >= 0.15)[0][0] if any(cumsum >= 0.15) else 0
        idx_85 = np.where(cumsum >= 0.85)[0][0] if any(cumsum >= 0.85) else len(chip_df)-1
        concentration_70 = (chip_df.iloc[idx_85]['price'] - chip_df.iloc[idx_15]['price']) / avg_cost * 100
        
        return {
            'avg_cost': avg_cost,
            'profit_ratio': profit_ratio,
            'trapped_ratio': trapped_ratio,
            'concentration_90': concentration_90,
            'concentration_70': concentration_70,
            'cost_90_low': chip_df.iloc[idx_5]['price'],
            'cost_90_high': chip_df.iloc[idx_95]['price'],
            'cost_70_low': chip_df.iloc[idx_15]['price'],
            'cost_70_high': chip_df.iloc[idx_85]['price']
        }