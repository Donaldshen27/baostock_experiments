#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Strategy Builder Page
Create and configure trading strategies
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path
import json

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from core.data_manager import DataManager
from core.indicators import TechnicalIndicators
from strategies.base import PositionSizing

st.set_page_config(page_title="策略构建", page_icon="📈", layout="wide")

# Initialize session state
if 'strategies' not in st.session_state:
    st.session_state.strategies = {}
if 'current_strategy_config' not in st.session_state:
    st.session_state.current_strategy_config = {}

# Header
st.title("📈 策略构建器")
st.markdown("创建和配置您的量化交易策略")

# Strategy tabs
tab1, tab2, tab3, tab4 = st.tabs(["策略模板", "自定义策略", "策略管理", "策略代码"])

with tab1:
    st.markdown("### 📋 策略模板库")
    st.markdown("选择预定义的策略模板，快速开始交易")
    st.info("💡 提示：点击模板后会自动保存到策略列表，可直接进行回测")
    
    # Strategy templates
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 🔄 均线策略")
        st.markdown("""
        **双均线交叉策略**
        - 短期均线上穿长期均线时买入
        - 短期均线下穿长期均线时卖出
        - 适合趋势市场
        """)
        
        if st.button("使用此模板", key="ma_cross"):
            strategy_config = {
                "name": "MA Cross Strategy",
                "type": "ma_cross",
                "entry_conditions": ["MA交叉"],
                "exit_conditions": ["固定止损", "目标止盈"],
                "position_sizing": "等权重",
                "max_position_pct": 0.2,
                "max_positions": 5,
                "params": {
                    "short_period": 5,
                    "long_period": 20,
                    "stop_loss_pct": 0.05,
                    "take_profit_pct": 0.10
                },
                "created_at": datetime.now().isoformat()
            }
            st.session_state.current_strategy_config = strategy_config
            # Automatically save the template
            st.session_state.strategies["MA Cross Strategy"] = strategy_config
            st.success("已加载并保存均线策略模板")
    
    with col2:
        st.markdown("#### 📊 动量策略")
        st.markdown("""
        **RSI动量策略**
        - RSI超卖时买入
        - RSI超买时卖出
        - 适合震荡市场
        """)
        
        if st.button("使用此模板", key="rsi_momentum"):
            strategy_config = {
                "name": "RSI Momentum Strategy",
                "type": "rsi_momentum",
                "entry_conditions": ["RSI信号"],
                "exit_conditions": ["固定止损"],
                "position_sizing": "等权重",
                "max_position_pct": 0.15,
                "max_positions": 8,
                "params": {
                    "rsi_period": 14,
                    "oversold": 30,
                    "overbought": 70,
                    "stop_loss_pct": 0.03
                },
                "created_at": datetime.now().isoformat()
            }
            st.session_state.current_strategy_config = strategy_config
            st.session_state.strategies["RSI Momentum Strategy"] = strategy_config
            st.success("已加载并保存RSI动量策略模板")
    
    with col3:
        st.markdown("#### 📈 突破策略")
        st.markdown("""
        **布林带突破策略**
        - 价格突破上轨时买入
        - 价格跌破下轨时卖出
        - 适合波动市场
        """)
        
        if st.button("使用此模板", key="bb_breakout"):
            strategy_config = {
                "name": "Bollinger Breakout Strategy",
                "type": "bb_breakout",
                "entry_conditions": ["布林带"],
                "exit_conditions": ["固定止损", "时间止损"],
                "position_sizing": "波动率调整",
                "max_position_pct": 0.2,
                "max_positions": 5,
                "params": {
                    "bb_period": 20,
                    "bb_std": 2,
                    "hold_days": 5,
                    "stop_loss_pct": 0.04
                },
                "created_at": datetime.now().isoformat()
            }
            st.session_state.current_strategy_config = strategy_config
            st.session_state.strategies["Bollinger Breakout Strategy"] = strategy_config
            st.success("已加载并保存布林带突破策略模板")
    
    # More templates
    st.markdown("---")
    col4, col5, col6 = st.columns(3)
    
    with col4:
        st.markdown("#### 🎯 MACD策略")
        st.markdown("""
        **MACD信号策略**
        - MACD金叉买入
        - MACD死叉卖出
        - 结合趋势过滤
        """)
        
        if st.button("使用此模板", key="macd"):
            st.session_state.current_strategy_config = {
                "name": "MACD Strategy",
                "type": "macd",
                "params": {
                    "fast": 12,
                    "slow": 26,
                    "signal": 9,
                    "trend_filter": "MA60"
                }
            }
            st.success("已加载MACD策略模板")
    
    with col5:
        st.markdown("#### 💰 价值回归策略")
        st.markdown("""
        **均值回归策略**
        - 价格偏离均值买入
        - 价格回归均值卖出
        - 适合区间震荡
        """)
        
        if st.button("使用此模板", key="mean_reversion"):
            st.session_state.current_strategy_config = {
                "name": "Mean Reversion Strategy",
                "type": "mean_reversion",
                "params": {
                    "lookback": 20,
                    "entry_std": 2,
                    "exit_std": 0.5
                }
            }
            st.success("已加载均值回归策略模板")
    
    with col6:
        st.markdown("#### 🔀 组合策略")
        st.markdown("""
        **多信号组合策略**
        - 结合多个指标
        - 信号确认机制
        - 降低假信号
        """)
        
        if st.button("使用此模板", key="combo"):
            st.session_state.current_strategy_config = {
                "name": "Combo Strategy",
                "type": "combo",
                "params": {
                    "indicators": ["RSI", "MACD", "BB"],
                    "min_signals": 2
                }
            }
            st.success("已加载组合策略模板")

with tab2:
    st.markdown("### 🛠️ 自定义策略构建")
    
    # Current strategy configuration
    if st.session_state.current_strategy_config:
        st.info(f"当前编辑: {st.session_state.current_strategy_config.get('name', '未命名策略')}")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("#### 基本设置")
        
        strategy_name = st.text_input(
            "策略名称", 
            value=st.session_state.current_strategy_config.get('name', 'My Strategy')
        )
        
        strategy_type = st.selectbox(
            "策略类型",
            ["趋势跟踪", "均值回归", "动量策略", "突破策略", "混合策略"],
            index=0
        )
        
        st.markdown("#### 进场条件")
        
        entry_conditions = st.multiselect(
            "选择进场指标",
            ["MA交叉", "RSI信号", "MACD信号", "布林带", "成交量突破", "价格形态", "支撑阻力"],
            default=["MA交叉"]
        )
        
        st.markdown("#### 出场条件")
        
        exit_conditions = st.multiselect(
            "选择出场指标",
            ["固定止损", "移动止损", "目标止盈", "指标反转", "时间止损", "波动止损"],
            default=["固定止损", "目标止盈"]
        )
        
        st.markdown("#### 风险管理")
        
        position_sizing_method = st.selectbox(
            "仓位管理方法",
            ["固定仓位", "等权重", "凯利公式", "波动率调整", "风险平价"],
            index=0
        )
        
        max_position_pct = st.slider(
            "单只股票最大仓位 (%)",
            min_value=5,
            max_value=50,
            value=20,
            step=5
        )
        
        max_positions = st.number_input(
            "最大持仓数量",
            min_value=1,
            max_value=20,
            value=5
        )
    
    with col2:
        st.markdown("#### 参数配置")
        
        # Dynamic parameter configuration based on selected indicators
        params_config = {}
        
        if "MA交叉" in entry_conditions:
            st.markdown("##### 均线参数")
            col_ma1, col_ma2 = st.columns(2)
            with col_ma1:
                params_config['ma_short'] = st.number_input("短期均线", value=5, min_value=2, max_value=50)
            with col_ma2:
                params_config['ma_long'] = st.number_input("长期均线", value=20, min_value=10, max_value=200)
        
        if "RSI信号" in entry_conditions:
            st.markdown("##### RSI参数")
            col_rsi1, col_rsi2, col_rsi3 = st.columns(3)
            with col_rsi1:
                params_config['rsi_period'] = st.number_input("RSI周期", value=14, min_value=5, max_value=30)
            with col_rsi2:
                params_config['rsi_oversold'] = st.number_input("超卖线", value=30, min_value=10, max_value=40)
            with col_rsi3:
                params_config['rsi_overbought'] = st.number_input("超买线", value=70, min_value=60, max_value=90)
        
        if "MACD信号" in entry_conditions:
            st.markdown("##### MACD参数")
            col_macd1, col_macd2, col_macd3 = st.columns(3)
            with col_macd1:
                params_config['macd_fast'] = st.number_input("快线", value=12, min_value=5, max_value=20)
            with col_macd2:
                params_config['macd_slow'] = st.number_input("慢线", value=26, min_value=20, max_value=40)
            with col_macd3:
                params_config['macd_signal'] = st.number_input("信号线", value=9, min_value=5, max_value=15)
        
        if "布林带" in entry_conditions:
            st.markdown("##### 布林带参数")
            col_bb1, col_bb2 = st.columns(2)
            with col_bb1:
                params_config['bb_period'] = st.number_input("BB周期", value=20, min_value=10, max_value=50)
            with col_bb2:
                params_config['bb_std'] = st.number_input("标准差倍数", value=2.0, min_value=1.0, max_value=3.0, step=0.5)
        
        # Exit parameters
        st.markdown("##### 出场参数")
        
        if "固定止损" in exit_conditions:
            params_config['stop_loss_pct'] = st.slider(
                "止损百分比 (%)",
                min_value=1.0,
                max_value=10.0,
                value=5.0,
                step=0.5
            ) / 100
        
        if "目标止盈" in exit_conditions:
            params_config['take_profit_pct'] = st.slider(
                "止盈百分比 (%)",
                min_value=2.0,
                max_value=20.0,
                value=10.0,
                step=1.0
            ) / 100
        
        if "时间止损" in exit_conditions:
            params_config['max_hold_days'] = st.number_input(
                "最大持有天数",
                min_value=1,
                max_value=60,
                value=10
            )
        
        # Save strategy button
        col_save1, col_save2, col_save3 = st.columns([1, 1, 2])
        
        with col_save1:
            if st.button("💾 保存策略", type="primary"):
                strategy_config = {
                    "name": strategy_name,
                    "type": strategy_type,
                    "entry_conditions": entry_conditions,
                    "exit_conditions": exit_conditions,
                    "position_sizing": position_sizing_method,
                    "max_position_pct": max_position_pct / 100,
                    "max_positions": max_positions,
                    "params": params_config,
                    "created_at": datetime.now().isoformat()
                }
                
                st.session_state.strategies[strategy_name] = strategy_config
                st.session_state.current_strategy_config = strategy_config
                st.success(f"策略 '{strategy_name}' 已保存!")
        
        with col_save2:
            if st.button("🔄 重置", type="secondary"):
                st.session_state.current_strategy_config = {}
                st.rerun()
        
        # Display current configuration
        if params_config:
            st.markdown("##### 当前配置预览")
            st.json(params_config)

with tab3:
    st.markdown("### 📁 策略管理")
    
    if not st.session_state.strategies:
        st.info("还没有保存的策略。请在'自定义策略'标签页创建您的第一个策略。")
    else:
        # List saved strategies
        for strategy_name, strategy_config in st.session_state.strategies.items():
            with st.expander(f"📈 {strategy_name}"):
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    st.markdown(f"**类型**: {strategy_config['type']}")
                    st.markdown(f"**创建时间**: {strategy_config.get('created_at', 'Unknown')}")
                    st.markdown(f"**进场条件**: {', '.join(strategy_config['entry_conditions'])}")
                    st.markdown(f"**出场条件**: {', '.join(strategy_config['exit_conditions'])}")
                
                with col2:
                    if st.button("编辑", key=f"edit_{strategy_name}"):
                        st.session_state.current_strategy_config = strategy_config
                        st.rerun()
                    
                    if st.button("导出", key=f"export_{strategy_name}"):
                        # Convert strategy to JSON
                        strategy_json = json.dumps(strategy_config, indent=2, ensure_ascii=False)
                        st.download_button(
                            label="下载JSON",
                            data=strategy_json,
                            file_name=f"{strategy_name}.json",
                            mime="application/json",
                            key=f"download_{strategy_name}"
                        )
                
                with col3:
                    if st.button("删除", key=f"delete_{strategy_name}", type="secondary"):
                        del st.session_state.strategies[strategy_name]
                        st.success(f"策略 '{strategy_name}' 已删除")
                        st.rerun()
                
                # Show strategy details
                st.markdown("**参数配置:**")
                st.json(strategy_config['params'])

with tab4:
    st.markdown("### 💻 策略代码生成")
    st.markdown("基于您的配置自动生成策略代码")
    
    if st.session_state.current_strategy_config:
        config = st.session_state.current_strategy_config
        
        # Generate strategy code
        strategy_code = f'''
# Generated Strategy: {config.get('name', 'MyStrategy')}
# Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

from strategies.base import BaseStrategy, Signal, SignalType
from core.indicators import TechnicalIndicators
import pandas as pd

class {config.get('name', 'MyStrategy').replace(' ', '')}(BaseStrategy):
    """
    自动生成的策略
    类型: {config.get('type', 'Unknown')}
    """
    
    def __init__(self, params=None):
        # Default parameters
        default_params = {json.dumps(config.get('params', {}), indent=12)}
        
        if params:
            default_params.update(params)
        
        super().__init__(default_params)
        
    def update_indicators(self, data):
        """Calculate required indicators"""
        # Add all necessary indicators
        data = TechnicalIndicators.add_all_indicators(data)
        return data
    
    def generate_signals(self, data, stock_code):
        """Generate trading signals"""
        signals = []
        
        # Entry conditions
        entry_conditions = {config.get('entry_conditions', [])}
        
        for i in range(1, len(data)):
            current = data.iloc[i]
            previous = data.iloc[i-1]
            
            # Check entry conditions
            if self._check_entry_conditions(current, previous, entry_conditions):
                signal = Signal(
                    timestamp=current.name,
                    stock_code=stock_code,
                    signal_type=SignalType.BUY,
                    price=current['close'],
                    reason="Entry conditions met"
                )
                signals.append(signal)
            
            # Check exit conditions
            elif self._check_exit_conditions(current, previous):
                signal = Signal(
                    timestamp=current.name,
                    stock_code=stock_code,
                    signal_type=SignalType.SELL,
                    price=current['close'],
                    reason="Exit conditions met"
                )
                signals.append(signal)
        
        return signals
    
    def calculate_position_size(self, signal, portfolio_value, current_positions):
        """Calculate position size"""
        # Implement position sizing logic
        position_size_pct = self.params.get('max_position_pct', 0.1)
        position_value = portfolio_value * position_size_pct
        shares = int(position_value / signal.price)
        return shares

# Usage example:
# strategy = {config.get('name', 'MyStrategy').replace(' ', '')}()
# signals = strategy.generate_signals(data, 'sh.600000')
'''
        
        # Display code with syntax highlighting
        st.code(strategy_code, language='python')
        
        # Download button
        st.download_button(
            label="📥 下载策略代码",
            data=strategy_code,
            file_name=f"{config.get('name', 'strategy').replace(' ', '_').lower()}.py",
            mime="text/plain"
        )
        
        st.info("提示: 下载代码后，您可以进一步自定义和优化策略逻辑。")
    else:
        st.warning("请先创建或选择一个策略来生成代码。")