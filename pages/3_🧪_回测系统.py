#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Backtesting System Page
Run and analyze strategy backtests
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from core.data_manager import DataManager
from core.indicators import TechnicalIndicators
from backtest.engine import SimpleBacktester, BacktestResult
from strategies.base import BaseStrategy, Signal, SignalType
import json

st.set_page_config(page_title="回测系统", page_icon="🧪", layout="wide")

# Initialize session state
if 'backtest_results' not in st.session_state:
    st.session_state.backtest_results = {}
if 'strategies' not in st.session_state:
    st.session_state.strategies = {}

# Ensure backtest_results is always a dict
if st.session_state.backtest_results is None:
    st.session_state.backtest_results = {}

# Header
st.title("🧪 策略回测系统")
st.markdown("测试您的策略在历史数据上的表现")

# Initialize data manager
@st.cache_resource
def get_data_manager():
    return DataManager()

dm = get_data_manager()

# Sidebar configuration
with st.sidebar:
    st.markdown("### ⚙️ 回测配置")
    
    # Strategy selection
    if st.session_state.strategies:
        strategy_name = st.selectbox(
            "选择策略",
            list(st.session_state.strategies.keys())
        )
        selected_strategy = st.session_state.strategies[strategy_name]
    else:
        st.warning("请先在策略构建页面创建策略")
        st.stop()
    
    # Stock selection
    st.markdown("#### 股票池")
    
    stock_source = st.radio(
        "选择方式",
        ["热门股票", "指数成份股", "自定义"]
    )
    
    if stock_source == "热门股票":
        popular_stocks = {
            "浦发银行": "sh.600000",
            "万科A": "sz.000002", 
            "贵州茅台": "sh.600519",
            "中国平安": "sh.601318",
            "招商银行": "sh.600036",
            "五粮液": "sz.000858",
            "比亚迪": "sz.002594",
            "宁德时代": "sz.300750"
        }
        
        selected_names = st.multiselect(
            "选择股票",
            list(popular_stocks.keys()),
            default=list(popular_stocks.keys())[:3]
        )
        stock_codes = [popular_stocks[name] for name in selected_names]
        
    elif stock_source == "指数成份股":
        index = st.selectbox(
            "选择指数",
            ["沪深300", "上证50", "中证500"]
        )
        # In real implementation, fetch from data manager
        # For now, use sample stocks
        stock_codes = ["sh.600000", "sz.000002", "sh.600519"]
        st.info(f"将使用{index}成份股进行回测")
        
    else:
        custom_codes = st.text_area(
            "输入股票代码（每行一个）",
            "sh.600000\nsz.000002\nsh.600519"
        )
        stock_codes = [code.strip() for code in custom_codes.split('\n') if code.strip()]
    
    # Date range
    st.markdown("#### 回测时间")
    
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "开始日期",
            value=datetime.now() - timedelta(days=365),
            max_value=datetime.now()
        )
    
    with col2:
        end_date = st.date_input(
            "结束日期",
            value=datetime.now(),
            max_value=datetime.now()
        )
    
    # Capital settings
    st.markdown("#### 资金设置")
    
    initial_capital = st.number_input(
        "初始资金",
        min_value=10000,
        max_value=10000000,
        value=100000,
        step=10000
    )
    
    # Advanced settings
    with st.expander("高级设置"):
        commission_rate = st.number_input(
            "手续费率 (%)",
            min_value=0.01,
            max_value=0.5,
            value=0.03,
            step=0.01
        ) / 100
        
        slippage = st.number_input(
            "滑点 (%)",
            min_value=0.0,
            max_value=1.0,
            value=0.1,
            step=0.1
        ) / 100
        
        min_commission = st.number_input(
            "最低手续费",
            min_value=0.0,
            max_value=10.0,
            value=5.0,
            step=1.0
        )
    
    # Run backtest button
    run_backtest = st.button("🚀 运行回测", type="primary", use_container_width=True)

# Main content area
if run_backtest:
    with st.spinner("正在加载数据..."):
        # Load data for all stocks
        data = {}
        failed_stocks = []
        
        progress_bar = st.progress(0)
        for i, stock_code in enumerate(stock_codes):
            try:
                df = dm.get_stock_data(
                    stock_code,
                    start_date.strftime('%Y-%m-%d'),
                    end_date.strftime('%Y-%m-%d')
                )
                if not df.empty:
                    df = TechnicalIndicators.add_all_indicators(df)
                    data[stock_code] = df
            except Exception as e:
                failed_stocks.append(stock_code)
                st.warning(f"无法加载 {stock_code} 的数据: {e}")
            
            progress_bar.progress((i + 1) / len(stock_codes))
        
        progress_bar.empty()
        
        if not data:
            st.error("无法加载任何股票数据")
            st.stop()
        
        st.success(f"成功加载 {len(data)} 只股票的数据")
    
    # Create a simple strategy based on configuration
    class ConfigurableStrategy(BaseStrategy):
        """Strategy created from configuration"""
        
        def __init__(self, config):
            self.config = config
            super().__init__(config.get('params', {}))
            
        def generate_signals(self, data, stock_code):
            signals = []
            
            # Simple implementation based on entry conditions
            entry_conditions = self.config.get('entry_conditions', [])
            
            for i in range(20, len(data)):  # Need history for indicators
                current = data.iloc[i]
                prev = data.iloc[i-1]
                
                # Check MA crossover
                if "MA交叉" in entry_conditions:
                    if 'MA5' in data.columns and 'MA20' in data.columns:
                        if (current['MA5'] > current['MA20'] and 
                            prev['MA5'] <= prev['MA20']):
                            signals.append(Signal(
                                timestamp=current.name,
                                stock_code=stock_code,
                                signal_type=SignalType.BUY,
                                price=current['close'],
                                reason="MA Golden Cross"
                            ))
                        elif (current['MA5'] < current['MA20'] and 
                              prev['MA5'] >= prev['MA20']):
                            signals.append(Signal(
                                timestamp=current.name,
                                stock_code=stock_code,
                                signal_type=SignalType.SELL,
                                price=current['close'],
                                reason="MA Death Cross"
                            ))
                
                # Check RSI
                if "RSI信号" in entry_conditions:
                    if 'RSI' in data.columns:
                        if current['RSI'] < 30 and prev['RSI'] >= 30:
                            signals.append(Signal(
                                timestamp=current.name,
                                stock_code=stock_code,
                                signal_type=SignalType.BUY,
                                price=current['close'],
                                reason="RSI Oversold"
                            ))
                        elif current['RSI'] > 70 and prev['RSI'] <= 70:
                            signals.append(Signal(
                                timestamp=current.name,
                                stock_code=stock_code,
                                signal_type=SignalType.SELL,
                                price=current['close'],
                                reason="RSI Overbought"
                            ))
            
            return signals
        
        def calculate_position_size(self, signal, portfolio_value, current_positions):
            max_position_pct = self.config.get('max_position_pct', 0.2)
            max_positions = self.config.get('max_positions', 5)
            
            if current_positions >= max_positions:
                return 0
            
            position_value = portfolio_value * max_position_pct
            shares = int(position_value / signal.price)
            
            return shares
    
    # Run backtest
    with st.spinner("正在运行回测..."):
        # Create strategy instance
        strategy = ConfigurableStrategy(selected_strategy)
        
        # Create backtester
        backtester = SimpleBacktester(
            initial_capital=initial_capital,
            commission_rate=commission_rate,
            slippage=slippage,
            min_commission=min_commission
        )
        
        # Run backtest
        result = backtester.run(
            strategy,
            data,
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        )
        
        # Store result
        backtest_id = f"{strategy_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Ensure backtest_results exists and is a dict
        if 'backtest_results' not in st.session_state or st.session_state.backtest_results is None:
            st.session_state.backtest_results = {}
        
        # Double-check it's a dict before assignment
        if not isinstance(st.session_state.backtest_results, dict):
            st.session_state.backtest_results = {}
            
        st.session_state.backtest_results[backtest_id] = result
        
        st.success("回测完成!")

# Display results
if st.session_state.get('backtest_results') and isinstance(st.session_state.backtest_results, dict) and len(st.session_state.backtest_results) > 0:
    # Select which result to display
    if len(st.session_state.backtest_results) > 1:
        result_id = st.selectbox(
            "选择回测结果",
            list(st.session_state.backtest_results.keys()),
            index=len(st.session_state.backtest_results) - 1
        )
        result = st.session_state.backtest_results[result_id]
    else:
        result = list(st.session_state.backtest_results.values())[0]
    
    # Performance metrics
    st.markdown("### 📊 绩效指标")
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    metrics = result.metrics
    
    with col1:
        st.metric(
            "总收益率",
            f"{metrics.get('total_return_pct', 0):.2f}%",
            delta=f"{metrics.get('total_return_pct', 0):.2f}%"
        )
    
    with col2:
        st.metric(
            "胜率",
            f"{metrics.get('win_rate', 0)*100:.1f}%",
            delta=f"{metrics.get('winning_trades', 0)}/{metrics.get('total_trades', 0)}"
        )
    
    with col3:
        st.metric(
            "最大回撤",
            f"{metrics.get('max_drawdown_pct', 0):.2f}%"
        )
    
    with col4:
        st.metric(
            "夏普比率",
            f"{metrics.get('sharpe_ratio', 0):.2f}"
        )
    
    with col5:
        st.metric(
            "盈亏比",
            f"{metrics.get('profit_factor', 0):.2f}"
        )
    
    with col6:
        st.metric(
            "总交易次数",
            metrics.get('total_trades', 0)
        )
    
    # Equity curve
    st.markdown("### 📈 资金曲线")
    
    if not result.equity_curve.empty:
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.7, 0.3],
            subplot_titles=("资金曲线", "回撤")
        )
        
        # Equity curve
        fig.add_trace(
            go.Scatter(
                x=result.equity_curve.index,
                y=result.equity_curve['total_value'],
                mode='lines',
                name='总资产',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )
        
        # Add initial capital line
        fig.add_hline(
            y=initial_capital,
            line_dash="dash",
            line_color="gray",
            annotation_text="初始资金",
            row=1, col=1
        )
        
        # Drawdown
        fig.add_trace(
            go.Scatter(
                x=result.equity_curve.index,
                y=result.equity_curve['drawdown'] * 100,
                mode='lines',
                name='回撤',
                line=dict(color='red', width=1),
                fill='tozeroy',
                fillcolor='rgba(255, 0, 0, 0.1)'
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            height=600,
            showlegend=True,
            hovermode='x unified'
        )
        
        fig.update_xaxes(title_text="日期", row=2, col=1)
        fig.update_yaxes(title_text="资产价值", row=1, col=1)
        fig.update_yaxes(title_text="回撤 (%)", row=2, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Trade analysis
    st.markdown("### 📋 交易分析")
    
    tab1, tab2, tab3 = st.tabs(["交易列表", "收益分布", "持仓分析"])
    
    with tab1:
        if result.trades:
            # Create trades dataframe
            trades_data = []
            for trade in result.trades:
                trades_data.append({
                    "股票代码": trade.stock_code,
                    "买入日期": trade.entry_date.strftime('%Y-%m-%d'),
                    "卖出日期": trade.exit_date.strftime('%Y-%m-%d'),
                    "买入价格": f"¥{trade.entry_price:.2f}",
                    "卖出价格": f"¥{trade.exit_price:.2f}",
                    "数量": trade.shares,
                    "盈亏": f"¥{trade.pnl:.2f}",
                    "收益率": f"{trade.pnl_percent*100:.2f}%",
                    "持有天数": trade.hold_days,
                    "退出原因": trade.exit_reason
                })
            
            trades_df = pd.DataFrame(trades_data)
            
            # Add color coding
            def color_pnl(val):
                if '¥' in str(val):
                    amount = float(val.replace('¥', '').replace(',', ''))
                    color = 'green' if amount > 0 else 'red'
                elif '%' in str(val):
                    pct = float(val.replace('%', ''))
                    color = 'green' if pct > 0 else 'red'
                else:
                    return ''
                return f'color: {color}'
            
            styled_df = trades_df.style.applymap(
                color_pnl,
                subset=['盈亏', '收益率']
            )
            
            st.dataframe(styled_df, use_container_width=True)
            
            # Summary statistics
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 盈利交易")
                winning_trades = [t for t in result.trades if t.pnl > 0]
                if winning_trades:
                    st.write(f"- 交易次数: {len(winning_trades)}")
                    st.write(f"- 平均盈利: ¥{np.mean([t.pnl for t in winning_trades]):.2f}")
                    st.write(f"- 平均收益率: {np.mean([t.pnl_percent for t in winning_trades])*100:.2f}%")
                    st.write(f"- 最大盈利: ¥{max([t.pnl for t in winning_trades]):.2f}")
            
            with col2:
                st.markdown("#### 亏损交易")
                losing_trades = [t for t in result.trades if t.pnl <= 0]
                if losing_trades:
                    st.write(f"- 交易次数: {len(losing_trades)}")
                    st.write(f"- 平均亏损: ¥{np.mean([t.pnl for t in losing_trades]):.2f}")
                    st.write(f"- 平均亏损率: {np.mean([t.pnl_percent for t in losing_trades])*100:.2f}%")
                    st.write(f"- 最大亏损: ¥{min([t.pnl for t in losing_trades]):.2f}")
        else:
            st.info("没有交易记录")
    
    with tab2:
        if result.trades:
            # Profit distribution
            returns = [t.pnl_percent * 100 for t in result.trades]
            
            fig_dist = go.Figure()
            
            fig_dist.add_trace(
                go.Histogram(
                    x=returns,
                    nbinsx=20,
                    name='收益分布',
                    marker_color='blue',
                    opacity=0.7
                )
            )
            
            fig_dist.add_vline(
                x=0,
                line_dash="dash",
                line_color="red",
                annotation_text="0%"
            )
            
            fig_dist.add_vline(
                x=np.mean(returns),
                line_dash="dash",
                line_color="green",
                annotation_text=f"平均: {np.mean(returns):.1f}%"
            )
            
            fig_dist.update_layout(
                title="交易收益率分布",
                xaxis_title="收益率 (%)",
                yaxis_title="交易次数",
                height=400
            )
            
            st.plotly_chart(fig_dist, use_container_width=True)
            
            # Win/Loss pie chart
            col1, col2 = st.columns(2)
            
            with col1:
                fig_pie = go.Figure(data=[
                    go.Pie(
                        labels=['盈利', '亏损'],
                        values=[len([t for t in result.trades if t.pnl > 0]),
                               len([t for t in result.trades if t.pnl <= 0])],
                        hole=0.3,
                        marker_colors=['green', 'red']
                    )
                ])
                
                fig_pie.update_layout(
                    title="盈亏交易占比",
                    height=300
                )
                
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                # Holding period distribution
                hold_days = [t.hold_days for t in result.trades]
                
                fig_hold = go.Figure()
                fig_hold.add_trace(
                    go.Box(
                        y=hold_days,
                        name="持有天数",
                        boxpoints='all',
                        jitter=0.3,
                        pointpos=-1.8
                    )
                )
                
                fig_hold.update_layout(
                    title="持有期分布",
                    yaxis_title="天数",
                    height=300
                )
                
                st.plotly_chart(fig_hold, use_container_width=True)
    
    with tab3:
        if not result.positions.empty:
            # Position count over time
            position_counts = []
            for _, row in result.positions.iterrows():
                position_counts.append({
                    'date': row['date'],
                    'count': row['positions']
                })
            
            if position_counts:
                pos_df = pd.DataFrame(position_counts)
                
                fig_pos = go.Figure()
                fig_pos.add_trace(
                    go.Scatter(
                        x=pos_df['date'],
                        y=pos_df['count'],
                        mode='lines',
                        name='持仓数量',
                        line=dict(color='purple', width=2),
                        fill='tozeroy',
                        fillcolor='rgba(128, 0, 128, 0.1)'
                    )
                )
                
                fig_pos.update_layout(
                    title="持仓数量变化",
                    xaxis_title="日期",
                    yaxis_title="持仓数量",
                    height=400
                )
                
                st.plotly_chart(fig_pos, use_container_width=True)
                
                # Stock performance
                stock_performance = {}
                for trade in result.trades:
                    if trade.stock_code not in stock_performance:
                        stock_performance[trade.stock_code] = {
                            'trades': 0,
                            'total_pnl': 0,
                            'win_trades': 0
                        }
                    
                    stock_performance[trade.stock_code]['trades'] += 1
                    stock_performance[trade.stock_code]['total_pnl'] += trade.pnl
                    if trade.pnl > 0:
                        stock_performance[trade.stock_code]['win_trades'] += 1
                
                # Create stock performance table
                perf_data = []
                for stock, stats in stock_performance.items():
                    perf_data.append({
                        '股票代码': stock,
                        '交易次数': stats['trades'],
                        '总盈亏': f"¥{stats['total_pnl']:.2f}",
                        '胜率': f"{stats['win_trades']/stats['trades']*100:.1f}%"
                    })
                
                perf_df = pd.DataFrame(perf_data)
                perf_df = perf_df.sort_values('总盈亏', ascending=False)
                
                st.markdown("#### 个股表现")
                st.dataframe(perf_df, use_container_width=True)
        else:
            st.info("没有持仓记录")