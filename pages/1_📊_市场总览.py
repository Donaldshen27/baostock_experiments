#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Market Overview Page
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from core.data_manager import DataManager
from core.indicators import TechnicalIndicators
from core.chip_distribution import ChipDistribution

st.set_page_config(page_title="市场总览", page_icon="📊", layout="wide")

# Initialize data manager
@st.cache_resource
def get_data_manager():
    return DataManager()

dm = get_data_manager()

# Header
st.title("📊 市场总览")
st.markdown("实时查看股票行情、技术指标和市场动态")

# Top controls
col1, col2, col3, col4 = st.columns([2, 2, 2, 1])

with col1:
    # Popular stocks
    popular_stocks = {
        "浦发银行": "sh.600000",
        "万科A": "sz.000002",
        "平安银行": "sz.000001",
        "贵州茅台": "sh.600519",
        "中国平安": "sh.601318",
        "招商银行": "sh.600036",
        "五粮液": "sz.000858",
        "比亚迪": "sz.002594",
        "宁德时代": "sz.300750",
        "隆基绿能": "sh.601012"
    }
    
    selected_name = st.selectbox(
        "选择股票",
        list(popular_stocks.keys()),
        index=0
    )
    stock_code = popular_stocks[selected_name]

with col2:
    # Date range
    date_options = {
        "最近1月": 30,
        "最近3月": 90,
        "最近6月": 180,
        "最近1年": 365,
        "今年以来": -1,
        "自定义": 0
    }
    
    date_selection = st.selectbox("时间范围", list(date_options.keys()), index=1)
    
    if date_selection == "自定义":
        with col3:
            custom_start = st.date_input("开始日期", datetime.now() - timedelta(days=90))
            custom_end = st.date_input("结束日期", datetime.now())
            start_date = custom_start.strftime('%Y-%m-%d')
            end_date = custom_end.strftime('%Y-%m-%d')
    elif date_selection == "今年以来":
        start_date = f"{datetime.now().year}-01-01"
        end_date = datetime.now().strftime('%Y-%m-%d')
    else:
        days = date_options[date_selection]
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')

with col3:
    if date_selection != "自定义":
        freq_options = ["日线", "周线", "月线", "60分钟", "30分钟", "15分钟", "5分钟"]
        frequency = st.selectbox("数据频率", freq_options, index=0)
        freq_map = {
            "日线": "d", "周线": "w", "月线": "m",
            "60分钟": "60", "30分钟": "30", "15分钟": "15", "5分钟": "5"
        }
        freq_code = freq_map[frequency]

with col4:
    st.write("")  # Spacer
    refresh_button = st.button("🔄 刷新数据", use_container_width=True)

# Load data
with st.spinner('加载数据中...'):
    df = dm.get_stock_data(stock_code, start_date, end_date, frequency=freq_code)
    
    if df.empty:
        st.error("无法获取数据，请检查股票代码或日期范围")
        st.stop()
    
    # Calculate indicators
    df = TechnicalIndicators.add_all_indicators(df)

# Display metrics
st.markdown("### 📈 关键指标")
col1, col2, col3, col4, col5, col6 = st.columns(6)

latest = df.iloc[-1]
prev = df.iloc[-2] if len(df) > 1 else latest

with col1:
    st.metric(
        "最新价",
        f"¥{latest['close']:.2f}",
        f"{latest['pctChg']:.2f}%" if 'pctChg' in latest else None,
        delta_color="normal" if latest.get('pctChg', 0) >= 0 else "inverse"
    )

with col2:
    volume_change = ((latest['volume'] - prev['volume']) / prev['volume'] * 100) if prev['volume'] > 0 else 0
    st.metric(
        "成交量",
        f"{latest['volume']/1e8:.2f}亿",
        f"{volume_change:.1f}%"
    )

with col3:
    st.metric(
        "换手率",
        f"{latest.get('turn', 0):.2f}%"
    )

with col4:
    if 'RSI' in df.columns and not pd.isna(latest['RSI']):
        rsi_value = latest['RSI']
        rsi_status = "超买" if rsi_value > 70 else "超卖" if rsi_value < 30 else "正常"
        st.metric(
            "RSI(14)",
            f"{rsi_value:.1f}",
            rsi_status
        )

with col5:
    if 'MACD' in df.columns and not pd.isna(latest['MACD']):
        macd_signal = "买入" if latest['MACD'] > latest['MACD_Signal'] else "卖出"
        st.metric(
            "MACD",
            f"{latest['MACD']:.3f}",
            macd_signal
        )

with col6:
    # Calculate trend
    if len(df) >= 20:
        trend = "上涨" if latest['close'] > df['close'].iloc[-20] else "下跌"
        trend_pct = ((latest['close'] - df['close'].iloc[-20]) / df['close'].iloc[-20] * 100)
        st.metric(
            "20日趋势",
            trend,
            f"{trend_pct:.1f}%"
        )

# Main chart
st.markdown("### 📊 价格走势图")

# Chart options
chart_col1, chart_col2, chart_col3, chart_col4 = st.columns([1, 1, 1, 3])
with chart_col1:
    show_volume = st.checkbox("显示成交量", value=True)
with chart_col2:
    show_ma = st.checkbox("移动平均线", value=True)
with chart_col3:
    show_bb = st.checkbox("布林带", value=False)
with chart_col4:
    chart_type = st.radio(
        "图表类型",
        ["K线图", "收盘价线图", "OHLC图"],
        horizontal=True
    )

# Create chart
fig = make_subplots(
    rows=2 if show_volume else 1,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.03,
    row_heights=[0.7, 0.3] if show_volume else [1],
    subplot_titles=(None, "成交量") if show_volume else None
)

# Add price data based on chart type
if chart_type == "K线图":
    fig.add_trace(
        go.Candlestick(
            x=df.index if isinstance(df.index, pd.DatetimeIndex) else df['date'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name="K线",
            increasing_line_color='red',
            decreasing_line_color='green'
        ),
        row=1, col=1
    )
elif chart_type == "收盘价线图":
    fig.add_trace(
        go.Scatter(
            x=df.index if isinstance(df.index, pd.DatetimeIndex) else df['date'],
            y=df['close'],
            mode='lines',
            name='收盘价',
            line=dict(color='blue', width=2)
        ),
        row=1, col=1
    )
else:  # OHLC
    fig.add_trace(
        go.Ohlc(
            x=df.index if isinstance(df.index, pd.DatetimeIndex) else df['date'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name="OHLC"
        ),
        row=1, col=1
    )

# Add moving averages
if show_ma and 'MA5' in df.columns:
    for ma, color in [('MA5', 'blue'), ('MA10', 'orange'), ('MA20', 'purple'), ('MA60', 'brown')]:
        if ma in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index if isinstance(df.index, pd.DatetimeIndex) else df['date'],
                    y=df[ma],
                    mode='lines',
                    name=ma,
                    line=dict(color=color, width=1)
                ),
                row=1, col=1
            )

# Add Bollinger Bands
if show_bb and 'BB_Upper' in df.columns:
    fig.add_trace(
        go.Scatter(
            x=df.index if isinstance(df.index, pd.DatetimeIndex) else df['date'],
            y=df['BB_Upper'],
            mode='lines',
            name='BB上轨',
            line=dict(color='gray', width=1, dash='dash')
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=df.index if isinstance(df.index, pd.DatetimeIndex) else df['date'],
            y=df['BB_Lower'],
            mode='lines',
            name='BB下轨',
            line=dict(color='gray', width=1, dash='dash'),
            fill='tonexty',
            fillcolor='rgba(128, 128, 128, 0.1)'
        ),
        row=1, col=1
    )

# Add volume
if show_volume:
    colors = ['red' if row['close'] >= row['open'] else 'green' 
              for idx, row in df.iterrows()]
    fig.add_trace(
        go.Bar(
            x=df.index if isinstance(df.index, pd.DatetimeIndex) else df['date'],
            y=df['volume'],
            name='成交量',
            marker_color=colors,
            showlegend=False
        ),
        row=2, col=1
    )

# Update layout
fig.update_layout(
    title=f"{selected_name} ({stock_code}) - {frequency}数据",
    height=600,
    xaxis_rangeslider_visible=False,
    hovermode='x unified',
    legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01
    )
)

fig.update_xaxes(title_text="时间", row=2 if show_volume else 1, col=1)
fig.update_yaxes(title_text="价格", row=1, col=1)
if show_volume:
    fig.update_yaxes(title_text="成交量", row=2, col=1)

st.plotly_chart(fig, use_container_width=True)

# Technical indicators section
st.markdown("### 📉 技术指标")

# Create tabs for different indicator types
ind_tab1, ind_tab2, ind_tab3 = st.tabs(["动量指标", "趋势指标", "波动率指标"])

with ind_tab1:
    # RSI Chart
    if 'RSI' in df.columns:
        fig_rsi = go.Figure()
        fig_rsi.add_trace(
            go.Scatter(
                x=df.index if isinstance(df.index, pd.DatetimeIndex) else df['date'],
                y=df['RSI'],
                mode='lines',
                name='RSI',
                line=dict(color='purple', width=2)
            )
        )
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="超买")
        fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="超卖")
        fig_rsi.update_layout(height=300, title="RSI (14)", yaxis=dict(range=[0, 100]))
        st.plotly_chart(fig_rsi, use_container_width=True)

with ind_tab2:
    # MACD Chart
    if 'MACD' in df.columns:
        fig_macd = go.Figure()
        fig_macd.add_trace(
            go.Scatter(
                x=df.index if isinstance(df.index, pd.DatetimeIndex) else df['date'],
                y=df['MACD'],
                mode='lines',
                name='MACD',
                line=dict(color='blue', width=2)
            )
        )
        fig_macd.add_trace(
            go.Scatter(
                x=df.index if isinstance(df.index, pd.DatetimeIndex) else df['date'],
                y=df['MACD_Signal'],
                mode='lines',
                name='Signal',
                line=dict(color='red', width=1)
            )
        )
        fig_macd.add_trace(
            go.Bar(
                x=df.index if isinstance(df.index, pd.DatetimeIndex) else df['date'],
                y=df['MACD_Histogram'],
                name='Histogram',
                marker_color='gray'
            )
        )
        fig_macd.update_layout(height=300, title="MACD")
        st.plotly_chart(fig_macd, use_container_width=True)

with ind_tab3:
    # ATR Chart
    if 'ATR' in df.columns:
        fig_atr = go.Figure()
        fig_atr.add_trace(
            go.Scatter(
                x=df.index if isinstance(df.index, pd.DatetimeIndex) else df['date'],
                y=df['ATR'],
                mode='lines',
                name='ATR',
                line=dict(color='orange', width=2)
            )
        )
        fig_atr.update_layout(height=300, title="Average True Range (14)")
        st.plotly_chart(fig_atr, use_container_width=True)

# Chip Distribution Section
st.markdown("### 🎯 筹码分布")

# Traditional Chip Distribution Histogram (筹码峰)
st.markdown("#### 📊 筹码分布图")

# Calculate chip distribution using historical data
current_price = float(df['close'].iloc[-1])
chip_dist_df = ChipDistribution.calculate_chip_distribution(
    df.reset_index(), 
    current_price,
    lookback_days=min(120, len(df)),
    price_bins=80
)

# Create the traditional chip distribution histogram
fig_chip_hist = go.Figure()

# Add trapped chips (blue - 套牢筹码)
fig_chip_hist.add_trace(
    go.Bar(
        x=chip_dist_df['trapped_chips_pct'],
        y=chip_dist_df['price'],
        orientation='h',
        name='套牢筹码',
        marker_color='lightblue',
        width=1.5,
        hovertemplate='价格: %{y:.2f}<br>筹码: %{x:.2f}%<extra></extra>'
    )
)

# Add profit chips (red - 获利筹码)
fig_chip_hist.add_trace(
    go.Bar(
        x=chip_dist_df['profit_chips_pct'],
        y=chip_dist_df['price'],
        orientation='h',
        name='获利筹码',
        marker_color='lightcoral',
        width=1.5,
        hovertemplate='价格: %{y:.2f}<br>筹码: %{x:.2f}%<extra></extra>'
    )
)

# Calculate and add average cost line
chip_stats = ChipDistribution.calculate_cost_distribution_stats(chip_dist_df)
avg_cost = chip_stats['avg_cost']

# Add average cost line (yellow)
fig_chip_hist.add_hline(
    y=avg_cost,
    line_dash="dash",
    line_color="gold",
    line_width=2,
    annotation_text=f"平均成本: {avg_cost:.2f}",
    annotation_position="right"
)

# Add current price line (green)
fig_chip_hist.add_hline(
    y=current_price,
    line_dash="solid",
    line_color="green",
    line_width=2,
    annotation_text=f"当前价: {current_price:.2f}",
    annotation_position="right"
)

# Update layout
fig_chip_hist.update_layout(
    title="筹码分布直方图",
    xaxis_title="筹码占比 (%)",
    yaxis_title="价格 (元)",
    height=500,
    barmode='overlay',
    showlegend=True,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ),
    xaxis=dict(range=[0, max(chip_dist_df['chips_pct'].max() * 1.2, 5)]),
    yaxis=dict(
        range=[chip_dist_df['price'].min() * 0.98, chip_dist_df['price'].max() * 1.02]
    )
)

st.plotly_chart(fig_chip_hist, use_container_width=True)

# Display chip distribution statistics
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("获利筹码", f"{chip_stats['profit_ratio']:.1f}%", 
              delta=f"{chip_stats['profit_ratio']-50:.1f}%" if chip_stats['profit_ratio'] > 50 else None)
with col2:
    st.metric("套牢筹码", f"{chip_stats['trapped_ratio']:.1f}%")
with col3:
    st.metric("平均成本", f"¥{chip_stats['avg_cost']:.2f}",
              delta=f"{(current_price/chip_stats['avg_cost']-1)*100:.1f}%")
with col4:
    st.metric("90%集中度", f"{chip_stats['concentration_90']:.1f}%",
              help="90%的筹码集中在的价格区间占比")

# Get AkShare chip distribution data for additional analysis
st.markdown("#### 📈 筹码分布趋势")
chip_df = dm.get_chip_distribution(stock_code)

if not chip_df.empty:
    # Create two columns for chip distribution visualization
    chip_col1, chip_col2 = st.columns(2)
    
    with chip_col1:
        # Profit ratio and average cost chart
        fig_chip = go.Figure()
        
        # Add profit ratio line
        fig_chip.add_trace(
            go.Scatter(
                x=chip_df['日期'],
                y=chip_df['获利比例'],
                mode='lines',
                name='获利比例 (%)',
                line=dict(color='green', width=2),
                yaxis='y'
            )
        )
        
        # Add average cost line
        fig_chip.add_trace(
            go.Scatter(
                x=chip_df['日期'],
                y=chip_df['平均成本'],
                mode='lines',
                name='平均成本',
                line=dict(color='red', width=2),
                yaxis='y2'
            )
        )
        
        fig_chip.update_layout(
            title="获利比例与平均成本",
            height=400,
            yaxis=dict(
                title="获利比例 (%)",
                side="left",
                showgrid=True
            ),
            yaxis2=dict(
                title="平均成本 (元)",
                side="right",
                overlaying="y",
                showgrid=False
            ),
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_chip, use_container_width=True)
    
    with chip_col2:
        # Concentration chart
        fig_conc = go.Figure()
        
        # Add 90% concentration
        fig_conc.add_trace(
            go.Scatter(
                x=chip_df['日期'],
                y=chip_df['90集中度'],
                mode='lines',
                name='90%集中度',
                line=dict(color='blue', width=2)
            )
        )
        
        # Add 70% concentration
        fig_conc.add_trace(
            go.Scatter(
                x=chip_df['日期'],
                y=chip_df['70集中度'],
                mode='lines',
                name='70%集中度',
                line=dict(color='orange', width=2)
            )
        )
        
        fig_conc.update_layout(
            title="筹码集中度",
            height=400,
            yaxis=dict(title="集中度 (%)"),
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_conc, use_container_width=True)
    
    # Cost distribution range
    st.markdown("#### 📊 成本分布区间")
    latest_chip = chip_df.iloc[-1]
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("70%成本区间", 
                 f"{latest_chip['70成本-低']:.2f} - {latest_chip['70成本-高']:.2f}")
    with col2:
        st.metric("90%成本区间", 
                 f"{latest_chip['90成本-低']:.2f} - {latest_chip['90成本-高']:.2f}")
    with col3:
        st.metric("当前获利比例", f"{latest_chip['获利比例']:.1f}%")
    with col4:
        st.metric("平均成本", f"{latest_chip['平均成本']:.2f}")
    
    # Chip distribution analysis
    with st.expander("💡 筹码分布解读"):
        st.markdown(f"""
        **筹码分布分析 ({latest_chip['日期']})**
        
        - **获利比例**: {latest_chip['获利比例']:.1f}% 的持仓处于盈利状态
        - **平均成本**: 市场平均持仓成本为 {latest_chip['平均成本']:.2f} 元
        - **70%集中度**: {latest_chip['70集中度']:.1f}% - 表示70%的筹码集中在 {latest_chip['70成本-低']:.2f}-{latest_chip['70成本-高']:.2f} 元区间
        - **90%集中度**: {latest_chip['90集中度']:.1f}% - 表示90%的筹码集中在 {latest_chip['90成本-低']:.2f}-{latest_chip['90成本-高']:.2f} 元区间
        
        **解读提示**:
        - 集中度越小，说明筹码越集中，可能存在主力控盘
        - 获利比例高时需注意获利回吐压力
        - 平均成本可作为重要支撑/压力位参考
        """)
else:
    st.info("暂无筹码分布数据，可能是因为该股票不支持或数据暂时不可用")

# Data table
with st.expander("📋 查看原始数据"):
    # Select columns to display
    display_columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'pctChg', 'turn']
    available_columns = [col for col in display_columns if col in df.columns]
    
    st.dataframe(
        df[available_columns].tail(50).sort_index(ascending=False),
        use_container_width=True
    )

# Save selected stock to session state
st.session_state.selected_stocks = [stock_code]