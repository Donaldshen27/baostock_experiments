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