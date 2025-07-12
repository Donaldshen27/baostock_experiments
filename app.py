#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
A-Share Stock Strategy Platform
Main application entry point
"""

import streamlit as st
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Page configuration
st.set_page_config(
    page_title="A股量化策略平台",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/yourusername/stock-platform',
        'Report a bug': "https://github.com/yourusername/stock-platform/issues",
        'About': "# A-Share Stock Strategy Platform\n基于Baostock数据的量化交易平台"
    }
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1.2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = True
    st.session_state.selected_stocks = ['sh.600000']
    st.session_state.current_strategy = None
    st.session_state.backtest_results = None

# Main header
st.markdown('<h1 class="main-header">A股量化策略平台</h1>', unsafe_allow_html=True)

# Welcome message
st.markdown("""
### 👋 欢迎使用A股量化策略平台

本平台基于 **Baostock** 数据源，提供完整的量化交易解决方案：

#### 🎯 核心功能
- **📊 市场总览**: 实时查看市场数据和技术指标
- **📈 策略构建**: 创建和测试自定义交易策略
- **🧪 回测系统**: 历史数据回测，评估策略表现
- **💼 投资组合**: 多股票组合分析和风险管理
- **📉 绩效分析**: 详细的策略表现报告

#### 🚀 快速开始
1. 👈 从左侧菜单选择功能模块
2. 📊 在"市场总览"查看实时数据
3. 📈 在"策略构建"创建您的第一个策略
4. 🧪 使用"回测系统"验证策略效果

#### 📌 使用提示
- 所有数据来自 Baostock，支持日线、分钟线等多种频率
- 策略回测考虑了A股特有的T+1交易规则
- 支持自定义技术指标和策略参数

#### 🔗 导航
请使用左侧菜单栏访问各个功能模块：
- **市场总览** - 查看股票行情和技术分析
- **策略构建** - 创建量化交易策略
- **回测系统** - 测试策略历史表现
- **投资组合** - 管理多股票组合
- **绩效分析** - 查看详细业绩报告

---

*💡 提示: 点击左上角 `>` 展开侧边栏菜单*
""")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>数据来源: Baostock | 仅供学习研究使用，不构成投资建议</p>
    <p>© 2024 A股量化策略平台</p>
</div>
""", unsafe_allow_html=True)