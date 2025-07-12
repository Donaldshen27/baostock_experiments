# A-Share Quantitative Trading Platform

A comprehensive quantitative trading platform for Chinese A-share market, built with Streamlit and powered by Baostock data.

## 🚀 Overview

This platform provides a complete solution for developing, testing, and analyzing quantitative trading strategies for the Chinese stock market. It features an intuitive web interface, comprehensive backtesting capabilities, and real-time market analysis tools.

## ✨ Key Features

### 📊 Market Overview
- Real-time stock data visualization with interactive charts
- 30+ technical indicators (MA, RSI, MACD, Bollinger Bands, etc.)
- Multi-timeframe analysis (5min to monthly)
- Volume analysis and market breadth indicators
- **NEW: Chip Distribution (筹码分布) Analysis**
  - Traditional chip distribution histogram (筹码峰图)
  - Profit/trapped chips visualization
  - Average cost and concentration metrics
  - Real-time chip flow analysis

### 📈 Strategy Builder
- Visual strategy creation interface
- Pre-built strategy templates (Golden Cross, RSI Momentum, etc.)
- Custom indicator combinations
- Automatic code generation

### 🧪 Backtesting System
- Historical strategy testing with A-share specific rules (T+1)
- Comprehensive performance metrics
- Risk analysis and drawdown tracking
- Multi-stock portfolio backtesting

### 💼 Portfolio Management
- Position tracking and risk management
- Real-time P&L monitoring
- Sector allocation analysis
- Performance attribution

### 📉 Analytics Dashboard
- Sharpe ratio, win rate, profit factor
- Trade analysis and distribution
- Equity curve visualization
- Detailed performance reports

## Installation

1. Create and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install baostock akshare pandas matplotlib numpy streamlit plotly
```

## 🚀 Quick Start

### Launch the Platform
```bash
# Activate virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Start the application
streamlit run app.py
```

The platform will automatically open in your browser at `http://localhost:8501`

### Platform Navigation

1. **Home** - Platform overview and quick start guide
2. **📊 市场总览** - Real-time market data and technical analysis
3. **📈 策略构建** - Create and configure trading strategies
4. **🧪 回测系统** - Test strategies on historical data
5. **💼 投资组合** - Portfolio management and analysis (coming soon)
6. **📉 绩效分析** - Detailed performance reports (coming soon)

## 📁 Project Structure

```
stock/
├── app.py                      # Main application entry
├── pages/                      # Streamlit pages
│   ├── 1_📊_市场总览.py       # Market overview
│   ├── 2_📈_策略构建.py       # Strategy builder
│   ├── 3_🧪_回测系统.py       # Backtesting system
│   └── ...                    # More pages coming
├── core/                      # Core modules
│   ├── data_manager.py        # Baostock data management
│   └── indicators.py          # Technical indicators
├── strategies/                # Trading strategies
│   ├── base.py               # Base strategy class
│   ├── golden_cross.py       # MA crossover strategy
│   └── rsi_momentum.py       # RSI momentum strategy
├── backtest/                  # Backtesting engine
│   └── engine.py             # Simple backtester
├── analytics/                 # Performance analytics
├── utils/                     # Utility functions
└── data/                      # Data storage
    └── cache/                 # Cached market data
```

## 🛠️ Core Components

### Data Manager (`core/data_manager.py`)
- Automatic Baostock login/logout management
- Data caching to reduce API calls
- Batch data fetching for multiple stocks
- Support for all Baostock data types
- **Enhanced with AkShare integration:**
  - Chip distribution data (`get_chip_distribution`)
  - Stock info with outstanding shares (`get_stock_info_ak`)
  - Real-time quotes (`get_realtime_data`)
  - Combined historical + chip data (`get_stock_data_with_chip`)

### Strategy Framework (`strategies/base.py`)
- Abstract base class for all strategies
- Signal generation interface
- Position sizing methods
- Risk management integration

### Backtesting Engine (`backtest/engine.py`)
- A-share specific features (T+1, price limits)
- Commission and slippage modeling
- Performance metrics calculation
- Trade tracking and analysis

### Technical Indicators (`core/indicators.py`)
- 30+ indicators implemented
- Optimized for pandas DataFrames
- Trading signal generation
- Indicator combination support

### Chip Distribution (`core/chip_distribution.py`)
- Calculate chip distribution from historical data
- Triangular distribution algorithm for realistic modeling
- Decay factor based on turnover rate
- Statistical analysis (concentration, profit ratio)
- Support for custom lookback periods and price bins

## Data Sources

### Baostock
- Historical price and volume data
- Financial statements (profit, balance, cash flow)
- Index components (HS300, SZ50, ZZ500)
- Trading calendar

### AkShare
- Chip distribution metrics (筹码分布)
- Real-time quotes and market data
- Stock info with outstanding shares
- Market breadth indicators

## Data Dimensions

### Market Data Fields
- **Price**: open, high, low, close, volume, amount
- **Metrics**: pctChg, turn, peTTM, pbMRQ, isST
- **Timeframes**: 5min, 15min, 30min, 60min, daily, weekly, monthly
- **Chip Distribution**: profit_ratio, avg_cost, concentration, cost ranges

### Financial Data Categories
1. **Profitability**: ROE, profit margins, EPS
2. **Operations**: Turnover ratios
3. **Growth**: YoY growth rates
4. **Solvency**: Debt and liquidity ratios
5. **Cash Flow**: Operating cash flow metrics
6. **DuPont Analysis**: ROE decomposition

## Stock Code Format

- Shanghai: `sh.` prefix (e.g., `sh.600000`)
- Shenzhen: `sz.` prefix (e.g., `sz.000002`)

## Generated Files

- `sh600000_daily_data.csv` - Sample daily price data
- `stock_technical_indicators.csv` - Price data with indicators
- `stock_technical_analysis.png` - Technical analysis chart
- `baostock_all_indicators.csv` - Complete indicator dataset
- `streamlit_demo.py` - Interactive web dashboard

## Technical Indicators

### Built-in (6 indicators)
- pctChg (涨跌幅)
- turn (换手率)
- peTTM, psTTM, pcfNcfTTM, pbMRQ

### Custom Calculated (30+ indicators)
- **Trend**: SMA, EMA, MACD, ADX, Ichimoku
- **Momentum**: RSI, KDJ, Williams %R, CCI, MFI
- **Volatility**: Bollinger Bands, ATR
- **Volume**: OBV, Volume MA

## 📝 Creating Custom Strategies

### Quick Example
```python
from strategies.base import BaseStrategy, Signal, SignalType
from core.indicators import TechnicalIndicators

class MyStrategy(BaseStrategy):
    def generate_signals(self, data, stock_code):
        signals = []
        
        # Your strategy logic here
        if data['RSI'].iloc[-1] < 30:
            signals.append(Signal(
                timestamp=data.index[-1],
                stock_code=stock_code,
                signal_type=SignalType.BUY,
                price=data['close'].iloc[-1],
                reason="RSI Oversold"
            ))
        
        return signals
    
    def calculate_position_size(self, signal, portfolio_value, current_positions):
        # Position sizing logic
        return int(portfolio_value * 0.1 / signal.price)
```

### Using Your Strategy
1. Create strategy in the Strategy Builder UI
2. Configure parameters and indicators
3. Run backtest to evaluate performance
4. Export code for further customization

## ⚠️ Important Notes

### A-Share Market Specifics
- **T+1 Trading**: Stocks bought today can only be sold tomorrow
- **Price Limits**: ±10% daily limit for most stocks
- **Trading Hours**: 9:30-11:30, 13:00-15:00 (Beijing time)
- **Minimum Commission**: Usually ¥5 per trade

### Platform Usage
1. Always activate virtual environment before running
2. Data is cached for 24 hours to reduce API calls
3. Backtest results consider realistic trading constraints
4. All timestamps are in Beijing timezone

## License

This project uses the following libraries:
- Baostock - BSD License
- AkShare - MIT License

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## 🎯 Usage Examples

### 1. Analyzing a Single Stock
```python
from core.data_manager import DataManager
from core.indicators import TechnicalIndicators

dm = DataManager()
df = dm.get_stock_data('sh.600519', '2024-01-01', '2024-12-31')
df = TechnicalIndicators.add_all_indicators(df)
print(df[['close', 'RSI', 'MACD']].tail())
```

### 2. Backtesting a Strategy
```python
from backtest.engine import SimpleBacktester
from strategies.golden_cross import GoldenCrossStrategy

strategy = GoldenCrossStrategy({'short_period': 5, 'long_period': 20})
backtester = SimpleBacktester(initial_capital=100000)

# Load data for multiple stocks
data = dm.batch_get_stock_data(['sh.600519', 'sz.000002'], '2024-01-01', '2024-12-31')

# Run backtest
result = backtester.run(strategy, data)
print(f"Total Return: {result.metrics['total_return_pct']:.2f}%")
print(f"Sharpe Ratio: {result.metrics['sharpe_ratio']:.2f}")
```

### 3. Finding Oversold Stocks
```python
# Get all HS300 stocks
hs300 = dm.get_index_components('hs300')
oversold_stocks = []

for stock in hs300['code']:
    df = dm.get_stock_data(stock, start_date, end_date)
    df = TechnicalIndicators.add_rsi(df)
    
    if df['RSI'].iloc[-1] < 30:
        oversold_stocks.append(stock)

print(f"Found {len(oversold_stocks)} oversold stocks")
```

## 🚧 Roadmap

- [ ] Real-time data integration
- [ ] More pre-built strategies
- [ ] Machine learning models
- [ ] Risk management tools
- [ ] Portfolio optimization
- [ ] Automated trading interface
- [ ] Mobile app support

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📚 Resources

- [Baostock Documentation](https://pypi.org/project/baostock/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Chinese Stock Market Info](http://www.sse.com.cn/)

## ⚖️ Disclaimer

This platform is for educational and research purposes only. It does not constitute investment advice. Always do your own research and consider consulting with a qualified financial advisor before making investment decisions.

---

**Built with ❤️ for A-Share quantitative traders**