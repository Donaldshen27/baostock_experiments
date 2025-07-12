# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2024-07-12

### Added
- Initial release of A-Share Quantitative Trading Platform
- Multi-page Streamlit application with Chinese UI
- Market Overview page with real-time data visualization
- Strategy Builder with visual interface and code generation
- Backtesting System with A-share specific rules (T+1, price limits)
- Core modules:
  - DataManager for efficient Baostock data handling
  - TechnicalIndicators with 30+ indicators
  - BaseStrategy framework for custom strategies
  - SimpleBacktester engine
- Example strategies:
  - Golden Cross (MA crossover)
  - RSI Momentum
- Data caching system to reduce API calls
- Comprehensive documentation and examples

### Features
- **Data Sources**: Integration with Baostock for Chinese A-share market data
- **Technical Analysis**: Built-in indicators including MA, EMA, RSI, MACD, Bollinger Bands, KDJ, ATR, etc.
- **Strategy Development**: Template-based and custom strategy creation
- **Backtesting**: Historical simulation with realistic constraints
- **Performance Metrics**: Sharpe ratio, maximum drawdown, win rate, profit factor
- **Visualization**: Interactive charts with Plotly

### Technical Stack
- Python 3.8+
- Streamlit for web interface
- Pandas for data manipulation
- Plotly for interactive charts
- Baostock for market data

### Known Issues
- Limited to historical data (no real-time streaming)
- Backtesting assumes perfect order execution
- No options or futures support

## [Unreleased]

### Planned
- Portfolio management page
- Performance analytics dashboard
- More strategy templates
- Real-time data integration
- Machine learning models
- Automated trading interface

---

## Version History

### Pre-release Development
- 2024-07-11: Project initialization
- 2024-07-11: Basic Baostock integration
- 2024-07-12: Streamlit UI development
- 2024-07-12: Strategy framework implementation
- 2024-07-12: Backtesting engine creation
- 2024-07-12: First public release