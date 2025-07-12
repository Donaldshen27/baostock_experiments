# Project Structure

```
stock/
├── app.py                          # Main Streamlit application entry point
├── requirements.txt                # Python dependencies
├── README.md                       # Project overview and usage guide
├── ROADMAP.md                      # Development roadmap
├── CONTRIBUTING.md                 # Contribution guidelines
├── CHANGELOG.md                    # Version history
├── .gitignore                      # Git ignore patterns
│
├── pages/                          # Streamlit pages (multi-page app)
│   ├── 1_📊_市场总览.py           # Market overview page
│   ├── 2_📈_策略构建.py           # Strategy builder page
│   └── 3_🧪_回测系统.py           # Backtesting system page
│
├── core/                           # Core functionality modules
│   ├── __init__.py
│   ├── data_manager.py             # Baostock data fetching and caching
│   └── indicators.py               # Technical indicators calculations
│
├── strategies/                     # Trading strategy modules
│   ├── __init__.py
│   ├── base.py                     # Abstract base strategy class
│   ├── golden_cross.py             # Golden cross MA strategy
│   └── rsi_momentum.py             # RSI momentum strategy
│
├── backtest/                       # Backtesting engine
│   ├── __init__.py
│   └── engine.py                   # Simple backtesting implementation
│
├── analytics/                      # Performance analytics (future)
│   └── __init__.py
│
├── utils/                          # Utility functions (future)
│   └── __init__.py
│
├── data/                           # Data storage
│   └── cache/                      # Cached market data (.pkl files)
│
└── docs/                           # Documentation
    └── PROJECT_STRUCTURE.md        # This file
```

## Module Descriptions

### `app.py`
- Main entry point for the Streamlit application
- Handles page configuration and navigation
- Initializes session state

### `pages/`
Contains individual pages for the multi-page Streamlit app:
- **市场总览**: Real-time market data visualization and technical analysis
- **策略构建**: Visual strategy creation and configuration
- **回测系统**: Historical backtesting and performance analysis

### `core/`
Core functionality modules:
- **data_manager.py**: Handles all Baostock data operations including caching
- **indicators.py**: Implements 30+ technical indicators

### `strategies/`
Trading strategy implementations:
- **base.py**: Abstract base class defining strategy interface
- **golden_cross.py**: Moving average crossover strategy
- **rsi_momentum.py**: RSI-based momentum strategy

### `backtest/`
Backtesting functionality:
- **engine.py**: Simulates trading with A-share specific rules

## Data Flow

1. **Data Fetching**: `DataManager` → Baostock API → Cache
2. **Strategy Development**: UI → Strategy Config → Code Generation
3. **Backtesting**: Strategy + Data → Backtester → Results
4. **Visualization**: Results → Plotly Charts → UI

## Key Design Patterns

### Strategy Pattern
All trading strategies inherit from `BaseStrategy`, implementing:
- `generate_signals()`: Signal generation logic
- `calculate_position_size()`: Position sizing logic

### Singleton Pattern
`DataManager` uses caching to avoid redundant API calls

### Factory Pattern
Strategy creation in the UI generates strategy instances dynamically

## Adding New Features

### New Strategy
1. Create new file in `strategies/`
2. Inherit from `BaseStrategy`
3. Implement required methods
4. Add to strategy templates in UI

### New Indicator
1. Add static method to `TechnicalIndicators` class
2. Follow naming convention: `add_indicator_name()`
3. Update `add_all_indicators()` if needed

### New Page
1. Create file in `pages/` with emoji prefix
2. Follow Streamlit page naming convention
3. Import required modules
4. Add to navigation in `app.py`