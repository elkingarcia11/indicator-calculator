# Technical Indicators Calculator Module

A Python module for calculating various technical indicators on financial market data, with support for both batch processing and real-time single-tick calculations. Features advanced support for different asset types including stocks, crypto, and options trading.

## Features

### Available Technical Indicators

- **RSI (Relative Strength Index)** - Momentum oscillator
- **Stochastic RSI** - Stochastic oscillator applied to RSI (%K and %D)
- **MACD** - Moving Average Convergence Divergence with signal line and histogram
- **ROC (Rate of Change)** - Price momentum indicator
- **ROC of ROC** - Second derivative of price change
- **EMA** - Exponential Moving Averages
- **VWMA** - Volume Weighted Moving Average
- **SMA** - Simple Moving Averages
- **Volatility** - Rolling standard deviation
- **Price Change** - Percentage change in price
- **Bollinger Bands** - Upper and lower bands based on standard deviation
- **Bollinger Bands Width** - Width of the Bollinger Bands
- **ATR (Average True Range)** - Volatility indicator

### Asset Type Support

- **Stocks & Crypto**: Uses `close` prices for calculations
- **Options Trading**: Uses `mark_price` for more accurate options pricing
- **Custom Assets**: Flexible column selection for any price source

## Installation

1. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Indicator Calculator Module

The `indicator_calculator.py` file contains the `IndicatorCalculator` class, which provides both static methods for calculating technical indicators on historical data and instance methods for real-time calculations with advanced function mapping architecture.

#### Available Static Methods

- `calculate_rsi(data, period=14)` - Relative Strength Index
- `calculate_stochastic_rsi(data, period=14, k_period=3, d_period=3)` - Stochastic RSI (%K and %D)
- `calculate_macd(data, fast_period=12, slow_period=26, signal_period=9)` - MACD (line, signal, histogram)
- `calculate_roc(data, period=10)` - Rate of Change
- `calculate_roc_of_roc(data, period=10)` - ROC of ROC (second derivative)
- `calculate_ema(data, period=20)` - Exponential Moving Average
- `calculate_vwma(data, period=20)` - Volume Weighted Moving Average
- `calculate_sma(data, period=20)` - Simple Moving Average
- `calculate_price_change(data)` - Percentage price change
- `calculate_volatility(data, period=20)` - Rolling volatility (standard deviation)
- `calculate_bollinger_bands(data, period=20, std_dev=2)` - Bollinger Bands (upper/lower bands)
- `calculate_bollinger_bands_width(data, period=20, std_dev=2)` - Bollinger Bands Width
- `calculate_atr(data, period=14)` - Average True Range

#### Available Instance Methods

- `calculate_all_indicators(data, indicator_periods={}, is_option=False)` - Calculate all indicators for historical data
- `calculate_latest_tick_indicators(existing_data, new_row, indicator_periods, is_option=False)` - Calculate indicators for a single new tick

### Real-Time Indicator Calculation

The `calculate_latest_tick_indicators` method is designed for real-time trading applications where you need to efficiently update indicators as new market data arrives.

#### Key Features:

- **Efficient single-tick calculation** - Only calculates indicators for the latest data point
- **Configurable indicators** - Specify exactly which indicators and periods you need
- **Error handling** - Returns empty strings for insufficient data or calculation errors
- **Memory efficient** - Minimal memory overhead for streaming data
- **Function mapping architecture** - Clean, maintainable code with isolated calculations
- **Asset type support** - Automatic price column selection for different asset types

#### Real-Time Usage Examples

##### For Stocks and Cryptocurrency:

```python
from indicator_calculator import IndicatorCalculator
import pandas as pd

# Create calculator instance
calculator = IndicatorCalculator()

# Define which indicators and periods you want
indicator_periods = {
    'rsi': 14,
    'ema': 20,
    'sma': 50,
    'macd_fast': 12,
    'macd_slow': 26,
    'macd_signal': 9,
    'bollinger_bands': 20,
    'bollinger_std_dev': 2,
    'atr': 14,
    'volatility': 20,
    'roc': 10,
    'stoch_rsi_k': 3,
    'stoch_rsi_d': 3,
    'vwma': 20,
    'price_change': 1
}

# Your existing historical data (uses 'close' prices)
existing_data = pd.DataFrame({
    'open': [100, 101, 102, 103],
    'high': [105, 106, 107, 108],
    'low': [99, 100, 101, 102],
    'close': [104, 105, 106, 107],
    'volume': [1000, 1100, 1200, 1300]
})

# New incoming tick/row
new_row = pd.Series({
    'open': 107,
    'high': 109,
    'low': 106,
    'close': 108,
    'volume': 1400
})

# Calculate indicators for the new tick (uses 'close' prices)
result = calculator.calculate_latest_tick_indicators(
    existing_data=existing_data,
    new_row=new_row,
    indicator_periods=indicator_periods,
    is_option=False  # Default: uses 'close' column
)

print(result)
# Output includes basic OHLCV data plus calculated indicators
```

##### For Options Trading:

```python
from indicator_calculator import IndicatorCalculator
import pandas as pd

# Create calculator instance
calculator = IndicatorCalculator()

# Options data with mark_price column
existing_options_data = pd.DataFrame({
    'open': [5.20, 5.25, 5.30, 5.35],
    'high': [5.40, 5.45, 5.50, 5.55],
    'low': [5.10, 5.15, 5.20, 5.25],
    'close': [5.30, 5.35, 5.40, 5.45],
    'mark_price': [5.32, 5.37, 5.42, 5.47],  # More accurate for options
    'volume': [500, 600, 700, 800]
})

# New options tick
new_options_row = pd.Series({
    'open': 5.45,
    'high': 5.60,
    'low': 5.40,
    'close': 5.55,
    'mark_price': 5.52,  # Used for calculations when is_option=True
    'volume': 900
})

# Calculate indicators using mark_price for options
result = calculator.calculate_latest_tick_indicators(
    existing_data=existing_options_data,
    new_row=new_options_row,
    indicator_periods=indicator_periods,
    is_option=True  # Uses 'mark_price' column instead of 'close'
)

print(result)
# All price-based indicators (RSI, EMA, SMA, MACD, etc.) calculated using mark_price
```

### Batch Processing Usage Examples

##### For Regular Assets:

```python
from indicator_calculator import IndicatorCalculator
import pandas as pd

# Create calculator instance
calculator = IndicatorCalculator()

# Load your OHLCV data
data = pd.read_csv('your_stock_data.csv')

# Calculate individual indicators using static methods
rsi = IndicatorCalculator.calculate_rsi(data['close'], period=14)
macd_line, signal_line, histogram = IndicatorCalculator.calculate_macd(data['close'])

# Or calculate all indicators at once (uses 'close' prices)
indicator_periods = {
    'rsi': 14,
    'macd_fast': 12,
    'macd_slow': 26,
    'macd_signal': 9,
    'ema': 20,
    'sma': 50,
    'bollinger_bands': 20,
    'bollinger_std_dev': 2,
    'atr': 14,
    'volatility': 20
}

result = calculator.calculate_all_indicators(data, indicator_periods, is_option=False)
```

##### For Options Data:

```python
# Load options data with mark_price column
options_data = pd.read_csv('your_options_data.csv')

# Calculate all indicators using mark_price for more accurate options analysis
result = calculator.calculate_all_indicators(
    data=options_data,
    indicator_periods=indicator_periods,
    is_option=True  # Uses 'mark_price' instead of 'close'
)
```

### Price Column Selection Logic

The module intelligently selects the appropriate price column based on asset type:

- **`is_option=False`** (default): Uses `'close'` column for stocks, crypto, forex
- **`is_option=True`**: Uses `'mark_price'` column for options trading

### Supported Indicator Periods Configuration

The `indicator_periods` dictionary supports the following keys:

- `'rsi'`: RSI period (default: 14)
- `'ema'`: EMA period (default: 20)
- `'sma'`: SMA period (default: 20)
- `'macd_fast'`: MACD fast EMA period (default: 12)
- `'macd_slow'`: MACD slow EMA period (default: 26)
- `'macd_signal'`: MACD signal line period (default: 9)
- `'bollinger_bands'`: Bollinger Bands SMA period (default: 20)
- `'bollinger_std_dev'`: Bollinger Bands standard deviation multiplier (default: 2)
- `'roc'`: Rate of Change period (default: 10)
- `'roc_of_roc'`: ROC of ROC period (default: 10)
- `'stoch_rsi_k'`: Stochastic RSI %K smoothing period (default: 3)
- `'stoch_rsi_d'`: Stochastic RSI %D smoothing period (default: 3)
- `'atr'`: Average True Range period (default: 14)
- `'vwma'`: Volume Weighted Moving Average period (default: 20)
- `'volatility'`: Volatility (rolling standard deviation) period (default: 20)
- `'price_change'`: Price change calculation (always 1 period)

### Data Requirements

#### For Stocks, Crypto, Forex:

```python
data = pd.DataFrame({
    'open': [...],    # Opening prices
    'high': [...],    # High prices
    'low': [...],     # Low prices
    'close': [...],   # Closing prices (used for calculations)
    'volume': [...]   # Volume data (required for VWMA)
})
```

#### For Options Trading:

```python
options_data = pd.DataFrame({
    'open': [...],       # Opening prices
    'high': [...],       # High prices
    'low': [...],        # Low prices
    'close': [...],      # Closing prices
    'mark_price': [...], # Mark prices (used for calculations when is_option=True)
    'volume': [...]      # Volume data
})
```

The module automatically handles both lowercase and uppercase column names.

### Output Format

- **Successful calculations**: Returns numerical values
- **Insufficient data**: Returns empty strings (`""`)
- **Calculation errors**: Returns empty strings (`""`)

This makes it easy to identify missing data and handle it appropriately in your trading applications.

### Architecture Benefits

- **Function Mapping**: Clean separation of concerns with individual calculation functions
- **Maintainable**: Easy to add new indicators without modifying existing code
- **Memory Efficient**: No temporary columns created, direct column references
- **Error Isolation**: Individual indicator failures don't affect others
- **Testable**: Each indicator calculation can be tested independently

### Performance Characteristics

- **Real-time calculations**: Optimized for single-tick updates
- **Memory efficient**: Minimal memory overhead for streaming data
- **Error resilient**: Individual indicator failures don't affect others
- **Configurable**: Only calculate the indicators you need
- **Asset type aware**: Automatic price source selection
- **No data duplication**: Direct column access without copying data

### Use Cases

- **Algorithmic Trading**: Real-time indicator updates for trading bots
- **Options Strategy Analysis**: Accurate pricing using mark prices
- **Portfolio Management**: Multi-asset indicator calculations
- **Technical Analysis**: Comprehensive indicator suite for market analysis
- **Backtesting**: Efficient historical indicator calculation
- **Risk Management**: Volatility and momentum indicators for risk assessment
