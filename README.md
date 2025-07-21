# Technical Indicators Calculator Module

A Python module for calculating various technical indicators on financial market data, with support for both batch processing and real-time single-tick calculations.

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

## Installation

1. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Indicator Calculator Module

The `indicator_calculator.py` file contains the `IndicatorCalculator` class, which provides both static methods for calculating technical indicators on historical data and instance methods for real-time calculations.

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

- `calculate_all_indicators(data, indicator_periods={})` - Calculate all indicators for historical data
- `calculate_latest_tick_indicators(existing_data, new_row, indicator_periods)` - Calculate indicators for a single new tick

### Real-Time Indicator Calculation

The `calculate_latest_tick_indicators` method is designed for real-time trading applications where you need to efficiently update indicators as new market data arrives.

#### Key Features:

- **Efficient single-tick calculation** - Only calculates indicators for the latest data point
- **Configurable indicators** - Specify exactly which indicators and periods you need
- **Error handling** - Returns empty strings for insufficient data or calculation errors
- **Memory efficient** - Doesn't store unnecessary historical calculations

#### Real-Time Usage Example

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

# Your existing historical data
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

# Calculate indicators for the new tick only
result = calculator.calculate_latest_tick_indicators(
    existing_data=existing_data,
    new_row=new_row,
    indicator_periods=indicator_periods
)

print(result)
# Output includes basic OHLCV data plus calculated indicators
# Insufficient data indicators will be empty strings ("")
```

### Batch Processing Usage Example

```python
from indicator_calculator import IndicatorCalculator
import pandas as pd

# Create calculator instance
calculator = IndicatorCalculator()

# Load your OHLCV data
data = pd.read_csv('your_data.csv')

# Calculate individual indicators using static methods
rsi = IndicatorCalculator.calculate_rsi(data['close'], period=14)
macd_line, signal_line, histogram = IndicatorCalculator.calculate_macd(data['close'])

# Or calculate all indicators at once
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

result = calculator.calculate_all_indicators(data, indicator_periods)
```

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

The module expects OHLCV (Open, High, Low, Close, Volume) data in a pandas DataFrame with the following columns:

- `open` (or `Open`)
- `high` (or `High`)
- `low` (or `Low`)
- `close` (or `Close`)
- `volume` (or `Volume`) - Required for VWMA calculation

The module automatically handles both lowercase and uppercase column names.

### Output Format

- **Successful calculations**: Returns numerical values
- **Insufficient data**: Returns empty strings (`""`)
- **Calculation errors**: Returns empty strings (`""`)

This makes it easy to identify missing data and handle it appropriately in your trading applications.

### Performance Characteristics

- **Real-time calculations**: Optimized for single-tick updates
- **Memory efficient**: Minimal memory overhead for streaming data
- **Error resilient**: Individual indicator failures don't affect others
- **Configurable**: Only calculate the indicators you need
