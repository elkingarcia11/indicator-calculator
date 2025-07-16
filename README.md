# Technical Indicators Module

A Python module for calculating various technical indicators on financial market data.

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

The `indicator_calculator.py` file contains the `IndicatorCalculator` class, which provides a comprehensive set of static methods for calculating technical indicators.

#### Available Methods

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
- `calculate_all_indicators(data, indicator_periods={})` - Calculate all indicators at once

#### Usage Example

```python
from indicator_calculator import IndicatorCalculator
import pandas as pd

# Load your OHLCV data
data = pd.read_csv('your_data.csv')

# Calculate individual indicators
rsi = IndicatorCalculator.calculate_rsi(data['Close'], period=14)
macd_line, signal_line, histogram = IndicatorCalculator.calculate_macd(data['Close'])

# Or calculate all indicators at once
indicator_periods = {
    'RSI': 14,
    'MACD_Fast': 12,
    'MACD_Slow': 26,
    'MACD_Signal': 9,
    'EMA': 20,
    'SMA': 50,
    'Bollinger_Bands': 20,
    'Bollinger_Bands_Width': 20,
    'ATR': 14,
    'Volatility': 20
}
result = IndicatorCalculator().calculate_all_indicators(data, indicator_periods)
```

### Data Requirements

The module expects OHLCV (Open, High, Low, Close, Volume) data in a pandas DataFrame with the following columns:

- `open` (or `Open`)
- `high` (or `High`)
- `low` (or `Low`)
- `close` (or `Close`)
- `volume` (or `Volume`) - Required for VWMA calculation

The module automatically handles both lowercase and uppercase column names.
