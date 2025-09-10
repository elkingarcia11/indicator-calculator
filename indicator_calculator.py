import pandas as pd
import numpy as np

class IndicatorCalculator:
    """Class to calculate various technical indicators"""
    
    def __init__(self):
        """Initialize the indicator calculator with function mappings"""
        self.indicator_functions = {
            'rsi': self._calculate_single_rsi,
            'ema': self._calculate_single_ema,
            'ema_fast': self._calculate_single_ema,
            'ema_slow': self._calculate_single_ema,
            'sma': self._calculate_single_sma,
            'macd_line': self._calculate_single_macd_line,
            'macd_signal': self._calculate_single_macd_signal,
            'bollinger_upper': self._calculate_single_bollinger_upper,
            'bollinger_lower': self._calculate_single_bollinger_lower,
            'bollinger_bands_width': self._calculate_single_bollinger_width,
            'roc': self._calculate_single_roc,
            'roc_fast': self._calculate_single_roc,
            'roc_slow': self._calculate_single_roc,
            'roc_of_roc': self._calculate_single_roc_of_roc,
            'rsi_roc': self._calculate_single_rsi_roc,
            'stoch_rsi_k': self._calculate_single_stoch_rsi_k,
            'stoch_rsi_d': self._calculate_single_stoch_rsi_d,
            'atr': self._calculate_single_atr,
            'vwma': self._calculate_single_vwma,
            'price_change': self._calculate_single_price_change,
            'gaussian_smoothing': self._calculate_single_gaussian_smoothing,
            'gaussian_smoothing_price': self._calculate_single_gaussian_smoothing_price,
            'gaussian_smoothing_ema_fast': self._calculate_single_gaussian_smoothing_ema_fast,
            'gaussian_smoothing_ema_slow': self._calculate_single_gaussian_smoothing_ema_slow,
            'ema_fast': self._calculate_single_ema_fast,
            'ema_slow': self._calculate_single_ema_slow,
            'rsi_roc': self._calculate_single_rsi_roc
        }
    
    def _get_volume_column(self, data: pd.DataFrame) -> str:
        """
        Get the appropriate volume column to use.
        Prefers 'tick_volume' if available, otherwise falls back to 'volume'.
        
        Args:
            data: DataFrame containing volume data
            
        Returns:
            str: Column name to use for volume ('tick_volume' or 'volume')
        """
        if 'tick_volume' in data.columns:
            return 'tick_volume'
        elif 'volume' in data.columns:
            return 'volume'
        else:
            raise ValueError("Neither 'tick_volume' nor 'volume' column found in data")
    
    @staticmethod
    def calculate_rsi(data: pd.DataFrame, period: int = 14, price_column: str = 'close') -> pd.Series:
        """
        Calculate Relative Strength Index (RSI)

        Args:
            data: Price series (typically Close prices)
            period: Lookback period for RSI calculation  
            price_column: Column name to use for price data
            
        Returns:
            RSI values as pandas Series
        """
        if 'mark_price' in data.columns:
            price_column = 'mark_price'
        else:
            price_column = 'close'
        
        # Calculate the difference between the current and previous price
        delta = data[price_column].diff()
        # Calculate the average gain
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        # Calculate the average loss
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        # Calculate the relative strength
        rs = gain / loss
        # Calculate the relative strength index
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def calculate_stochastic_rsi(data: pd.DataFrame, period: int = 14, k_period: int = 3, d_period: int = 3, price_column: str = 'close') -> tuple:
        """
        Calculate Stochastic RSI
        %K starts at Row [Period]
        %D starts at Row [K Period + D Period - 1]
        
        Args:
            data: Price series
            period: RSI period (for lookback)
            k_period: %K smoothing period
            d_period: %D smoothing period
            
        Returns:
            Tuple of (%K, %D) values
        """
        if len(data) == 0:
            empty_series = pd.Series(index=data.index, dtype=float)
            return empty_series, empty_series
        
        # Calculate the RSI (starts immediately)
        rsi = IndicatorCalculator.calculate_rsi(data, period, price_column)
        
        # Stochastic %K: starts at Row [Period] - needs full lookback period
        stoch_rsi_raw = (rsi - rsi.rolling(window=period).min()) / (rsi.rolling(window=period).max() - rsi.rolling(window=period).min())
        # Apply %K smoothing
        stoch_rsi_k = stoch_rsi_raw.rolling(window=k_period).mean() * 100
        
        # Stochastic %D: starts at Row [K Period + D Period - 1]
        # If %K=14, %D=3: Row 16 (14+3-1)
        stoch_rsi_d = stoch_rsi_k.rolling(window=d_period).mean()
        
        return stoch_rsi_k, stoch_rsi_d
    
    @staticmethod
    def calculate_macd(data: pd.DataFrame, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9, price_column: str = 'close') -> tuple:
        """
        Calculate MACD (Moving Average Convergence Divergence)
        Both MACD line and Signal line start from Row 1 (platform behavior)
        
        Args:
            data: Price series
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal line period
            price_column: Column name to use for price data
        Returns:
            Tuple of (MACD line, Signal line)
        """
        if len(data) == 0:
            empty_series = pd.Series(index=data.index, dtype=float)
            return empty_series, empty_series
        
        # Platform behavior: Both EMAs start immediately
        ema_fast = IndicatorCalculator.calculate_ema(data, fast_period, price_column)
        ema_slow = IndicatorCalculator.calculate_ema(data, slow_period, price_column)
        
        # MACD line: difference calculated immediately from Row 1
        macd_line = ema_fast - ema_slow
        
        # Signal line: EMA of MACD line, starts immediately from Row 1
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        
        return macd_line, signal_line
    
    @staticmethod
    def calculate_roc(data: pd.DataFrame, period: int, price_column: str = 'close') -> pd.Series:
        """
        Calculate Rate of Change (ROC)
        Starts at Row [Period + 1] - needs current price vs price N periods ago
        
        Args:
            data: Price series DataFrame
            period: Lookback period
            price_column: Column name to use for calculation
            
        Returns:
            ROC values as pandas Series (starts at Row [Period + 1])
        """
        if len(data) == 0:
            return pd.Series(index=data.index, dtype=float)
        
        # ROC starts at Row [Period + 1] - naturally handled by shift()
        # ROC 9 starts at Row 10, ROC 14 starts at Row 15
        return ((data[price_column] - data[price_column].shift(period)) / data[price_column].shift(period)) * 100
    
    @staticmethod
    def calculate_roc_of_roc(data: pd.DataFrame, roc_of_roc_period: int = 10) -> pd.Series:
        """
        Calculate ROC of ROC using the existing 'roc' column.
        Starts at Row [2×Period + 1] - needs ROC to be established first
        """
        if len(data) == 0 or 'roc' not in data.columns:
            return pd.Series(index=data.index, dtype=float)
        
        # ROC of ROC starts at Row [2×Period + 1]
        # First ROC appears at row [Period + 1], then need another [Period] periods
        # For ROC of ROC 18: Row 37 (18 + 18 + 1)
        roc = data['roc']
        roc_of_roc = ((roc - roc.shift(roc_of_roc_period)) / roc.shift(roc_of_roc_period)) * 100
        return roc_of_roc
    
    @staticmethod
    def calculate_ema(data: pd.DataFrame, period: int, price_column: str = 'close') -> pd.Series:
        """
        Calculate Exponential Moving Average (EMA)
        Starts immediately from Row 1 using first price as seed (platform behavior)
        
        Args:
            data: Price series DataFrame
            period: EMA period
            price_column: Column name to use for calculation
            
        Returns:
            EMA values as pandas Series (starts from Row 1)
        """
        if len(data) == 0:
            return pd.Series(index=data.index, dtype=float)
        
        # Platform behavior: EMA starts immediately from Row 1
        # Use pandas EMA with adjust=False for immediate start
        return data[price_column].ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def calculate_vwma(data: pd.DataFrame, period: int, price_column: str = 'close') -> pd.Series:
        """
        Calculate Volume Weighted Moving Average (VWMA)
        Starts at Row [Period] - needs full window of price/volume data
        
        Args:
            data: DataFrame with price and volume columns
            period: VWMA period
            price_column: Column name to use for calculation
            
        Returns:
            VWMA values as pandas Series (starts at Row [Period])
        """
        if len(data) == 0:
            return pd.Series(index=data.index, dtype=float)
        
        # Get the appropriate volume column (tick_volume if available, otherwise volume)
        try:
            if 'tick_volume' in data.columns:
                volume_column = 'tick_volume'
            elif 'volume' in data.columns:
                volume_column = 'volume'
            else:
                return pd.Series(index=data.index, dtype=float)
        except Exception:
            return pd.Series(index=data.index, dtype=float)
        
        # VWMA needs full window - starts at Row [Period] 
        # Calculate VWMA using rolling windows (naturally handles the timing)
        price_volume = data[price_column] * data[volume_column]
        vwma = price_volume.rolling(window=period).sum() / data[volume_column].rolling(window=period).sum()
        
        return vwma

    @staticmethod
    def calculate_sma(data: pd.DataFrame, period: int, price_column: str = 'close') -> pd.Series:
        """
        Calculate Simple Moving Average (SMA)
        
        Args:
            data: Price series DataFrame
            period: SMA period
            price_column: Column name to use for calculation
            
        Returns:
            SMA values as pandas Series (NaN until sufficient data available)
        """
        return data[price_column].rolling(window=period).mean()
    
    @staticmethod
    def calculate_price_change(data: pd.DataFrame, price_column: str = 'close') -> pd.Series:
        """
        Calculate Price Change
        """
        # Calculate the Price Change
        return data[price_column].pct_change()
    
    @staticmethod
    def calculate_gaussian_smoothing(data: pd.DataFrame, period: int, column: str = 'close') -> pd.Series:
        """
        Calculate Gaussian Smoothing (Weighted Moving Average)
        
        Gaussian smoothing applies weights that follow a normal distribution,
        giving more weight to recent data points and less weight to older ones.
        Can be applied to any column (price, indicators, etc.).
        
        Args:
            data: DataFrame with the column to smooth
            period: Period for Gaussian smoothing calculation
            column: Column name to apply smoothing to (default: 'close')
            
        Returns:
            Gaussian smoothed values as pandas Series
        """
        if len(data) == 0:
            return pd.Series(index=data.index, dtype=float)
        
        # Create Gaussian weights (normal distribution)
        # Center the weights around the middle of the period
        center = (period - 1) / 2
        sigma = period / 6  # Standard deviation (covers ~99.7% of data)
        
        # Generate weights for the period
        weights = []
        for i in range(period):
            # Calculate Gaussian weight
            weight = np.exp(-0.5 * ((i - center) / sigma) ** 2)
            weights.append(weight)
        
        # Normalize weights to sum to 1
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        # Apply weighted rolling average
        gaussian_smoothed = data[column].rolling(window=period).apply(
            lambda x: np.sum(x * weights), raw=True
        )
        
        return gaussian_smoothed
    
    @staticmethod
    def calculate_bollinger_bands(data: pd.DataFrame, period: int, std_dev: int, price_column: str = 'close') -> tuple:
        """
        Calculate Bollinger Bands
        
        Args:
            data: Price series DataFrame
            period: Period for moving average and standard deviation
            std_dev: Number of standard deviations for bands
            price_column: Column name to use for calculation
            
        Returns:
            Tuple of (upper_band, lower_band) as pandas Series
        """
        sma = data[price_column].rolling(window=period).mean()
        std = data[price_column].rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return upper_band, lower_band
    
    @staticmethod
    def calculate_bollinger_bands_width(data: pd.DataFrame, period: int, std_dev: int, price_column: str = 'close') -> pd.Series:
        """
        Calculate Bollinger Bands Width
        
        Args:
            data: Price series DataFrame
            period: Period for standard deviation calculation
            std_dev: Number of standard deviations for width
            price_column: Column name to use for calculation
            
        Returns:
            Bollinger Bands width as pandas Series
        """
        return data[price_column].rolling(window=period).std() * std_dev * 2  # Width is 2 * std_dev * std
        
    @staticmethod
    def calculate_atr(data: pd.DataFrame, period: int, price_column: str = 'close') -> pd.Series:
        """
        Calculate Average True Range (ATR)
        
        Args:
            data: DataFrame with high, low, close columns
            period: Period for ATR calculation
            price_column: Not used for ATR (included for consistency)
            
        Returns:
            ATR values as pandas Series (NaN until sufficient data available)
        """
        # ATR requires high, low, close columns
        if not all(col in data.columns for col in ['high_price', 'low_price', 'close_price']):
            return data[price_column].rolling(window=period).std()
        
        # Calculate True Range
        high_low = data['high_price'] - data['low_price']
        high_close = abs(data['high_price'] - data['close_price'].shift())
        low_close = abs(data['low_price'] - data['close_price'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        
        # Calculate ATR as moving average of True Range
        return true_range.rolling(window=period).mean()
    
    def calculate_all_indicators(self, data: pd.DataFrame, indicator_periods: dict, price_column: str = 'close') -> pd.DataFrame:
        """
        Calculate requested technical indicators for the given data.
        Only calculates indicators that are explicitly requested in indicator_periods.
        
        Args:
            data: OHLCV DataFrame
            indicator_periods: Dictionary with indicator names and their periods (required)
            price_column: Specific column name to use for price-based indicators (overrides is_option)
            
        Returns:
            DataFrame with calculated indicators (NaN values until sufficient periods available)
        """
        # Make a copy to avoid modifying original data
        result = data.copy()
        
        # RSI
        if 'rsi' in indicator_periods:
            result['rsi'] = self.calculate_rsi(result, indicator_periods['rsi'], price_column)
            
        # Stochastic RSI (requires all three parameters)
        if all(key in indicator_periods for key in ['stoch_rsi_period', 'stoch_rsi_k', 'stoch_rsi_d']):
            stoch_rsi_k, stoch_rsi_d = self.calculate_stochastic_rsi(
                result, 
                indicator_periods['stoch_rsi_period'], 
                indicator_periods['stoch_rsi_k'], 
                indicator_periods['stoch_rsi_d'], 
                price_column
            )
            result['stoch_rsi_k'] = stoch_rsi_k
            result['stoch_rsi_d'] = stoch_rsi_d
        
        # MACD (requires all three parameters)
        if all(key in indicator_periods for key in ['macd_fast', 'macd_slow', 'macd_signal']):
            macd_line, signal_line = self.calculate_macd(
                result, 
                indicator_periods['macd_fast'], 
                indicator_periods['macd_slow'], 
                indicator_periods['macd_signal'], 
                price_column
            )
            result['macd_line'] = macd_line
            result['macd_signal'] = signal_line
        
        # Rate of Change
        if 'roc' in indicator_periods:
            result['roc'] = self.calculate_roc(result, indicator_periods['roc'], price_column)
            
        # Rate of Change of Rate of Change (requires ROC to be calculated first)
        if 'roc_of_roc' in indicator_periods and 'roc' in result.columns:
            result['roc_of_roc'] = self.calculate_roc_of_roc(result, indicator_periods['roc_of_roc'])
        
        # Exponential Moving Average
        if 'ema' in indicator_periods:
            result['ema'] = self.calculate_ema(result, indicator_periods['ema'], price_column)
        
        # EMA variants
        if 'ema_fast' in indicator_periods:
            result['ema_fast'] = self.calculate_ema(result, indicator_periods['ema_fast'], price_column)
        
        if 'ema_slow' in indicator_periods:
            result['ema_slow'] = self.calculate_ema(result, indicator_periods['ema_slow'], price_column)
        
        # Volume Weighted Moving Average
        if 'vwma' in indicator_periods:
            result['vwma'] = self.calculate_vwma(result, indicator_periods['vwma'], price_column)
        
        # VWMA variants
        if 'vwma_fast' in indicator_periods:
            result['vwma_fast'] = self.calculate_vwma(result, indicator_periods['vwma_fast'], price_column)
            
        if 'vwma_slow' in indicator_periods:
            result['vwma_slow'] = self.calculate_vwma(result, indicator_periods['vwma_slow'], price_column)
        
        # Simple Moving Average
        if 'sma' in indicator_periods:
            result['sma'] = self.calculate_sma(result, indicator_periods['sma'], price_column)
        
        # Price Change
        if 'price_change' in indicator_periods:
            result['price_change'] = self.calculate_price_change(result, price_column)
        
        # Average True Range
        if 'atr' in indicator_periods:
            result['atr'] = self.calculate_atr(result, indicator_periods['atr'], price_column)
        
        # Bollinger Bands (requires period and std_dev)
        if 'bollinger_period' in indicator_periods and 'bollinger_std' in indicator_periods:
            upper, lower = self.calculate_bollinger_bands(
                result, 
                indicator_periods['bollinger_period'], 
                indicator_periods['bollinger_std'], 
                price_column
            )
            result['bollinger_upper'] = upper
            result['bollinger_lower'] = lower
            
        # Bollinger Bands Width
        if 'bollinger_period' in indicator_periods and 'bollinger_std' in indicator_periods:
            result['bollinger_bands_width'] = self.calculate_bollinger_bands_width(
                result, 
                indicator_periods['bollinger_period'], 
                indicator_periods['bollinger_std'], 
                price_column
            )
        
        return result
            
    def calculate_latest_tick_indicators(self, existing_data: pd.DataFrame, new_row: pd.Series, indicator_periods: dict) -> pd.Series:
        """
        Calculate technical indicators for a new tick/row based on existing historical data
        
        Args:
            existing_data: Historical OHLCV DataFrame with existing indicators
            new_row: New data row (Series) with OHLCV data
            indicator_periods: Dictionary with indicator names and their periods
            is_option: If True, uses 'mark_price' instead of 'close' for price-based indicators
            
        Returns:
            Series with calculated indicator values for the new tick (empty strings for insufficient data)
        """
        try:
    
            # Create a temporary dataframe with existing data + new row
            temp_data = existing_data.copy()
            temp_data = pd.concat([temp_data, new_row.to_frame().T], ignore_index=True)
            
            if 'mark_price' in temp_data.columns:
                price_column = 'mark_price'
            else:
                price_column = 'close'
            
            # Initialize result series with the new row's basic data
            result = new_row.copy()
            last_idx = len(temp_data) - 1
            
            # Map indicator_periods keys to their corresponding output indicators
            indicator_mapping = {
                'rsi': ['rsi'],
                'ema': ['ema'],
                'sma': ['sma'],
                'ema_fast': ['ema_fast'],
                'ema_slow': ['ema_slow'],
                'macd_fast': ['macd_line', 'macd_signal'],
                'macd_slow': ['macd_line', 'macd_signal'],
                'macd_signal': ['macd_line', 'macd_signal'],
                'bollinger_bands': ['bollinger_upper', 'bollinger_lower', 'bollinger_bands_width'],
                'roc': ['roc'],
                'roc_fast': ['roc_fast'],
                'roc_slow': ['roc_slow'],
                'roc_of_roc': ['roc_of_roc'],
                'stoch_rsi_period': ['stoch_rsi_period'],  # Add the period configuration
                'stoch_rsi_k': ['stoch_rsi_k'],
                'stoch_rsi_d': ['stoch_rsi_d'],
                'atr': ['atr'],
                'vwma': ['vwma'],
                'price_change': ['price_change'],
                'gaussian_smoothing_price': ['gaussian_smoothing_price'],
                'gaussian_smoothing_ema': ['gaussian_smoothing_ema'],
                'gaussian_smoothing_vwma': ['gaussian_smoothing_vwma'],
                'gaussian_smoothing_ema_fast': ['gaussian_smoothing_ema_fast'],
                'gaussian_smoothing_ema_slow': ['gaussian_smoothing_ema_slow']
            }
            
            # Calculate each indicator using the function mapping
            calculated_indicators = set()
            
            for period_key, period_value in indicator_periods.items():
                # Skip if this indicator group was already calculated
                if period_key in ['macd_slow', 'macd_signal'] and 'macd_line' in calculated_indicators:
                    continue
                if period_key == 'stoch_rsi_d' and 'stoch_rsi_k' in calculated_indicators:
                    continue
                
                try:
                    if period_key == 'rsi' and 'rsi' in self.indicator_functions:
                        print(f"🔧 DEBUG: Processing RSI with period={period_value}, last_idx={last_idx}, data_len={len(temp_data)}")
                        rsi_value = self._calculate_single_rsi(temp_data, period_value, last_idx, price_column)
                        result['rsi'] = rsi_value
                        print(f"🔧 DEBUG: RSI result = {rsi_value}")
                        calculated_indicators.add('rsi')
                        
                    elif period_key == 'ema' and 'ema' in self.indicator_functions:
                        result['ema'] = self._calculate_single_ema(temp_data, period_value, last_idx, price_column)
                        # Add EMA to temp_data so dependent indicators can access it
                        temp_data['ema'] = self.calculate_ema(temp_data, period_value, price_column)
                        calculated_indicators.add('ema')
                        
                    elif period_key == 'sma' and 'sma' in self.indicator_functions:
                        result['sma'] = self._calculate_single_sma(temp_data, period_value, last_idx, price_column)
                        calculated_indicators.add('sma')
                        
                    elif period_key == 'ema_fast' and 'ema_fast' in self.indicator_functions:
                        print(f"🔧 DEBUG: Processing ema_fast with period={period_value}, last_idx={last_idx}")
                        ema_fast_value = self._calculate_single_ema(temp_data, period_value, last_idx, price_column)
                        result['ema_fast'] = ema_fast_value
                        print(f"🔧 DEBUG: ema_fast calculated value: {ema_fast_value}")
                        # Add ema_fast to temp_data so dependent indicators can access it
                        # Calculate the full EMA series for the entire temp_data
                        ema_fast_series = self.calculate_ema(temp_data, period_value, price_column)
                        temp_data['ema_fast'] = ema_fast_series
                        print(f"🔧 DEBUG: Added full ema_fast series to temp_data with {ema_fast_series.notna().sum()} non-null values")
                        calculated_indicators.add('ema_fast')
                        
                    elif period_key == 'ema_slow' and 'ema_slow' in self.indicator_functions:
                        print(f"🔧 DEBUG: Processing ema_slow with period={period_value}, last_idx={last_idx}")
                        ema_slow_value = self._calculate_single_ema(temp_data, period_value, last_idx, price_column)
                        result['ema_slow'] = ema_slow_value
                        print(f"🔧 DEBUG: ema_slow calculated value: {ema_slow_value}")
                        # Add ema_slow to temp_data so dependent indicators can access it
                        # Calculate the full EMA series for the entire temp_data
                        ema_slow_series = self.calculate_ema(temp_data, period_value, price_column)
                        temp_data['ema_slow'] = ema_slow_series
                        print(f"🔧 DEBUG: Added full ema_slow series to temp_data with {ema_slow_series.notna().sum()} non-null values")
                        calculated_indicators.add('ema_slow')
                        
                    elif period_key in ['macd_fast', 'macd_slow', 'macd_signal'] and 'macd_line' not in calculated_indicators:
                        # Calculate all MACD components at once
                        if all(k in indicator_periods for k in ['macd_fast', 'macd_slow', 'macd_signal']):
                            result['macd_line'] = self._calculate_single_macd_line(temp_data, indicator_periods, last_idx, price_column)
                            result['macd_signal'] = self._calculate_single_macd_signal(temp_data, indicator_periods, last_idx, price_column)
                            calculated_indicators.update(['macd_line', 'macd_signal'])
                        
                    elif period_key == 'bollinger_bands':
                        result['bollinger_upper'] = self._calculate_single_bollinger_upper(temp_data, indicator_periods, last_idx, price_column)
                        result['bollinger_lower'] = self._calculate_single_bollinger_lower(temp_data, indicator_periods, last_idx, price_column)
                        result['bollinger_bands_width'] = self._calculate_single_bollinger_width(temp_data, indicator_periods, last_idx, price_column)
                        calculated_indicators.update(['bollinger_upper', 'bollinger_lower', 'bollinger_bands_width'])
                        
                    elif period_key == 'roc' and 'roc' in self.indicator_functions:
                        result['roc'] = self._calculate_single_roc(temp_data, period_value, last_idx, price_column)
                        calculated_indicators.add('roc')
                        
                    elif period_key == 'roc_fast' and 'roc_fast' in self.indicator_functions:
                        result['roc_fast'] = self._calculate_single_roc(temp_data, period_value, last_idx, price_column)
                        calculated_indicators.add('roc_fast')
                        
                    elif period_key == 'roc_slow' and 'roc_slow' in self.indicator_functions:
                        result['roc_slow'] = self._calculate_single_roc(temp_data, period_value, last_idx, price_column)
                        calculated_indicators.add('roc_slow')
                        
                    elif period_key == 'roc_of_roc' and 'roc_of_roc' in self.indicator_functions:
                        # Calculate ROC of ROC using the ROC function on ROC data
                        roc_period = indicator_periods.get('roc', 11)
                        roc_of_roc_period = period_value
                        
                        # First calculate ROC for the entire temp_data
                        roc_series = self.calculate_roc(temp_data, roc_period, price_column)
                        
                        # Then calculate ROC of ROC using ROC function on the ROC series
                        roc_df = pd.DataFrame({'roc': roc_series})
                        roc_of_roc_series = self.calculate_roc(roc_df, roc_of_roc_period, 'roc')
                        
                        # Get the latest value
                        result['roc_of_roc'] = roc_of_roc_series.iloc[last_idx] if not pd.isna(roc_of_roc_series.iloc[last_idx]) else float('nan')
                        calculated_indicators.add('roc_of_roc')
                    elif period_key == 'rsi_roc' and 'rsi_roc' in self.indicator_functions:
                        result['rsi_roc'] = self._calculate_single_rsi_roc(temp_data, period_value, last_idx, price_column)
                        calculated_indicators.add('rsi_roc')
                        
                    elif period_key in ['stoch_rsi_k', 'stoch_rsi_d'] and 'stoch_rsi_k' not in calculated_indicators:
                        # Calculate both Stochastic RSI components if both periods are provided
                        if 'stoch_rsi_k' in indicator_periods and 'stoch_rsi_d' in indicator_periods:
                            result['stoch_rsi_k'] = self._calculate_single_stoch_rsi_k(temp_data, indicator_periods, last_idx, price_column)
                            result['stoch_rsi_d'] = self._calculate_single_stoch_rsi_d(temp_data, indicator_periods, last_idx, price_column)
                            calculated_indicators.update(['stoch_rsi_k', 'stoch_rsi_d'])
                        
                    elif period_key == 'atr' and 'atr' in self.indicator_functions:
                        result['atr'] = self._calculate_single_atr(temp_data, period_value, last_idx, price_column)
                        calculated_indicators.add('atr')
                        
                    elif period_key == 'vwma' and 'vwma' in self.indicator_functions:
                        result['vwma'] = self._calculate_single_vwma(temp_data, period_value, last_idx, price_column)
                        # Add VWMA to temp_data so dependent indicators can access it
                        temp_data['vwma'] = self.calculate_vwma(temp_data, period_value, price_column)
                        calculated_indicators.add('vwma')
                    elif period_key == 'price_change' and 'price_change' in self.indicator_functions:
                        result['price_change'] = self._calculate_single_price_change(temp_data, last_idx, price_column)
                        calculated_indicators.add('price_change')
                    elif period_key == 'gaussian_smoothing_ema_fast' and 'gaussian_smoothing_ema_fast' in self.indicator_functions:
                        print(f"🔧 DEBUG: Processing gaussian_smoothing_ema_fast with period={period_value}, last_idx={last_idx}")
                        print(f"🔧 DEBUG: temp_data columns: {temp_data.columns.tolist()}")
                        print(f"🔧 DEBUG: temp_data has 'ema_fast' column: {'ema_fast' in temp_data.columns}")
                        if 'ema_fast' in temp_data.columns:
                            ema_fast_values = temp_data['ema_fast'].tail(period_value)
                            print(f"🔧 DEBUG: Last {period_value} ema_fast values: {ema_fast_values.tolist()}")
                            print(f"🔧 DEBUG: ema_fast non-null values: {ema_fast_values.notna().sum()}/{len(ema_fast_values)}")
                        result['gaussian_smoothing_ema_fast'] = self._calculate_single_gaussian_smoothing(temp_data, period_value, last_idx, 'ema_fast')
                        print(f"🔧 DEBUG: gaussian_smoothing_ema_fast result = {result['gaussian_smoothing_ema_fast']}")
                        calculated_indicators.add('gaussian_smoothing_ema_fast')
                    elif period_key == 'gaussian_smoothing_ema_slow' and 'gaussian_smoothing_ema_slow' in self.indicator_functions:
                        print(f"🔧 DEBUG: Processing gaussian_smoothing_ema_slow with period={period_value}, last_idx={last_idx}")
                        print(f"🔧 DEBUG: temp_data columns: {temp_data.columns.tolist()}")
                        print(f"🔧 DEBUG: temp_data has 'ema_slow' column: {'ema_slow' in temp_data.columns}")
                        if 'ema_slow' in temp_data.columns:
                            ema_slow_values = temp_data['ema_slow'].tail(period_value)
                            print(f"🔧 DEBUG: Last {period_value} ema_slow values: {ema_slow_values.tolist()}")
                            print(f"🔧 DEBUG: ema_slow non-null values: {ema_slow_values.notna().sum()}/{len(ema_slow_values)}")
                        result['gaussian_smoothing_ema_slow'] = self._calculate_single_gaussian_smoothing(temp_data, period_value, last_idx, 'ema_slow')
                        print(f"🔧 DEBUG: gaussian_smoothing_ema_slow result = {result['gaussian_smoothing_ema_slow']}")
                        calculated_indicators.add('gaussian_smoothing_ema_slow')

                        
                except Exception as e:  
                    # Set error values for this specific indicator
                    if period_key in indicator_mapping:
                        for indicator_name in indicator_mapping[period_key]:
                            result[indicator_name] = float('nan')
                    
            return result
            
        except Exception as e:
            print(f"Error calculating latest tick indicators: {str(e)}")
            # Return the new row with ",," for all requested indicator columns
            result = new_row.copy()
            
            # Create mapping to determine output columns from input periods
            for key in indicator_periods.keys():
                if key in ['macd_fast', 'macd_slow', 'macd_signal']:
                    result['macd_line'] = float('nan')
                    result['macd_signal'] = float('nan')
                elif key == 'bollinger_bands':
                    result['bollinger_upper'] = float('nan')
                    result['bollinger_lower'] = float('nan')
                    result['bollinger_bands_width'] = float('nan')
                elif key in ['stoch_rsi_k', 'stoch_rsi_d']:
                    result['stoch_rsi_k'] = float('nan')
                    result['stoch_rsi_d'] = float('nan')
                elif key in ['roc', 'roc_fast', 'roc_slow']:
                    result[key] = float('nan')
                else:
                    result[key] = float('nan')
            return result
    
    # Individual indicator calculation functions for single-tick calculations
    def _calculate_single_rsi(self, temp_data: pd.DataFrame, period: int, last_idx: int, price_column: str = 'close') -> float:
        """Calculate RSI for the latest tick"""
        print(f"🔧 DEBUG: _calculate_single_rsi called - period={period}, last_idx={last_idx}, data_len={len(temp_data)}")
        
        if len(temp_data) >= period + 1:
            delta = temp_data[price_column].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            
            # Handle division by zero case
            current_gain = gain.iloc[last_idx]
            current_loss = loss.iloc[last_idx]
            
            if pd.isna(current_gain) or pd.isna(current_loss):
                print(f"🔧 DEBUG: RSI calculation - gain or loss is NaN")
                return float('nan')
            elif current_loss == 0:
                # When loss = 0, RSI should be 100 (extremely overbought)
                print(f"🔧 DEBUG: RSI calculation - loss = 0, setting RSI = 100")
                return 100.0
            elif current_gain == 0:
                # When gain = 0, RSI should be 0 (extremely oversold)
                print(f"🔧 DEBUG: RSI calculation - gain = 0, setting RSI = 0")
                return 0.0
            else:
                rs = current_gain / current_loss
                rsi = 100 - (100 / (1 + rs))
                print(f"🔧 DEBUG: RSI calculation - delta range: {delta.min():.4f} to {delta.max():.4f}")
                print(f"🔧 DEBUG: RSI calculation - gain: {current_gain:.4f}, loss: {current_loss:.4f}")
                print(f"🔧 DEBUG: RSI calculation - RS: {rs:.4f}, RSI: {rsi:.4f}")
                return rsi
        else:
            print(f"🔧 DEBUG: Not enough data for RSI - need {period + 1}, have {len(temp_data)}")
        return float('nan')
    
    def _calculate_single_ema(self, prev_ema: float, new_price: float, period: int) -> float:
        """
        Incrementally update EMA for streaming data.

        Args:
            prev_ema (float): The previous EMA value (NaN if not available yet).
            new_price (float): The latest price tick.
            period (int): EMA period length.

        Returns:
            float: Updated EMA value.
        """
        alpha = 2 / (period + 1)

        if prev_ema is None or pd.isna(prev_ema):
            # Initialize EMA with the first price
            return new_price  

        # Standard recursive EMA formula
        return alpha * new_price + (1 - alpha) * prev_ema
    
    def _calculate_single_sma(self, temp_data: pd.DataFrame, period: int, last_idx: int, price_column: str = 'close') -> float:
        """Calculate SMA for the latest tick"""
        if len(temp_data) > period:  # Need past 'period' periods + current period  
            sma = temp_data[price_column].rolling(window=period).mean()
            return sma.iloc[last_idx] if not pd.isna(sma.iloc[last_idx]) else float('nan')
        return float('nan')
    
    def _calculate_single_macd_line(self, temp_data: pd.DataFrame, periods: dict, last_idx: int, price_column: str = 'close') -> float:
        """Calculate MACD line for the latest tick - starts from Row 1 (platform behavior)"""
        fast_period = periods.get('macd_fast', 12)
        slow_period = periods.get('macd_slow', 26)
        
        if len(temp_data) == 0:
            return float('nan')
        
        # Platform behavior: MACD starts immediately from Row 1
        # Check if we have existing EMA data for incremental calculation
        if 'ema_fast' in temp_data.columns and 'ema_slow' in temp_data.columns and last_idx > 0:
            prev_fast_ema = temp_data['ema_fast'].iloc[last_idx - 1]
            prev_slow_ema = temp_data['ema_slow'].iloc[last_idx - 1]
            
            if pd.notna(prev_fast_ema) and pd.notna(prev_slow_ema):
                # Incremental EMA calculation
                alpha_fast = 2 / (fast_period + 1)
                alpha_slow = 2 / (slow_period + 1)
                current_price = temp_data[price_column].iloc[last_idx]
                
                new_fast_ema = alpha_fast * current_price + (1 - alpha_fast) * prev_fast_ema
                new_slow_ema = alpha_slow * current_price + (1 - alpha_slow) * prev_slow_ema
                
                return new_fast_ema - new_slow_ema
        
        # Fallback: calculate from scratch - starts immediately from Row 1
        ema_fast = temp_data[price_column].ewm(span=fast_period, adjust=False).mean()
        ema_slow = temp_data[price_column].ewm(span=slow_period, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        return macd_line.iloc[last_idx] if not pd.isna(macd_line.iloc[last_idx]) else float('nan')
    
    def _calculate_single_macd_signal(self, temp_data: pd.DataFrame, periods: dict, last_idx: int, price_column: str = 'close') -> float:
        """Calculate MACD signal for the latest tick - starts from Row 1 (platform behavior)"""
        signal_period = periods.get('macd_signal', 9)
        
        if len(temp_data) == 0:
            return float('nan')
        
        # Platform behavior: MACD signal starts immediately from Row 1
        # Check if we have existing MACD signal data for incremental calculation
        if 'macd_signal' in temp_data.columns and 'macd_line' in temp_data.columns and last_idx > 0:
            prev_signal = temp_data['macd_signal'].iloc[last_idx - 1]
            current_macd = temp_data['macd_line'].iloc[last_idx]
            
            if pd.notna(prev_signal) and pd.notna(current_macd):
                # Incremental signal EMA calculation
                alpha = 2 / (signal_period + 1)
                return alpha * current_macd + (1 - alpha) * prev_signal
        
        # Fallback: calculate from scratch - starts immediately from Row 1
        fast_period = periods.get('macd_fast', 12)
        slow_period = periods.get('macd_slow', 26)
        
        ema_fast = temp_data[price_column].ewm(span=fast_period, adjust=False).mean()
        ema_slow = temp_data[price_column].ewm(span=slow_period, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        return signal_line.iloc[last_idx] if not pd.isna(signal_line.iloc[last_idx]) else float('nan')
    
    def _calculate_single_bollinger_upper(self, temp_data: pd.DataFrame, periods: dict, last_idx: int, price_column: str = 'close') -> float:
        """Calculate Bollinger upper band for the latest tick"""
        period = periods.get('bollinger_bands', 20)
        std_dev = periods.get('bollinger_std_dev', 2)
        if len(temp_data) >= period:
            sma = temp_data[price_column].rolling(window=period).mean()
            std = temp_data[price_column].rolling(window=period).std()
            upper_band = sma + (std * std_dev)
            return upper_band.iloc[last_idx] if not pd.isna(upper_band.iloc[last_idx]) else float('nan')
        return float('nan')
    
    def _calculate_single_bollinger_lower(self, temp_data: pd.DataFrame, periods: dict, last_idx: int, price_column: str = 'close') -> float:
        """Calculate Bollinger lower band for the latest tick"""
        period = periods.get('bollinger_bands', 20)
        std_dev = periods.get('bollinger_std_dev', 2)
        if len(temp_data) >= period:
            sma = temp_data[price_column].rolling(window=period).mean()
            std = temp_data[price_column].rolling(window=period).std()
            lower_band = sma - (std * std_dev)
            return lower_band.iloc[last_idx] if not pd.isna(lower_band.iloc[last_idx]) else float('nan')
        return float('nan')
    
    def _calculate_single_bollinger_width(self, temp_data: pd.DataFrame, periods: dict, last_idx: int, price_column: str = 'close') -> float:
        """Calculate Bollinger bands width for the latest tick"""
        period = periods.get('bollinger_bands', 20)
        std_dev = periods.get('bollinger_std_dev', 2)
        if len(temp_data) >= period:
            sma = temp_data[price_column].rolling(window=period).mean()
            std = temp_data[price_column].rolling(window=period).std()
            upper_band = sma + (std * std_dev)
            lower_band = sma - (std * std_dev)
            bb_width = (upper_band - lower_band) / sma * 100
            return bb_width.iloc[last_idx] if not pd.isna(bb_width.iloc[last_idx]) else float('nan')
        return float('nan')
    
    def _calculate_single_roc(self, temp_data: pd.DataFrame, period: int, last_idx: int, price_column: str = 'close') -> float:
        """Calculate ROC for the latest tick using direct calculation for maximum efficiency"""
        if len(temp_data) <= period or last_idx < period:
            return float('nan')
        
        # Direct ROC calculation: (current_price - past_price) / past_price * 100
        current_price = temp_data[price_column].iloc[last_idx]
        past_price = temp_data[price_column].iloc[last_idx - period]
        
        if pd.isna(current_price) or pd.isna(past_price) or past_price == 0:
            return float('nan')
        
        return ((current_price - past_price) / past_price) * 100
    
    def _calculate_single_roc_of_roc(self, temp_data: pd.DataFrame, periods: dict, last_idx: int, price_column: str = 'close') -> float:
        """Calculate ROC of ROC for the latest tick using existing ROC data for maximum efficiency"""
        roc_of_roc_period = periods.get('roc_of_roc', 10)
        
        # Check if we have existing ROC data to use
        if 'roc' in temp_data.columns and last_idx >= roc_of_roc_period:
            current_roc = temp_data['roc'].iloc[last_idx]
            past_roc = temp_data['roc'].iloc[last_idx - roc_of_roc_period]
            
            if pd.notna(current_roc) and pd.notna(past_roc) and past_roc != 0:
                # Direct ROC of ROC calculation: (current_roc - past_roc) / past_roc * 100
                return ((current_roc - past_roc) / past_roc) * 100
        
        return float('nan')
    def _calculate_single_rsi_roc(self, temp_data: pd.DataFrame, period: int, last_idx: int, price_column: str = 'close') -> float:
        """Calculate ROC of RSI for the latest tick"""
        rsi_period = 14  # Default RSI period
        if len(temp_data) < rsi_period + period + 1:
            return float('nan')
        
        # First calculate RSI series
        rsi_series = self.calculate_rsi(temp_data, rsi_period, price_column)
        
        # Then calculate ROC of the RSI series
        if len(rsi_series) >= period + 1:
            current_rsi = rsi_series.iloc[last_idx]
            past_rsi = rsi_series.iloc[last_idx - period]
            
            if pd.isna(current_rsi) or pd.isna(past_rsi) or past_rsi == 0:
                return float('nan')
            
            # ROC calculation: (current_rsi - past_rsi) / past_rsi * 100
            return ((current_rsi - past_rsi) / past_rsi) * 100
        
        return float('nan')
    
    def _calculate_single_stoch_rsi_k(self, temp_data: pd.DataFrame, periods: dict, last_idx: int, price_column: str = 'close') -> float:
        """Calculate Stochastic RSI %K for the latest tick"""
        rsi_period = periods.get('stoch_rsi_period', 14)
        k_period = periods.get('stoch_rsi_k', 3)
        # Need: rsi_period for RSI + rsi_period for min/max + k_period for smoothing
        if len(temp_data) >= rsi_period * 2 + k_period:
            delta = temp_data[price_column].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            rsi_min = rsi.rolling(window=rsi_period).min()
            rsi_max = rsi.rolling(window=rsi_period).max()
            stoch_rsi = (rsi - rsi_min) / (rsi_max - rsi_min)
            stoch_rsi_k = stoch_rsi.rolling(window=k_period).mean() * 100
            return stoch_rsi_k.iloc[last_idx] if not pd.isna(stoch_rsi_k.iloc[last_idx]) else float('nan')
        return float('nan')
    
    def _calculate_single_stoch_rsi_d(self, temp_data: pd.DataFrame, periods: dict, last_idx: int, price_column: str = 'close') -> float:
        """Calculate Stochastic RSI %D for the latest tick"""
        rsi_period = periods.get('stoch_rsi_period', 14)
        k_period = periods.get('stoch_rsi_k', 3)
        d_period = periods.get('stoch_rsi_d', 3)
        # Need: rsi_period for RSI + rsi_period for min/max + k_period for %K + d_period for %D
        if len(temp_data) >= rsi_period * 2 + k_period + d_period:
            delta = temp_data[price_column].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            rsi_min = rsi.rolling(window=rsi_period).min()
            rsi_max = rsi.rolling(window=rsi_period).max()
            stoch_rsi = (rsi - rsi_min) / (rsi_max - rsi_min)
            stoch_rsi_k = stoch_rsi.rolling(window=k_period).mean() * 100
            stoch_rsi_d = stoch_rsi_k.rolling(window=d_period).mean()
            return stoch_rsi_d.iloc[last_idx] if not pd.isna(stoch_rsi_d.iloc[last_idx]) else float('nan')
        return float('nan')
    
    def _calculate_single_atr(self, temp_data: pd.DataFrame, period: int, last_idx: int) -> float:
        """Calculate ATR for the latest tick"""
        
        # For options data, use tick_high, tick_low, tick_close (OHLCV for ticks)
        if 'tick_close' in temp_data.columns:
            price_col = 'tick_close'
            high_col = 'tick_high'
            low_col = 'tick_low'
        # For equity data, use high_price, low_price, close_price
        elif 'close_price' in temp_data.columns:
            price_col = 'close_price'
            high_col = 'high_price'
            low_col = 'low_price'
        # Fallback for mark_price (options)
        elif 'mark_price' in temp_data.columns:
            price_col = 'mark_price'
            high_col = 'high_price'
            low_col = 'low_price'
        else:
            return float('nan')
        
        if len(temp_data) >= period and all(col in temp_data.columns for col in [high_col, low_col, price_col]):
            temp_data_copy = temp_data.copy()
            temp_data_copy['prev_close'] = temp_data_copy[price_col].shift(1)
            temp_data_copy['tr1'] = temp_data_copy[high_col] - temp_data_copy[low_col]
            temp_data_copy['tr2'] = abs(temp_data_copy[high_col] - temp_data_copy[price_col].shift(1))
            temp_data_copy['tr3'] = abs(temp_data_copy[low_col] - temp_data_copy[price_col].shift(1))
            temp_data_copy['true_range'] = temp_data_copy[['tr1', 'tr2', 'tr3']].max(axis=1)
            atr = temp_data_copy['true_range'].rolling(window=period).mean()
            return atr.iloc[last_idx] if not pd.isna(atr.iloc[last_idx]) else float('nan')
        return float('nan')
    
    def _calculate_single_vwma(self, price: float, volume: float) -> float:
        """Calculate VWMA for the latest tick using direct window calculation for maximum efficiency"""
        """
        Update VWMA with a new (price, volume) tick.

        Args:
            price (float): Latest price
            volume (float): Latest volume

        Returns:
            float: Updated VWMA value (NaN until window is full)
        """
        # If window is full, remove oldest contribution
        if len(price_window) == period:
            old_price = price_window[0]
            old_volume = volume_window[0]
            numerator -= old_price * old_volume
            denominator -= old_volume

        # Add new tick
        price_window.append(price)
        volume_window.append(volume)
        numerator += price * volume
        denominator += volume

        # Compute VWMA
        if len(price_window) < period or denominator == 0:
            return float('nan')
        return numerator / denominator
    
    
    def _calculate_single_price_change(self, temp_data: pd.DataFrame, last_idx: int, price_column: str = 'close') -> float:
        """Calculate price change for the latest tick"""
        if len(temp_data) >= 2:
            price_change = temp_data[price_column].pct_change()
            return price_change.iloc[last_idx] if not pd.isna(price_change.iloc[last_idx]) else float('nan')
        return float('nan')
    

    def _calculate_single_gaussian_smoothing(self, temp_data: pd.DataFrame, period: int, last_idx: int, column: str = 'close') -> float:
        """Calculate Gaussian smoothing for the latest tick"""
        print(f"🔧 DEBUG: _calculate_single_gaussian_smoothing called - column={column}, period={period}, last_idx={last_idx}, data_len={len(temp_data)}")
        
        if len(temp_data) < period or last_idx < period - 1:
            print(f"🔧 DEBUG: Not enough data - need {period}, have {len(temp_data)}, last_idx={last_idx}")
            return float('nan')
        
        # Check if column exists
        if column not in temp_data.columns:
            print(f"🔧 DEBUG: Column '{column}' not found in temp_data.columns: {temp_data.columns.tolist()}")
            return float('nan')
        
        # Get the column window for calculation
        start_idx = last_idx - period + 1
        column_window = temp_data[column].iloc[start_idx:last_idx + 1]
        print(f"🔧 DEBUG: Column window indices: {start_idx} to {last_idx}")
        print(f"🔧 DEBUG: Column window values: {column_window.tolist()}")
        
        # Check if we have enough valid (non-NaN) values in the column window
        valid_values = column_window.notna().sum()
        print(f"🔧 DEBUG: Valid values in window: {valid_values}/{len(column_window)}")
        
        if valid_values < period:
            print(f"🔧 DEBUG: Not enough valid values - need {period}, have {valid_values}")
            return float('nan')
        
        # Create Gaussian weights (normal distribution)
        # Center the weights around the middle of the period
        center = (period - 1) / 2
        sigma = period / 6  # Standard deviation (covers ~99.7% of data)
        
        # Generate weights for the period
        weights = []
        for i in range(period):
            # Calculate Gaussian weight
            weight = np.exp(-0.5 * ((i - center) / sigma) ** 2)
            weights.append(weight)
        
        # Normalize weights to sum to 1
        weights = np.array(weights)
        weights = weights / weights.sum()
        print(f"🔧 DEBUG: Gaussian weights: {weights.tolist()}")
        
        # Calculate weighted average
        gaussian_smoothed = np.sum(column_window * weights)
        print(f"🔧 DEBUG: Gaussian smoothed result: {gaussian_smoothed}")
        
        return gaussian_smoothed if not pd.isna(gaussian_smoothed) else float('nan')
    
    def _calculate_single_gaussian_smoothing_price(self, temp_data: pd.DataFrame, period: int, last_idx: int) -> float:
        """Calculate Gaussian smoothing for close prices"""
        return self._calculate_single_gaussian_smoothing(temp_data, period, last_idx, 'close')

    def _calculate_single_gaussian_smoothing_ema(self, prev_emas, new_price: float, period: int, passes: int = 2) -> float:
        """
        Update Gaussian EMA (multi-pass EMA) for streaming data.

        Args:
            prev_emas (list[float]): Previous EMA values for each pass (len = passes).
                                    Use [None]*passes for initialization.
            new_price (float): Latest price tick.
            period (int): EMA period length.
            passes (int): How many EMA passes to apply (2–4 typical).

        Returns:
            list[float]: Updated EMA values for each pass.
            float: Final Gaussian EMA (last pass).
        """
        alpha = 2 / (period + 1)
        updated = []

        for i in range(passes):
            if prev_emas[i] is None:
                # First initialization for this pass
                ema_value = new_price if i == 0 else updated[i-1]
            else:
                price_input = new_price if i == 0 else updated[i-1]
                ema_value = alpha * price_input + (1 - alpha) * prev_emas[i]
            updated.append(ema_value)

        return updated, updated[-1]

    def _calculate_single_gaussian_smoothing_ema_fast(self, temp_data: pd.DataFrame, period: int, last_idx: int) -> float:
        """Calculate Gaussian smoothing for EMA"""
        return self._calculate_single_gaussian_smoothing(temp_data, period, last_idx, 'ema_fast')
    
    def _calculate_single_gaussian_smoothing_ema_slow(self, temp_data: pd.DataFrame, period: int, last_idx: int) -> float:
        """Calculate Gaussian smoothing for VWMA"""
        return self._calculate_single_gaussian_smoothing(temp_data, period, last_idx, 'ema_slow')

    def _calculate_single_ema_fast(self, temp_data: pd.DataFrame, period: int, last_idx: int) -> float:
        """Calculate EMA fast"""
        return self._calculate_single_ema(temp_data, period, last_idx, 'ema_fast')
    
    def _calculate_single_ema_slow(self, temp_data: pd.DataFrame, period: int, last_idx: int) -> float:
        """Calculate EMA slow"""
        return self._calculate_single_ema(temp_data, period, last_idx, 'ema_slow')

    def _calculate_single_ema(self, temp_data: pd.DataFrame, period: int, last_idx: int, price_column: str = 'close') -> float:
        """Calculate EMA for the latest tick using DataFrame data"""
        if len(temp_data) < period or last_idx < period - 1:
            return float('nan')
        
        # Calculate EMA using pandas
        ema = temp_data[price_column].ewm(span=period, adjust=False).mean()
        return ema.iloc[last_idx] if not pd.isna(ema.iloc[last_idx]) else float('nan')

    def _calculate_single_vwma(self, temp_data: pd.DataFrame, period: int, last_idx: int, price_column: str = 'close') -> float:
        """Calculate VWMA for the latest tick using DataFrame data"""
        if len(temp_data) < period or last_idx < period - 1:
            return float('nan')
        
        # Get volume column
        volume_column = self._get_volume_column(temp_data)
        
        # Calculate VWMA using rolling windows
        price_volume = temp_data[price_column] * temp_data[volume_column]
        vwma = price_volume.rolling(window=period).sum() / temp_data[volume_column].rolling(window=period).sum()
        
        return vwma.iloc[last_idx] if not pd.isna(vwma.iloc[last_idx]) else float('nan')