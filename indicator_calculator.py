import pandas as pd

class IndicatorCalculator:
    """Class to calculate various technical indicators"""
    
    @staticmethod
    def calculate_rsi(data: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index (RSI)
        
        Args:
            data: Price series (typically Close prices)
            period: Lookback period for RSI calculation
            
        Returns:
            RSI values as pandas Series
        """
        # Calculate the difference between the current and previous price
        delta = data.diff()
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
    def calculate_stochastic_rsi(data: pd.Series, period: int = 14, k_period: int = 3, d_period: int = 3) -> tuple:
        """
        Calculate Stochastic RSI
        
        Args:
            data: Price series
            period: RSI period
            k_period: %K period
            d_period: %D period
            
        Returns:
            Tuple of (%K, %D) values
        """
        # Calculate the RSI
        rsi = IndicatorCalculator.calculate_rsi(data, period)
        # Calculate the Stochastic RSI %K
        stoch_rsi_k = (rsi - rsi.rolling(window=period).min()) / (rsi.rolling(window=period).max() - rsi.rolling(window=period).min())
        # Calculate the Stochastic RSI %K
        stoch_rsi_k = stoch_rsi_k.rolling(window=k_period).mean() * 100
        # Calculate the Stochastic RSI %D
        stoch_rsi_d = stoch_rsi_k.rolling(window=d_period).mean()
        return stoch_rsi_k, stoch_rsi_d
    
    @staticmethod
    def calculate_macd(data: pd.Series, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> tuple:
        """
        Calculate MACD (Moving Average Convergence Divergence)
        
        Args:
            data: Price series
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal line period
            
        Returns:
            Tuple of (MACD line, Signal line, Histogram)
        """
        # Calculate the fast EMA
        ema_fast = data.ewm(span=fast_period, adjust=False).mean()
        # Calculate the slow EMA
        ema_slow = data.ewm(span=slow_period, adjust=False).mean()
        # Calculate the MACD line
        macd_line = ema_fast - ema_slow
        # Calculate the signal line
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        # Calculate the histogram
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    @staticmethod
    def calculate_roc(data: pd.Series, period: int = 10) -> pd.Series:
        """
        Calculate Rate of Change (ROC)
        
        Args:
            data: Price series
            period: Lookback period
            
        Returns:
            ROC values as pandas Series
        """
        # Calculate the Rate of Change
        return ((data - data.shift(period)) / data.shift(period)) * 100
    
    @staticmethod
    def calculate_roc_of_roc(data: pd.Series, period: int = 10) -> pd.Series:
        """
        Calculate the ROC of the ROC for the data
        """
        # Calculate the ROC of the ROC
        data['roc_of_roc'] = data['roc'].pct_change(period=period)
        # Return the data
        return data
    
    @staticmethod
    def calculate_ema(data: pd.Series, period: int = 20) -> pd.Series:
        """
        Calculate Exponential Moving Average (EMA)
        
        Args:
            data: Price series
            period: EMA period
            
        Returns:
            EMA values as pandas Series
        """
        # Calculate the EMA
        return data.ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def calculate_vwma(data: pd.DataFrame, period: int = 20) -> pd.Series:
        """
        Calculate Volume Weighted Moving Average (VWMA)
        
        Args:
            data: DataFrame with 'Close' and 'Volume' columns
            period: VWMA period
            
        Returns:
            VWMA values as pandas Series
        """
        if 'volume' not in data.columns:
            print("Volume data not available, using SMA instead of VWMA")
            return data['close'].rolling(window=period).mean()
        
        # Use typical price for VWMA calculation
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        
        # Handle potential volume data issues
        volume = data['volume'].fillna(1)  # Replace NaN volumes with 1
        
        # Calculate the VWMA
        vwma = (typical_price * volume).rolling(window=period).sum() / volume.rolling(window=period).sum()
        # Return the VWMA
        return vwma

    @staticmethod
    def calculate_sma(data: pd.Series, period: int = 20) -> pd.Series:
        """
        Calculate Simple Moving Average (SMA)
        """
        # Calculate the SMA
        return data.rolling(window=period).mean()
    
    @staticmethod
    def calculate_price_change(data: pd.Series) -> pd.Series:
        """
        Calculate Price Change
        """
        # Calculate the Price Change
        return data.pct_change()
    
    @staticmethod
    def calculate_volatility(data: pd.Series, period: int = 20) -> pd.Series:
        """
        Calculate Volatility
        """
        # Calculate the Volatility
        return data.rolling(window=period).std()
    
    @staticmethod
    def calculate_bollinger_bands(data: pd.Series, period: int = 20, std_dev: int = 2) -> tuple:
        """
        Calculate Bollinger Bands
        """
        # Calculate the Bollinger Bands
        return data.rolling(window=period).mean(), data.rolling(window=period).std() * std_dev
    
    @staticmethod
    def calculate_bollinger_bands_width(data: pd.Series, period: int = 20, std_dev: int = 2) -> pd.Series:
        """
        Calculate Bollinger Bands Width
        """
        # Calculate the Bollinger Bands
        return data.rolling(window=period).std() * std_dev
        
    @staticmethod
    def calculate_atr(data: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Average True Range (ATR)
        """
        # Calculate the ATR
        return data.rolling(window=period).mean()
    
    def calculate_all_indicators(self, data: pd.DataFrame, indicator_periods: dict = {}) -> pd.DataFrame:
        """
        Calculate all technical indicators for the given data
        
        Args:
            data: OHLCV DataFrame
            
        Returns:
            DataFrame with calculated indicators
        """
        try:
            # Make a copy to avoid modifying original data
            result = data.copy()
            
            if 'rsi' in indicator_periods:
                # Calculate RSI
                result['rsi'] = IndicatorCalculator.calculate_rsi(result['close'], indicator_periods['rsi'])
                
            if 'stoch_rsi_k' in indicator_periods and 'stoch_rsi_d' in indicator_periods:
                # Calculate Stochastic RSI
                stoch_rsi_k, stoch_rsi_d = IndicatorCalculator.calculate_stochastic_rsi(result['close'], indicator_periods['stoch_rsi_k'], indicator_periods['stoch_rsi_d'])
                result['stoch_rsi_k'] = stoch_rsi_k
                result['stoch_rsi_d'] = stoch_rsi_d
            
            if 'macd_fast' in indicator_periods and 'macd_slow' in indicator_periods and 'macd_signal' in indicator_periods:
                # Calculate MACD
                macd_line, signal_line, histogram = IndicatorCalculator.calculate_macd(result['close'], indicator_periods['macd_fast'], indicator_periods['macd_slow'], indicator_periods['macd_signal'])
                result['macd_line'] = macd_line
                result['macd_signal'] = signal_line
                result['macd_histogram'] = histogram
            
            if 'roc' in indicator_periods:
                # Calculate ROC
                result['roc'] = IndicatorCalculator.calculate_roc(result['close'], indicator_periods['roc'])
                if 'roc_of_roc' in indicator_periods:
                    result['roc_of_roc'] = IndicatorCalculator.calculate_roc_of_roc(result['roc'], indicator_periods['roc_of_roc'])
            
            if 'ema' in indicator_periods:
                # Calculate EMAs
                result['ema'] = IndicatorCalculator.calculate_ema(result['close'], indicator_periods['ema'])
            
            if 'vwma' in indicator_periods:
                # Calculate VWMA
                result['vwma'] = IndicatorCalculator.calculate_vwma(result, indicator_periods['vwma'])
            
            if 'sma' in indicator_periods:
                # Calculate SMA
                result['sma'] = IndicatorCalculator.calculate_sma(result['close'], indicator_periods['sma'])
            
            if 'volatility' in indicator_periods:
                # Calculate Volatility
                result['volatility'] = IndicatorCalculator.calculate_volatility(result['close'], indicator_periods['volatility'])
            
            if 'price_change' in indicator_periods:
                # Calculate Price Change
                result['price_change'] = IndicatorCalculator.calculate_price_change(data=result['close'])
            
            if 'bollinger_bands' in indicator_periods:
                # Calculate Bollinger Bands
                result['bollinger_bands'] = IndicatorCalculator.calculate_bollinger_bands(result['close'], indicator_periods['bollinger_bands'])
            
            if 'bollinger_bands_width' in indicator_periods:
                # Calculate Bollinger Bands Width
                result['bollinger_bands_width'] = IndicatorCalculator.calculate_bollinger_bands_width(result['close'], indicator_periods['bollinger_bands_width'])
            
            if 'atr' in indicator_periods:
                # Calculate ATR
                result['atr'] = IndicatorCalculator.calculate_atr(result['close'], indicator_periods['atr'])
            
            print(f"Calculated {len(result.columns) - len(data.columns)} indicators")
            return result
            
        except Exception as e:
            print(f"Error calculating indicators: {str(e)}")
            return data