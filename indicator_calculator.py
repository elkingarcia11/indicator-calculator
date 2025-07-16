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
        data['ROC of ROC'] = data['ROC'].pct_change(period=period)
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
        if 'Volume' not in data.columns:
            print("Volume data not available, using SMA instead of VWMA")
            return data['Close'].rolling(window=period).mean()
        
        # Use typical price for VWMA calculation
        typical_price = (data['High'] + data['Low'] + data['Close']) / 3
        
        # Handle potential volume data issues
        volume = data['Volume'].fillna(1)  # Replace NaN volumes with 1
        
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
            
            if 'RSI' in indicator_periods:
                # Calculate RSI
                result['RSI'] = IndicatorCalculator.calculate_rsi(result['Close'], indicator_periods['RSI'])
            if 'Stoch_RSI_K' in indicator_periods and 'Stoch_RSI_D' in indicator_periods:
                # Calculate Stochastic RSI
                stoch_rsi_k, stoch_rsi_d = IndicatorCalculator.calculate_stochastic_rsi(result['Close'], indicator_periods['Stoch_RSI_K'], indicator_periods['Stoch_RSI_D'])
                result['Stoch_RSI_K'] = stoch_rsi_k
                result['Stoch_RSI_D'] = stoch_rsi_d
            
            if 'MACD_Fast' in indicator_periods and 'MACD_Slow' in indicator_periods and 'MACD_Signal' in indicator_periods:
                # Calculate MACD
                macd_line, signal_line, histogram = IndicatorCalculator.calculate_macd(result['Close'], indicator_periods['MACD_Fast'], indicator_periods['MACD_Slow'], indicator_periods['MACD_Signal'])
                result['MACD_Line'] = macd_line
                result['MACD_Signal'] = signal_line
                result['MACD_Histogram'] = histogram
            
            if 'ROC' in indicator_periods:
                # Calculate ROC
                result['ROC'] = IndicatorCalculator.calculate_roc(result['Close'], indicator_periods['ROC'])
                if 'ROC_of_ROC' in indicator_periods:
                    result['ROC_of_ROC'] = IndicatorCalculator.calculate_roc_of_roc(result['ROC'], indicator_periods['ROC_of_ROC'])
            
            if 'EMA' in indicator_periods:
                # Calculate EMAs
                result['EMA'] = IndicatorCalculator.calculate_ema(result['Close'], indicator_periods['EMA'])
            
            if 'VWMA' in indicator_periods:
                # Calculate VWMA
                result['VWMA'] = IndicatorCalculator.calculate_vwma(result, indicator_periods['VWMA'])
            
            if 'SMA' in indicator_periods:
                # Calculate SMA
                result['SMA'] = IndicatorCalculator.calculate_sma(result['Close'], indicator_periods['SMA'])
            
            if 'Volatility' in indicator_periods:
                # Calculate Volatility
                result['Volatility'] = IndicatorCalculator.calculate_volatility(result['Close'], indicator_periods['Volatility'])
            
            if 'Price_Change' in indicator_periods:
                # Calculate Price Change
                result['Price_Change'] = IndicatorCalculator.calculate_price_change(data=result['Close'])
            
            if 'Bollinger_Bands' in indicator_periods:
                # Calculate Bollinger Bands
                result['Bollinger_Bands'] = IndicatorCalculator.calculate_bollinger_bands(result['Close'], indicator_periods['Bollinger_Bands'])
            
            if 'Bollinger_Bands_Width' in indicator_periods:
                # Calculate Bollinger Bands Width
                result['Bollinger_Bands_Width'] = IndicatorCalculator.calculate_bollinger_bands_width(result['Close'], indicator_periods['Bollinger_Bands_Width'])
            
            if 'ATR' in indicator_periods:
                # Calculate ATR
                result['ATR'] = IndicatorCalculator.calculate_atr(result['Close'], indicator_periods['ATR'])
            
            print(f"Calculated {len(result.columns) - len(data.columns)} indicators")
            return result
            
        except Exception as e:
            print(f"Error calculating indicators: {str(e)}")
            return data