import pandas as pd

class IndicatorCalculator:
    """Class to calculate various technical indicators"""
    
    def __init__(self):
        """Initialize the indicator calculator with function mappings"""
        self.indicator_functions = {
            'rsi': self._calculate_single_rsi,
            'ema': self._calculate_single_ema,
            'sma': self._calculate_single_sma,
            'macd_line': self._calculate_single_macd_line,
            'macd_signal': self._calculate_single_macd_signal,
            'bollinger_upper': self._calculate_single_bollinger_upper,
            'bollinger_lower': self._calculate_single_bollinger_lower,
            'bollinger_bands_width': self._calculate_single_bollinger_width,
            'roc': self._calculate_single_roc,
            'roc_of_roc': self._calculate_single_roc_of_roc,
            'stoch_rsi_k': self._calculate_single_stoch_rsi_k,
            'stoch_rsi_d': self._calculate_single_stoch_rsi_d,
            'atr': self._calculate_single_atr,
            'vwma': self._calculate_single_vwma,
            'volatility': self._calculate_single_volatility,
            'price_change': self._calculate_single_price_change
        }
    
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
        
        Args:
            data: Price series
            period: RSI period
            k_period: %K period
            d_period: %D period
            
        Returns:
            Tuple of (%K, %D) values
        """
        # Calculate the RSI
        rsi = IndicatorCalculator.calculate_rsi(data, period, price_column)
        # Calculate the Stochastic RSI %K
        stoch_rsi_k = (rsi - rsi.rolling(window=period).min()) / (rsi.rolling(window=period).max() - rsi.rolling(window=period).min())
        # Calculate the Stochastic RSI %K
        stoch_rsi_k = stoch_rsi_k.rolling(window=k_period).mean() * 100
        # Calculate the Stochastic RSI %D
        stoch_rsi_d = stoch_rsi_k.rolling(window=d_period).mean()
        return stoch_rsi_k, stoch_rsi_d
    
    @staticmethod
    def calculate_macd(data: pd.DataFrame, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9, price_column: str = 'close') -> tuple:
        """
        Calculate MACD (Moving Average Convergence Divergence)
        
        Args:
            data: Price series
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal line period
            price_column: Column name to use for price data
        Returns:
            Tuple of (MACD line, Signal line)
        """
        # Calculate the fast EMA
        ema_fast = data[price_column].ewm(span=fast_period, adjust=False).mean()
        # Calculate the slow EMA
        ema_slow = data[price_column].ewm(span=slow_period, adjust=False).mean()
        # Calculate the MACD line
        macd_line = ema_fast - ema_slow
        # Calculate the signal line
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        return macd_line, signal_line
    
    @staticmethod
    def calculate_roc(data: pd.DataFrame, period: int = 10, price_column: str = 'close') -> pd.Series:
        """
        Calculate Rate of Change (ROC)
        
        Args:
            data: Price series
            period: Lookback period
            
        Returns:
            ROC values as pandas Series
        """
        # Calculate the Rate of Change
        return ((data[price_column] - data[price_column].shift(period)) / data[price_column].shift(period)) * 100
    
    @staticmethod
    def calculate_roc_of_roc(data: pd.DataFrame, roc_of_roc_period: int = 10) -> pd.Series:
        """
        Calculate ROC of ROC using the existing 'roc' column.
        """
        # should start calculating roc of roc after roc_period
        roc = data['roc']
        roc_of_roc = ((roc - roc.shift(roc_of_roc_period)) / roc.shift(roc_of_roc_period)) * 100
        return roc_of_roc
    
    @staticmethod
    def calculate_ema(data: pd.DataFrame, period: int = 20, price_column: str = 'close') -> pd.Series:
        """
        Calculate Exponential Moving Average (EMA)
        
        Args:
            data: Price series
            period: EMA period
            
        Returns:
            EMA values as pandas Series
        """
        # Calculate the EMA
        return data[price_column].ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def calculate_vwma(data: pd.DataFrame, period: int = 20, is_options: bool = False) -> pd.Series:
        """
        Calculate Volume Weighted Moving Average (VWMA)
        
        Args:
            data: DataFrame with 'Close' and 'Volume' columns
            period: VWMA period
            
        Returns:
            VWMA values as pandas Series
        """
        if is_options:
            typical_price = data['last_price']
        elif all(col in data.columns for col in ['high', 'low']):
            typical_price = (data['high'] + data['low'] + data['close']) / 3
        else:
            typical_price = data['close']
        
        # Calculate the VWMA
        numerator = (typical_price * data['volume']).rolling(window=period).sum()
        denominator = data['volume'].rolling(window=period).sum()
        
        # Avoid division by zero
        vwma = numerator / denominator.replace(0, float('nan'))
        
        # Return the VWMA
        return vwma

    @staticmethod
    def calculate_sma(data: pd.DataFrame, period: int = 20, price_column: str = 'close') -> pd.Series:
        """
        Calculate Simple Moving Average (SMA)
        """
        # Calculate the SMA
        return data[price_column].rolling(window=period).mean()
    
    @staticmethod
    def calculate_price_change(data: pd.DataFrame, price_column: str = 'close') -> pd.Series:
        """
        Calculate Price Change
        """
        # Calculate the Price Change
        return data[price_column].pct_change()
    
    @staticmethod
    def calculate_volatility(data: pd.DataFrame, period: int = 20, price_column: str = 'close') -> pd.Series:
        """
        Calculate Volatility
        """
        # Calculate the Volatility
        return data[price_column].rolling(window=period).std()
    
    @staticmethod
    def calculate_bollinger_bands(data: pd.DataFrame, period: int = 20, std_dev: int = 2, price_column: str = 'close') -> tuple:
        """
        Calculate Bollinger Bands
        """
        # Calculate the Bollinger Bands
        return data[price_column].rolling(window=period).mean(), data[price_column].rolling(window=period).std() * std_dev
    
    @staticmethod
    def calculate_bollinger_bands_width(data: pd.DataFrame, period: int = 20, std_dev: int = 2, price_column: str = 'close') -> pd.Series:
        """
        Calculate Bollinger Bands Width
        """
        # Calculate the Bollinger Bands
        return data[price_column].rolling(window=period).std() * std_dev
        
    @staticmethod
    def calculate_atr(data: pd.DataFrame, period: int = 14, price_column: str = 'close') -> pd.Series:
        """
        Calculate Average True Range (ATR)
        """
        # Calculate the ATR
        return data[price_column].rolling(window=period).mean()
    
    def calculate_all_indicators(self, data: pd.DataFrame, indicator_periods: dict = {}, is_option: bool = False) -> pd.DataFrame:
        """
        Calculate all technical indicators for the given data
        
        Args:
            data: OHLCV DataFrame
            indicator_periods: Dictionary with indicator names and their periods
            is_option: If True, uses 'last_price' instead of 'close' for price-based indicators
            price_column: Specific column name to use for price-based indicators (overrides is_option)
            
        Returns:
            DataFrame with calculated indicators
        """
        try:
            # Make a copy to avoid modifying original data
            result = data.copy()
            
            # Select the appropriate price column based on asset type
            price_column = 'last_price' if is_option else 'close'
            
            if 'rsi' in indicator_periods:
                # Calculate RSI
                result['rsi'] = IndicatorCalculator.calculate_rsi(result, indicator_periods['rsi'], price_column)
                
            if 'stoch_rsi_k' in indicator_periods and 'stoch_rsi_d' in indicator_periods and 'stoch_rsi_period' in indicator_periods:
                # Calculate Stochastic RSI
                stoch_rsi_k, stoch_rsi_d = IndicatorCalculator.calculate_stochastic_rsi(result,indicator_periods['stoch_rsi_period'], indicator_periods['stoch_rsi_k'], indicator_periods['stoch_rsi_d'], price_column)
                result['stoch_rsi_k'] = stoch_rsi_k
                result['stoch_rsi_d'] = stoch_rsi_d
            
            if 'macd_fast' in indicator_periods and 'macd_slow' in indicator_periods and 'macd_signal' in indicator_periods:
                # Calculate MACD
                macd_line, signal_line = IndicatorCalculator.calculate_macd(result, indicator_periods['macd_fast'], indicator_periods['macd_slow'], indicator_periods['macd_signal'], price_column)
                result['macd_line'] = macd_line
                result['macd_signal'] = signal_line
            
            if 'roc' in indicator_periods:
                # Calculate ROC_
                result['roc'] = IndicatorCalculator.calculate_roc(result, indicator_periods['roc'], price_column)
                if 'roc_of_roc' in indicator_periods:
                    result['roc_of_roc'] = IndicatorCalculator.calculate_roc_of_roc(result, indicator_periods['roc_of_roc'])
            
            if 'ema' in indicator_periods:
                # Calculate EMAs
                result['ema'] = IndicatorCalculator.calculate_ema(result, indicator_periods['ema'], price_column)
            
            if 'vwma' in indicator_periods:
                # Calculate VWMA
                result['vwma'] = IndicatorCalculator.calculate_vwma(result, indicator_periods['vwma'], is_option)
            
            if 'sma' in indicator_periods:
                # Calculate SMA
                result['sma'] = IndicatorCalculator.calculate_sma(result, indicator_periods['sma'], price_column)
            
            if 'volatility' in indicator_periods:
                # Calculate Volatility
                result['volatility'] = IndicatorCalculator.calculate_volatility(result, indicator_periods['volatility'], price_column)
            
            if 'price_change' in indicator_periods:
                # Calculate Price Change
                result['price_change'] = IndicatorCalculator.calculate_price_change(data=result, price_column=price_column)
            
            if 'bollinger_bands' in indicator_periods:
                # Calculate Bollinger Bands
                result['bollinger_bands'] = IndicatorCalculator.calculate_bollinger_bands(result, indicator_periods['bollinger_bands'], price_column)
            
            if 'bollinger_bands_width' in indicator_periods:
                # Calculate Bollinger Bands Width
                result['bollinger_bands_width'] = IndicatorCalculator.calculate_bollinger_bands_width(result, indicator_periods['bollinger_bands_width'], price_column)
            
            if 'atr' in indicator_periods:
                # Calculate ATR
                result['atr'] = IndicatorCalculator.calculate_atr(result, indicator_periods['atr'], price_column)
            return result
            
        except Exception as e:
            print(f"Error calculating indicators: {str(e)}")
            return data
            
    def calculate_latest_tick_indicators(self, existing_data: pd.DataFrame, new_row: pd.Series, indicator_periods: dict, is_option: bool = False) -> pd.Series:
        """
        Calculate technical indicators for a new tick/row based on existing historical data
        
        Args:
            existing_data: Historical OHLCV DataFrame with existing indicators
            new_row: New data row (Series) with OHLCV data
            indicator_periods: Dictionary with indicator names and their periods
            is_option: If True, uses 'last_price' instead of 'close' for price-based indicators
            
        Returns:
            Series with calculated indicator values for the new tick (empty strings for insufficient data)
        """
        try:
            # Create a temporary dataframe with existing data + new row
            temp_data = existing_data.copy()
            temp_data = pd.concat([temp_data, new_row.to_frame().T], ignore_index=True)
            
            # Select the appropriate price column based on asset type
            price_column = 'last_price' if is_option else 'close'
            
            # Initialize result series with the new row's basic data
            result = new_row.copy()
            last_idx = len(temp_data) - 1
            
            # Map indicator_periods keys to their corresponding output indicators
            indicator_mapping = {
                'rsi': ['rsi'],
                'ema': ['ema'],
                'sma': ['sma'],
                'macd_fast': ['macd_line', 'macd_signal'],
                'macd_slow': ['macd_line', 'macd_signal'],
                'macd_signal': ['macd_line', 'macd_signal'],
                'bollinger_bands': ['bollinger_upper', 'bollinger_lower', 'bollinger_bands_width'],
                'roc': ['roc'],
                'roc_of_roc': ['roc_of_roc'],
                'stoch_rsi_k': ['stoch_rsi_k'],
                'stoch_rsi_d': ['stoch_rsi_d'],
                'atr': ['atr'],
                'vwma': ['vwma'],
                'volatility': ['volatility'],
                'price_change': ['price_change']
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
                        result['rsi'] = self._calculate_single_rsi(temp_data, period_value, last_idx, price_column)
                        calculated_indicators.add('rsi')
                        
                    elif period_key == 'ema' and 'ema' in self.indicator_functions:
                        result['ema'] = self._calculate_single_ema(temp_data, period_value, last_idx, price_column)
                        calculated_indicators.add('ema')
                        
                    elif period_key == 'sma' and 'sma' in self.indicator_functions:
                        result['sma'] = self._calculate_single_sma(temp_data, period_value, last_idx, price_column)
                        calculated_indicators.add('sma')
                        
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
                        
                        # After calculating ROC, check if we need to calculate ROC of ROC
                        if 'roc_of_roc' in indicator_periods and 'roc_of_roc' in self.indicator_functions:
                            # Calculate ROC for the entire temp_data DataFrame to create the full ROC column
                            temp_data_with_roc = temp_data.copy()
                            roc_period = period_value
                            temp_data_with_roc['roc'] = ((temp_data_with_roc[price_column] - temp_data_with_roc[price_column].shift(roc_period)) / temp_data_with_roc[price_column].shift(roc_period)) * 100
                            result['roc_of_roc'] = self._calculate_single_roc_of_roc(temp_data_with_roc, indicator_periods, last_idx)
                            calculated_indicators.add('roc_of_roc')
                        
                    elif period_key == 'roc_of_roc' and 'roc_of_roc' in self.indicator_functions and 'roc_of_roc' not in calculated_indicators:
                        # ROC of ROC as standalone (in case ROC was calculated in a previous iteration)
                        result['roc_of_roc'] = self._calculate_single_roc_of_roc(temp_data, indicator_periods, last_idx)
                        calculated_indicators.add('roc_of_roc')
                        
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
                        calculated_indicators.add('vwma')
                        
                    elif period_key == 'volatility' and 'volatility' in self.indicator_functions:
                        result['volatility'] = self._calculate_single_volatility(temp_data, period_value, last_idx, price_column)
                        calculated_indicators.add('volatility')
                        
                    elif period_key == 'price_change' and 'price_change' in self.indicator_functions:
                        result['price_change'] = self._calculate_single_price_change(temp_data, last_idx, price_column)
                        calculated_indicators.add('price_change')
                        
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
                else:
                    result[key] = float('nan')
            return result
    
    # Individual indicator calculation functions for single-tick calculations
    def _calculate_single_rsi(self, temp_data: pd.DataFrame, period: int, last_idx: int, price_column: str = 'close') -> float:
        """Calculate RSI for the latest tick"""
        if len(temp_data) >= period + 1:
            delta = temp_data[price_column].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi.iloc[last_idx] if not pd.isna(rsi.iloc[last_idx]) else float('nan')
        return float('nan')
    
    def _calculate_single_ema(self, temp_data: pd.DataFrame, period: int, last_idx: int, price_column: str = 'close') -> float:
        """Calculate EMA for the latest tick"""
        if len(temp_data) > period:  # Need period+1 data points for meaningful EMA
            ema = temp_data[price_column].ewm(span=period, adjust=False).mean()
            return ema.iloc[last_idx] if not pd.isna(ema.iloc[last_idx]) else float('nan')
        return float('nan')
    
    def _calculate_single_sma(self, temp_data: pd.DataFrame, period: int, last_idx: int, price_column: str = 'close') -> float:
        """Calculate SMA for the latest tick"""
        if len(temp_data) > period:  # Need past 'period' periods + current period  
            sma = temp_data[price_column].rolling(window=period).mean()
            return sma.iloc[last_idx] if not pd.isna(sma.iloc[last_idx]) else float('nan')
        return float('nan')
    
    def _calculate_single_macd_line(self, temp_data: pd.DataFrame, periods: dict, last_idx: int, price_column: str = 'close') -> float:
        """Calculate MACD line for the latest tick"""
        fast_period = periods.get('macd_fast', 12)
        slow_period = periods.get('macd_slow', 26)
        if len(temp_data) > slow_period:  # Need past slow_period periods + current
            ema_fast = temp_data[price_column].ewm(span=fast_period, adjust=False).mean()
            ema_slow = temp_data[price_column].ewm(span=slow_period, adjust=False).mean()
            macd_line = ema_fast - ema_slow
            return macd_line.iloc[last_idx] if not pd.isna(macd_line.iloc[last_idx]) else float('nan')
        return float('nan')
    
    def _calculate_single_macd_signal(self, temp_data: pd.DataFrame, periods: dict, last_idx: int, price_column: str = 'close') -> float:
        """Calculate MACD signal for the latest tick"""
        fast_period = periods.get('macd_fast', 12)
        slow_period = periods.get('macd_slow', 26)
        signal_period = periods.get('macd_signal', 9)
        if len(temp_data) > slow_period + signal_period - 1:  # Need MACD line + signal_period of MACD history
            ema_fast = temp_data[price_column].ewm(span=fast_period, adjust=False).mean()
            ema_slow = temp_data[price_column].ewm(span=slow_period, adjust=False).mean()
            macd_line = ema_fast - ema_slow
            signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
            return signal_line.iloc[last_idx] if not pd.isna(signal_line.iloc[last_idx]) else float('nan')
        return float('nan')
    
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
        """Calculate ROC for the latest tick"""
        if len(temp_data) > period:  # Need period+1 data points to shift back 'period' positions
            roc = ((temp_data[price_column] - temp_data[price_column].shift(period)) / temp_data[price_column].shift(period)) * 100
            return roc.iloc[last_idx] if not pd.isna(roc.iloc[last_idx]) else float('nan')
        return float('nan')
    
    def _calculate_single_roc_of_roc(self, temp_data: pd.DataFrame, periods: dict, last_idx: int) -> float:
        """
        Calculate ROC of ROC for the latest tick using the existing 'roc' column.
        """
        roc_of_roc_period = periods.get('roc_of_roc', 10)
        if 'roc' in temp_data.columns and len(temp_data) > roc_of_roc_period:
            roc_of_roc = ((temp_data['roc'] - temp_data['roc'].shift(roc_of_roc_period)) / temp_data['roc'].shift(roc_of_roc_period)) * 100
            return roc_of_roc.iloc[last_idx] if not pd.isna(roc_of_roc.iloc[last_idx]) else float('nan')
        return float('nan')
    
    def _calculate_single_stoch_rsi_k(self, temp_data: pd.DataFrame, periods: dict, last_idx: int, price_column: str = 'close') -> float:
        """Calculate Stochastic RSI %K for the latest tick"""
        rsi_period = periods.get('rsi', 14)
        k_period = periods.get('stoch_rsi_k', 3)
        if len(temp_data) >= rsi_period + k_period:
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
        rsi_period = periods.get('rsi', 14)
        k_period = periods.get('stoch_rsi_k', 3)
        d_period = periods.get('stoch_rsi_d', 3)
        if len(temp_data) >= rsi_period + k_period:
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
    
    def _calculate_single_atr(self, temp_data: pd.DataFrame, period: int, last_idx: int, price_column: str = 'close') -> float:
        """Calculate ATR for the latest tick"""
        if len(temp_data) >= period and all(col in temp_data.columns for col in ['high', 'low', price_column]):
            temp_data_copy = temp_data.copy()
            temp_data_copy['prev_close'] = temp_data_copy[price_column].shift(1)
            temp_data_copy['tr1'] = temp_data_copy['high'] - temp_data_copy['low']
            temp_data_copy['tr2'] = abs(temp_data_copy['high'] - temp_data_copy['prev_close'])
            temp_data_copy['tr3'] = abs(temp_data_copy['low'] - temp_data_copy['prev_close'])
            temp_data_copy['true_range'] = temp_data_copy[['tr1', 'tr2', 'tr3']].max(axis=1)
            atr = temp_data_copy['true_range'].rolling(window=period).mean()
            return atr.iloc[last_idx] if not pd.isna(atr.iloc[last_idx]) else float('nan')
        return float('nan')
    
    def _calculate_single_vwma(self, temp_data: pd.DataFrame, period: int, last_idx: int, price_column: str = 'close') -> float:
        """Calculate VWMA for the latest tick"""
        has_enough_data = len(temp_data) > period  # Need past 'period' periods + current period
        has_volume_column = 'volume' in temp_data.columns
        

        
        if has_enough_data and has_volume_column:
            # For options, use last_price; for others, use typical price if high/low are available
            if price_column == 'last_price':  # Options data
                typical_price = temp_data[price_column]
            elif all(col in temp_data.columns for col in ['high', 'low']):
                typical_price = (temp_data['high'] + temp_data['low'] + temp_data[price_column]) / 3
            else:
                typical_price = temp_data[price_column]
            volume = temp_data['volume'].fillna(1)
            

            
            vwma = (typical_price * volume).rolling(window=period).sum() / volume.rolling(window=period).sum()
            result = vwma.iloc[last_idx] if not pd.isna(vwma.iloc[last_idx]) else float('nan')
            
            return result
        return float('nan')
    
    def _calculate_single_volatility(self, temp_data: pd.DataFrame, period: int, last_idx: int, price_column: str = 'close') -> float:
        """Calculate volatility for the latest tick"""
        if len(temp_data) > period:  # Need past 'period' periods + current period
            volatility = temp_data[price_column].rolling(window=period).std()
            return volatility.iloc[last_idx] if not pd.isna(volatility.iloc[last_idx]) else float('nan')
        return float('nan')
    
    def _calculate_single_price_change(self, temp_data: pd.DataFrame, last_idx: int, price_column: str = 'close') -> float:
        """Calculate price change for the latest tick"""
        if len(temp_data) >= 2:
            price_change = temp_data[price_column].pct_change()
            return price_change.iloc[last_idx] if not pd.isna(price_change.iloc[last_idx]) else float('nan')
        return float('nan')