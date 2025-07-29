"""
Technical Indicators Module for MetaTrader5 LLM Trading Bot
-----------------------------------------------------------
This module contains functions for calculating technical indicators
used in market analysis.
"""

import numpy as np
import pandas as pd
from scipy.stats import entropy

def calculate_ema(series, period=9):
    """
    Calculate Exponential Moving Average (EMA)
    
    Args:
        series (pd.Series): Price series
        period (int): EMA period
        
    Returns:
        pd.Series: EMA values
    """
    return series.ewm(span=period, adjust=False).mean()

def calculate_atr(df, period=14):
    """
    Calculate Average True Range (ATR)
    
    Args:
        df (pd.DataFrame): OHLC dataframe with 'high', 'low', 'close' columns
        period (int): ATR period
        
    Returns:
        pd.Series: ATR values
    """
    high = df['high']
    low = df['low']
    close = df['close'].shift(1)
    
    # Handle first row where previous close is NaN
    close.iloc[0] = df['open'].iloc[0]
    
    # Calculate true range
    tr1 = high - low
    tr2 = abs(high - close)
    tr3 = abs(low - close)
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Calculate ATR
    atr = tr.rolling(window=period).mean()
    
    return atr

def calculate_directional_entropy(df, period=14, normalize=True):
    """
    Calculate directional entropy using close price movements
    
    Args:
        df (pd.DataFrame): DataFrame with OHLC data
        period (int): Period for entropy calculation
        normalize (bool): Whether to normalize entropy between 0 and 1
        
    Returns:
        pd.Series: Directional entropy values
    """
    # Calculate price changes
    price_changes = df['close'].diff()
    
    # Initialize entropy series
    entropy_series = pd.Series(index=df.index)
    
    # Calculate entropy for each window
    for i in range(period, len(df)):
        # Get window of price changes
        window = price_changes.iloc[i-period+1:i+1]
        
        # Count positive, negative, and zero changes
        pos_count = sum(window > 0)
        neg_count = sum(window < 0)
        zero_count = sum(window == 0)
        
        # Calculate probabilities
        total = pos_count + neg_count + zero_count
        p_pos = pos_count / total if total > 0 else 0
        p_neg = neg_count / total if total > 0 else 0
        p_zero = zero_count / total if total > 0 else 0
        
        # Calculate entropy (-sum(p * log(p)))
        h = 0
        for p in [p_pos, p_neg, p_zero]:
            if p > 0:
                h -= p * np.log2(p)
        
        entropy_series.iloc[i] = h
    
    # Normalize entropy between 0 and 1 if requested
    if normalize:
        # Maximum entropy for 3 possibilities (positive, negative, zero) is log2(3) ≈ 1.585
        max_entropy = np.log2(3)
        entropy_series = entropy_series / max_entropy
    
    return entropy_series

def calculate_volume_profile(df, price_levels=10):
    """
    Calculate Volume Profile
    
    Volume profile shows the distribution of volume across price levels.
    
    Args:
        df (pd.DataFrame): OHLC dataframe with 'close' and 'tick_volume' columns
        price_levels (int): Number of price levels
        
    Returns:
        dict: Volume profile as a dictionary of price levels and volumes
    """
    min_price = df['low'].min()
    max_price = df['high'].max()
    
    # Create price bins
    price_bins = np.linspace(min_price, max_price, price_levels+1)
    
    # Initialize volume profile
    volume_profile = {f"{price_bins[i]:.5f}-{price_bins[i+1]:.5f}": 0 for i in range(price_levels)}
    
    # Distribute volume across price bins
    for i in range(len(df)):
        # Approximate volume distribution across the price range of the candle
        candle_low = df['low'].iloc[i]
        candle_high = df['high'].iloc[i]
        candle_volume = df['tick_volume'].iloc[i]
        
        # Find which bins this candle spans
        for j in range(price_levels):
            bin_low = price_bins[j]
            bin_high = price_bins[j+1]
            
            # Check for overlap between candle and bin
            overlap_low = max(candle_low, bin_low)
            overlap_high = min(candle_high, bin_high)
            
            if overlap_high > overlap_low:
                # Calculate proportion of candle in this bin
                candle_range = candle_high - candle_low
                if candle_range > 0:
                    overlap_ratio = (overlap_high - overlap_low) / candle_range
                    volume_profile[f"{bin_low:.5f}-{bin_high:.5f}"] += candle_volume * overlap_ratio
    
    return volume_profile

def calculate_slope(series, period=14):
    """
    Calculate the slope (angle of the line) for a price or indicator series.
    The slope measures the rate of change over the specified period.
    
    Args:
        series (pd.Series): Price or indicator series
        period (int): Lookback period for slope calculation
        
    Returns:
        pd.Series: Slope values for each point in the series
    """
    import numpy as np
    import pandas as pd
    
    # Initialize slope series with same index as input
    slope_series = pd.Series(index=series.index, dtype=float)
    slope_series[:] = np.nan
    
    # Calculate slope for each point starting from period
    for i in range(period, len(series)):
        # Get the window of data
        window = series.iloc[i-period+1:i+1]
        
        # Create X values (0, 1, 2, ..., period-1)
        x = np.arange(period)
        
        # Get Y values (price/indicator)
        y = window.values
        
        # Calculate slope using linear regression formula
        # slope = (n*sum(x*y) - sum(x)*sum(y)) / (n*sum(x^2) - (sum(x))^2)
        n = period
        sum_x = np.sum(x)
        sum_y = np.sum(y)
        sum_xy = np.sum(x * y)
        sum_xx = np.sum(x * x)
        
        # Calculate slope
        denominator = (n * sum_xx - sum_x * sum_x)
        if denominator != 0:  # Avoid division by zero
            slope = (n * sum_xy - sum_x * sum_y) / denominator
            slope_series.iloc[i] = slope
    
    return slope_series

def detect_support_resistance(df, window=20, threshold=0.01):
    """
    Detect Support and Resistance levels
    
    Args:
        df (pd.DataFrame): OHLC dataframe
        window (int): Lookback window
        threshold (float): Price threshold as a percentage
        
    Returns:
        tuple: Lists of support and resistance levels
    """
    supports = []
    resistances = []
    
    # Identify local minima and maxima
    for i in range(window, len(df) - window):
        # Current price
        current_low = df['low'].iloc[i]
        current_high = df['high'].iloc[i]
        
        # Check if this is a local minimum (support)
        is_support = True
        for j in range(i - window, i + window + 1):
            if j != i and df['low'].iloc[j] < current_low:
                is_support = False
                break
                
        # Check if this is a local maximum (resistance)
        is_resistance = True
        for j in range(i - window, i + window + 1):
            if j != i and df['high'].iloc[j] > current_high:
                is_resistance = False
                break
        
        # Add to support/resistance lists if criteria are met
        if is_support:
            # Check if this level is already in our list
            if not any(abs(current_low - s) / s < threshold for s in supports):
                supports.append(current_low)
                
        if is_resistance:
            # Check if this level is already in our list
            if not any(abs(current_high - r) / r < threshold for r in resistances):
                resistances.append(current_high)
    
    return sorted(supports), sorted(resistances)