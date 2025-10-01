import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import yfinance as yf
import talib as ta
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import os
import mplfinance as mpf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')  # Suppress warnings

# Constants
SYMBOL = "NG=F"  # Natural Gas Futures
START_DATE = "2025-08-04"  # August 1, 2025
END_DATE = "2025-08-08"    # August 10, 2025 (current date)
INTERVAL = "30m"           # 30-minute candles
WINDOW_SIZE = 10           # Same window size as in training
MODELS_DIR = r"C:\Users\sousa\Documents\DataH\NinjatradeV2\models\indicators"
RESULTS_DIR = r"C:\Users\sousa\Documents\DataH\NinjatradeV2\results\validation"

# Create results directory if it doesn't exist
os.makedirs(RESULTS_DIR, exist_ok=True)

# Helper Functions
def calculate_slope(series, period=14):
    """Calculate slope using linear regression"""
    slopes = []
    for i in range(len(series)):
        if i < period:
            slopes.append(np.nan)
        else:
            y = series.iloc[i-period:i].values
            x = np.arange(period)
            slope, _ = np.polyfit(x, y, 1)
            slopes.append(slope)
    return pd.Series(slopes, index=series.index)

def calculate_directional_entropy(series, period=14):
    """Calculate directional entropy of a time series"""
    entropy = []
    for i in range(len(series)):
        if i < period:
            entropy.append(np.nan)
        else:
            # Calculate price changes
            changes = np.diff(series.iloc[i-period:i].values)
            # Convert to directions: up (1), down (-1) or sideways (0)
            directions = np.sign(changes)
            # Count occurrences of each direction
            unique, counts = np.unique(directions, return_counts=True)
            # Calculate probabilities
            probabilities = counts / np.sum(counts)
            # Calculate entropy
            h = -np.sum(probabilities * np.log2(probabilities + 1e-10))
            # Normalize entropy
            if len(unique) > 0:  # Add check to avoid divide by zero
                h_norm = h / np.log2(max(len(unique), 1))
            else:
                h_norm = 0
            entropy.append(h_norm)
    return pd.Series(entropy, index=series.index)

def download_m30_data(symbol, start_date, end_date):
    """Download 30-minute candle data for the specified period"""
    print(f"Downloading {symbol} 30-minute data from {start_date} to {end_date}...")
    
    try:
        # Download data
        data = yf.download(symbol, start=start_date, end=end_date, interval=INTERVAL)
        
        if len(data) == 0:
            print("No data was downloaded. The symbol might be incorrect or data unavailable.")
            return None
        
        print(f"Downloaded {len(data)} candles")
        print("Data structure:", data.columns)
        print(data.head())
        
        # Fix Yahoo Finance multi-index columns
        if isinstance(data.columns, pd.MultiIndex):
            print("Fixing multi-index columns...")
            # If we have a multi-index DataFrame (common with Yahoo Finance)
            # Example: ('Open', 'NG=F') -> 'Open'
            if data.columns.nlevels > 1:
                # Get just the first level which contains 'Open', 'High', etc.
                new_columns = data.columns.get_level_values(0)
                data.columns = new_columns
                print("New columns:", data.columns)
        
        # Verify required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_cols:
            if col not in data.columns:
                print(f"Error: Required column '{col}' not found in data")
                return None
                
        # Show the processed data
        print("Processed data sample:")
        print(data.head())
        
        return data
        
    except Exception as e:
        print(f"Error downloading data: {e}")
        return None

def process_data(data):
    """Process data and calculate all required indicators"""
    print("Processing data and calculating indicators...")
    
    # Create a copy
    df = data.copy()
    
    # Verify we have the required columns
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in required_cols:
        if col not in df.columns:
            print(f"Error: Required column '{col}' missing from data")
            return None
    
    # Basic price features
    df['HL_Pct'] = (df['High'] - df['Low']) / df['Close'] * 100.0
    df['Daily_Return'] = df['Close'].pct_change() * 100.0
    
    # 1. Slope (using linear regression on Close prices)
    df['Slope'] = calculate_slope(df['Close'], period=14)
    
    # 2. ATR (Average True Range) - volatility measure
    try:
        high, low, close = df['High'].values, df['Low'].values, df['Close'].values
        df['ATR'] = ta.ATR(high, low, close, timeperiod=14)
    except Exception as e:
        print(f"Error calculating ATR with talib: {e}")
        # Alternative calculation
        true_range = np.maximum(
            df['High'] - df['Low'],
            np.maximum(
                np.abs(df['High'] - df['Close'].shift(1)),
                np.abs(df['Low'] - df['Close'].shift(1))
            )
        )
        df['ATR'] = true_range.rolling(window=14).mean()
    
    # 3. Directional Entropy
    df['DirectionalEntropy'] = calculate_directional_entropy(df['Close'], period=14)
    
    # 4. EMA14
    try:
        df['EMA14'] = ta.EMA(df['Close'].values, timeperiod=14)
    except Exception as e:
        print(f"Error calculating EMA with talib: {e}")
        df['EMA14'] = df['Close'].ewm(span=14, adjust=False).mean()
    
    # Additional indicators
    df['EMA5'] = df['Close'].ewm(span=5, adjust=False).mean()
    df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA5_EMA20_Cross'] = ((df['EMA5'] > df['EMA20']).astype(int) - 
                             (df['EMA5'].shift(1) > df['EMA20'].shift(1)).astype(int))
    
    # RSI
    try:
        df['RSI'] = ta.RSI(df['Close'].values, timeperiod=14)
    except Exception as e:
        print(f"Error calculating RSI with talib: {e}")
        # Manual calculation
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / (avg_loss + 1e-10)  # Avoid division by zero
        df['RSI'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    try:
        df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = ta.BBANDS(
            df['Close'].values, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
    except Exception as e:
        print(f"Error calculating Bollinger Bands with talib: {e}")
        # Manual calculation
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        df['BB_Std'] = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (df['BB_Std'] * 2)
        df['BB_Lower'] = df['BB_Middle'] - (df['BB_Std'] * 2)
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
    
    # Create target variables for validation
    # These represent the actual changes that happened in each indicator
    for indicator in ['Slope', 'ATR', 'DirectionalEntropy', 'EMA14']:
        df[f'{indicator}_Next'] = df[indicator].shift(-1)
        
        if indicator == 'Slope':
            # For Slope, we're interested in the raw change
            df[f'{indicator}_Actual_Change'] = df[f'{indicator}_Next'] - df[indicator]
            df[f'{indicator}_Actual_Direction'] = (df[f'{indicator}_Actual_Change'] > 0).astype(int)
        else:
            # For other indicators, we use percentage change
            # Avoid division by zero
            change = np.where(df[indicator] != 0, 
                             ((df[f'{indicator}_Next'] - df[indicator]) / df[indicator]) * 100,
                             0)
            df[f'{indicator}_Actual_Change'] = change
            df[f'{indicator}_Actual_Direction'] = (df[f'{indicator}_Actual_Change'] > 0).astype(int)
    
    # Handle NaN values - instead of dropping, fill with appropriate values
    # This is critical to avoid losing all data
    df = df.fillna(method='ffill').fillna(method='bfill')
    
    # If we still have NaNs (at the beginning), drop those rows
    df = df.dropna()
    
    print(f"Data processing complete. Shape: {df.shape}")
    return df

def load_all_models():
    """Load all trained models and their configurations"""
    models_info = {}
    indicators = ['Slope', 'ATR', 'DirectionalEntropy', 'EMA14']
    
    for indicator in indicators:
        model_path = os.path.join(MODELS_DIR, f"{indicator}_model.h5")
        scaler_path = os.path.join(MODELS_DIR, f"{indicator}_scaler.pkl")
        config_path = os.path.join(MODELS_DIR, f"{indicator}_config.pkl")
        
        if not os.path.exists(model_path):
            print(f"Error: Model file for {indicator} not found at {model_path}")
            continue
            
        if not os.path.exists(scaler_path):
            print(f"Error: Scaler file for {indicator} not found at {scaler_path}")
            continue
            
        if not os.path.exists(config_path):
            print(f"Error: Config file for {indicator} not found at {config_path}")
            continue
        
        try:
            model = load_model(model_path)
            scaler = joblib.load(scaler_path)
            config = joblib.load(config_path)
            
            # Extract feature columns from config, with fallback
            if 'feature_cols' in config:
                feature_cols = config['feature_cols']
            else:
                # Use default feature columns if not in config
                print(f"No feature_cols in config for {indicator}, using defaults")
                base_features = ['Open', 'High', 'Low', 'Close', 'Volume', 'HL_Pct', 'Daily_Return']
                feature_cols = base_features + ['Slope', 'ATR', 'DirectionalEntropy', 'EMA14', 
                                            'EMA5', 'EMA20', 'EMA5_EMA20_Cross', 'RSI', 
                                            'BB_Upper', 'BB_Middle', 'BB_Lower', 'BB_Width']
            
            models_info[indicator] = {
                'model': model,
                'scaler': scaler,
                'feature_cols': feature_cols,
                'window_size': config.get('window_size', WINDOW_SIZE)
            }
            
            print(f"Successfully loaded model for {indicator}")
        except Exception as e:
            print(f"Error loading model for {indicator}: {e}")
    
    return models_info

def make_predictions(data, models_info):
    """Make predictions for each candle and evaluate accuracy"""
    print("Making predictions for each candle...")
    
    # Check if we have enough data
    if len(data) <= WINDOW_SIZE:
        print(f"Not enough data for prediction. Need at least {WINDOW_SIZE+1} candles, got {len(data)}.")
        return {}, pd.DataFrame()
    
    # Store all predictions
    all_predictions = {}
    
    # Validation data starts after we have enough candles for the window
    start_idx = WINDOW_SIZE
    end_idx = len(data) - 1  # Last candle doesn't have a "next" value to validate
    
    # Create a list to store all prediction results
    results = []
    
    for i in range(start_idx, end_idx):
        candle_time = data.index[i]
        candle_data = data.iloc[:i+1]  # Data up to and including current candle
        
        # Store predictions for this candle
        candle_predictions = {}
        
        for indicator in models_info:
            model = models_info[indicator]['model']
            scaler = models_info[indicator]['scaler']
            feature_cols = models_info[indicator]['feature_cols']
            window_size = models_info[indicator]['window_size']
            
            # Check if all feature columns are in the data
            missing_cols = [col for col in feature_cols if col not in candle_data.columns]
            if missing_cols:
                print(f"Missing columns for {indicator}: {missing_cols}")
                continue
            
            # Get data for prediction
            if len(candle_data) < window_size:
                continue  # Not enough data for this window
                
            recent_data = candle_data[feature_cols].values[-window_size:]
            
            # Normalize
            try:
                recent_data_scaled = scaler.transform(recent_data)
            except Exception as e:
                print(f"Error scaling data for {indicator}: {e}")
                continue
            
            # Prepare for prediction
            X_pred = recent_data_scaled.reshape(1, window_size, len(feature_cols))
            
            # Make prediction
            prediction = model.predict(X_pred, verbose=0)[0][0]
            binary_prediction = int(prediction > 0.5)
            
            # Get the actual result (for validation)
            actual_direction = data.iloc[i+1][f'{indicator}_Actual_Direction']
            actual_change = data.iloc[i+1][f'{indicator}_Actual_Change']
            
            # Store results
            candle_predictions[indicator] = {
                'probability': float(prediction),
                'binary_prediction': binary_prediction,
                'confidence': max(prediction, 1-prediction),
                'actual_direction': actual_direction,
                'actual_change': float(actual_change),
                'correct': binary_prediction == actual_direction
            }
            
            # Add to results list
            results.append({
                'candle_time': candle_time,
                'indicator': indicator,
                'prediction': binary_prediction,
                'actual': actual_direction,
                'confidence': max(prediction, 1-prediction),
                'correct': binary_prediction == actual_direction,
                'actual_change': float(actual_change)
            })
        
        all_predictions[candle_time] = candle_predictions
    
    # Create DataFrame from results list
    if results:
        results_df = pd.DataFrame(results)
        
        # Calculate accuracy
        if len(results_df) > 0:
            accuracy_by_indicator = results_df.groupby('indicator')['correct'].mean()
            print("\nPrediction Accuracy by Indicator:")
            for indicator, accuracy in accuracy_by_indicator.items():
                print(f"{indicator}: {accuracy:.4f} ({int(accuracy*100)}%)")
            
            overall_accuracy = results_df['correct'].mean()
            print(f"Overall Accuracy: {overall_accuracy:.4f} ({int(overall_accuracy*100)}%)")
        else:
            print("No prediction results to analyze.")
    else:
        results_df = pd.DataFrame()
        print("No prediction results generated.")
    
    return all_predictions, results_df

def visualize_fixed_chart(data, results_df):
    """Create visualizations with fixed x-axis scaling"""
    if results_df.empty:
        print("No results to visualize.")
        return
        
    print("Creating improved visualizations...")
    
    # For visual clarity, let's use just the last 40 candles (or all if less than 40)
    last_n_candles = min(40, len(data))
    plot_data = data.iloc[-last_n_candles:]
    
    # Filter results to match the plot data
    plot_results = results_df[results_df['candle_time'].isin(plot_data.index)]
    
    if plot_results.empty:
        print("No results to plot on candlestick chart.")
        return
    
    # Create a dictionary to store predictions for easy lookup
    predictions_dict = {}
    for _, row in plot_results.iterrows():
        candle_time = row['candle_time']
        indicator = row['indicator']
        if candle_time not in predictions_dict:
            predictions_dict[candle_time] = {}
        predictions_dict[candle_time][indicator] = {
            'prediction': row['prediction'],
            'correct': row['correct'],
            'confidence': row['confidence'],
            'actual_change': row['actual_change']
        }
    
    # Set up colors for indicators (for consistent coloring across charts)
    indicator_colors = {
        'Slope': 'blue',
        'ATR': 'purple',
        'DirectionalEntropy': 'orange',
        'EMA14': 'red'
    }
    
    # 1. Create individual indicator value plots
    plt.figure(figsize=(14, 16))
    indicators = ['Slope', 'ATR', 'DirectionalEntropy', 'EMA14']
    
    for i, indicator in enumerate(indicators):
        plt.subplot(4, 1, i+1)
        plt.plot(plot_data.index, plot_data[indicator], 
                 label=indicator, color=indicator_colors[indicator])
        plt.title(f'{indicator} Values')
        plt.grid(True, alpha=0.3)
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'indicator_values.png'))
    plt.close()
    
    # 2. Create direct mplfinance visualization with properly formatted date index
    try:
        # Collect overall indicator signals for the title
        indicator_predictions = []
        for indicator in indicators:
            bullish_count = 0
            bearish_count = 0
            total_count = 0
            
            for candle_time in plot_data.index[:-1]:
                if candle_time in predictions_dict and indicator in predictions_dict[candle_time]:
                    pred = predictions_dict[candle_time][indicator]
                    total_count += 1
                    if pred['prediction'] == 1:  # Bullish
                        bullish_count += 1
                    else:  # Bearish
                        bearish_count += 1
            
            if total_count > 0:
                bullish_pct = (bullish_count / total_count) * 100
                bearish_pct = (bearish_count / total_count) * 100
                
                if bullish_pct > bearish_pct:
                    direction = "↑"
                else:
                    direction = "↓"
                
                indicator_predictions.append(f"{indicator} {direction}")
        
        # Create subplots for the candlestick chart and indicator plots
        fig = plt.figure(figsize=(16, 10))
        gs = plt.GridSpec(5, 1, height_ratios=[3, 1, 1, 1, 1], hspace=0.3)
        
        # Main price chart
        ax1 = fig.add_subplot(gs[0])
        
        # Create arrays for the OHLC data
        dates = np.arange(len(plot_data))
        opens = plot_data['Open'].values
        highs = plot_data['High'].values
        lows = plot_data['Low'].values
        closes = plot_data['Close'].values
        
        # Calculate spacing for arrows
        price_range = max(highs) - min(lows)
        arrow_spacing = price_range * 0.1  # Vertical spacing between arrows
        
        # Plot candlesticks manually
        width = 0.6  # width of the candlesticks
        for i in range(len(plot_data)):
            # Determine if this is an up or down candle
            if closes[i] >= opens[i]:
                color = 'green'
                body_bottom = opens[i]
                body_height = closes[i] - opens[i]
            else:
                color = 'red'
                body_bottom = closes[i]
                body_height = opens[i] - closes[i]
            
            # Plot candle body
            ax1.add_patch(plt.Rectangle((dates[i] - width/2, body_bottom), width, body_height, 
                                      fill=True, color=color))
            
            # Plot candle wicks
            ax1.plot([dates[i], dates[i]], [lows[i], highs[i]], color='black', linewidth=1)
            
            # Add prediction arrows above candles
            if i < len(plot_data) - 1:  # Skip last candle (no prediction for the next candle)
                candle_time = plot_data.index[i]
                
                if candle_time in predictions_dict:
                    # Calculate position for arrows
                    arrow_y = highs[i] + arrow_spacing
                    
                    # Add arrows for each indicator
                    for idx, indicator in enumerate(indicators):
                        if indicator in predictions_dict[candle_time]:
                            pred = predictions_dict[candle_time][indicator]
                            
                            # Set position, with spacing between indicators
                            indicator_y = arrow_y + (idx * arrow_spacing * 2)
                            
                            # Set color and marker based on prediction and correctness
                            if pred['prediction'] == 1:  # Bullish
                                marker = '^'
                                color = 'green' if pred['correct'] else 'lightgreen'
                                confidence_pct = int(pred['confidence'] * 100)
                            else:  # Bearish
                                marker = 'v'
                                color = 'red' if pred['correct'] else 'salmon'
                                confidence_pct = int(pred['confidence'] * 100)
                            

                            
                            ax1.annotate(f"{confidence_pct}%", (dates[i], indicator_y), 
                                    fontsize=10, color=color, weight='bold',
                                    ha='center', va='center',
                                    bbox=dict(boxstyle="round,pad=0.3", fc=color, alpha=0.3))
        
        # Set x-axis tick labels to show dates
        # Format dates for better readability
        date_labels = [d.strftime('%m-%d %H:%M') for d in plot_data.index]
        
        # Show date labels at regular intervals to prevent overcrowding
        n_dates = len(date_labels)
        step = max(1, n_dates // 10)  # Show about 10 labels
        
        ax1.set_xticks(np.arange(0, n_dates, step))
        ax1.set_xticklabels([date_labels[i] for i in range(0, n_dates, step)], rotation=45)
        
        # Set x and y limits to ensure proper scaling
        ax1.set_xlim(-1, len(plot_data))
        ax1.set_ylim(min(lows) * 0.998, max(highs) * 1.01 + (6 * arrow_spacing))
        
        # Add title
        title_text = f"{SYMBOL} Candlestick with Indicator Predictions\n"
        title_text += " | ".join(indicator_predictions)
        ax1.set_title(title_text)
        
        # Add legend
        handles = [
            plt.Rectangle((0, 0), 1, 1, color='green', alpha=0.7, label='Bullish (Correct)'),
            plt.Rectangle((0, 0), 1, 1, color='lightgreen', alpha=0.7, label='Bullish (Wrong)'),
            plt.Rectangle((0, 0), 1, 1, color='red', alpha=0.7, label='Bearish (Correct)'),
            plt.Rectangle((0, 0), 1, 1, color='salmon', alpha=0.7, label='Bearish (Wrong)')
        ]
        # Add indicator colors to legend
        for indicator in indicators:
            handles.append(
                plt.Line2D([0], [0], color=indicator_colors[indicator], 
                          lw=2, label=indicator)
            )
        
        ax1.legend(handles=handles, loc='upper left')
        
        # Add indicator subplots
        ax_indicators = []
        for i, indicator in enumerate(indicators):
            ax = fig.add_subplot(gs[i+1], sharex=ax1)
            ax.plot(dates, plot_data[indicator].values, color=indicator_colors[indicator])
            ax.set_ylabel(indicator)
            ax.grid(True, alpha=0.3)
            
            # Only show x-axis tick labels on the bottom subplot
            if i < len(indicators) - 1:
                plt.setp(ax.get_xticklabels(), visible=False)
            
            # Set proper x-limits
            ax.set_xlim(-1, len(plot_data))
            
            ax_indicators.append(ax)
        
        # Format bottom subplot x-axis
        ax_indicators[-1].set_xticks(np.arange(0, n_dates, step))
        ax_indicators[-1].set_xticklabels([date_labels[i] for i in range(0, n_dates, step)], rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'candlestick_with_arrows.png'))
        plt.close('all')
        
        # 3. Create an additional chart with indicator signals and their percentage confidence
        plt.figure(figsize=(12, 8))
        
        # Create a bar chart showing prediction distribution for each indicator
        indicators_data = {}
        for indicator in indicators:
            bullish_correct = 0
            bullish_wrong = 0
            bearish_correct = 0
            bearish_wrong = 0
            total = 0
            
            for _, row in plot_results[plot_results['indicator'] == indicator].iterrows():
                total += 1
                if row['prediction'] == 1:  # Bullish
                    if row['correct']:
                        bullish_correct += 1
                    else:
                        bullish_wrong += 1
                else:  # Bearish
                    if row['correct']:
                        bearish_correct += 1
                    else:
                        bearish_wrong += 1
            
            if total > 0:
                indicators_data[indicator] = {
                    'Bullish (Correct)': bullish_correct / total * 100,
                    'Bullish (Wrong)': bullish_wrong / total * 100,
                    'Bearish (Correct)': bearish_correct / total * 100,
                    'Bearish (Wrong)': bearish_wrong / total * 100
                }
        
        # Convert to DataFrame for plotting
        signal_df = pd.DataFrame(indicators_data).T
        
        # Plot
        ax = signal_df.plot(kind='barh', stacked=True, figsize=(10, 6), 
                          color=['green', 'lightgreen', 'red', 'salmon'])
        
        # Add percentage labels
        for i, indicator in enumerate(signal_df.index):
            bullish_total = signal_df.loc[indicator, 'Bullish (Correct)'] + signal_df.loc[indicator, 'Bullish (Wrong)']
            bearish_total = signal_df.loc[indicator, 'Bearish (Correct)'] + signal_df.loc[indicator, 'Bearish (Wrong)']
            
            correct_total = signal_df.loc[indicator, 'Bullish (Correct)'] + signal_df.loc[indicator, 'Bearish (Correct)']
            
            # Add total bullish/bearish percentages
            if bullish_total > 0:
                ax.text(bullish_total/2, i, f"{bullish_total:.0f}% Bullish", 
                       ha='center', va='center', color='black', fontweight='bold')
            
            if bearish_total > 0:
                ax.text(100 - bearish_total/2, i, f"{bearish_total:.0f}% Bearish", 
                       ha='center', va='center', color='black', fontweight='bold')
            
            # Add accuracy percentage
            ax.text(105, i, f"Accuracy: {correct_total:.0f}%", 
                   ha='left', va='center', color='blue')
        
        plt.title(f'Indicator Signals Distribution ({SYMBOL})')
        plt.xlabel('Percentage')
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'indicator_signals.png'))
        plt.close('all')
        
    except Exception as e:
        print(f"Error creating visualization: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"All visualizations saved to {RESULTS_DIR}")

def main():
    """Main function to run the validation"""
    print(f"Starting validation for {SYMBOL} from {START_DATE} to {END_DATE}")
    
    # Download data
    data = download_m30_data(SYMBOL, START_DATE, END_DATE)
    if data is None:
        return
    
    # Process data
    processed_data = process_data(data)
    if processed_data is None or len(processed_data) == 0:
        print("Error: No data after processing. Check your data source and processing steps.")
        return
    
    # Save processed data
    processed_data.to_csv(os.path.join(RESULTS_DIR, f"{SYMBOL}_processed_data.csv"))
    
    # Load models
    models_info = load_all_models()
    if not models_info:
        print("No models could be loaded. Exiting.")
        return
    
    # Make predictions
    all_predictions, results_df = make_predictions(processed_data, models_info)
    
    # Save results
    if not results_df.empty:
        results_df.to_csv(os.path.join(RESULTS_DIR, "prediction_results.csv"))
        
        # Create visualizations with fixed x-axis
        visualize_fixed_chart(processed_data, results_df)
    
    print("Validation complete!")

if __name__ == "__main__":
    main()