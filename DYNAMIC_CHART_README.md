# Dynamic Chart Analyzer Feature

## Overview

The Dynamic Chart Analyzer is a new frontend feature that simulates how a human trader analyzes charts in real-time. It creates an engaging visual experience showing the LLM "thinking" while analyzing market data across multiple timeframes.

## Features

- **Automatic Timeframe Switching**: Cycles through M1, M5, M15, M30, H1, and H4 timeframes
- **Dynamic Chart Annotations**: Draws support/resistance lines and trend lines progressively
- **Zoom Effects**: Simulates zooming in/out to examine price action details
- **Thought Process Display**: Shows what the "AI is thinking" with a bank of market analysis phrases
- **Progress Tracking**: Visual progress bar showing analysis completion
- **Smooth Animations**: CSS animations for transitions and effects

## Files Created

1. **dynamic_chart_analyzer.py** - Main module containing the feature implementation
2. **integration_guide.py** - Step-by-step integration instructions
3. **app_integration_example.py** - Complete example of integration
4. **test_dynamic_chart.py** - Standalone test script

## Quick Start

### 1. Test the Feature Independently

```bash
python test_dynamic_chart.py
```

This will start a simple app on http://localhost:8050 to test the dynamic chart analyzer.

### 2. Integrate into Your Main App

#### Step 1: Import the module
Add to your imports in app.py:
```python
from dynamic_chart_analyzer import create_dynamic_chart_component, register_dynamic_chart_callbacks, dynamic_chart_css
```

#### Step 2: Add the CSS
After your existing `app_css` definition:
```python
app_css += dynamic_chart_css
```

#### Step 3: Create the component
Before defining your layout:
```python
dynamic_chart_component, dynamic_analyzer = create_dynamic_chart_component()
```

#### Step 4: Add to layout
Insert `dynamic_chart_component` where you want it to appear in your layout.

#### Step 5: Register callbacks
After defining your layout:
```python
register_dynamic_chart_callbacks(app, dynamic_analyzer)
```

## Customization

### Change Analysis Speed
In `dynamic_chart_analyzer.py`, modify the interval:
```python
interval=2000,  # Change to desired milliseconds
```

### Customize Analysis Phrases
Edit the `ANALYSIS_THOUGHTS` dictionary in `dynamic_chart_analyzer.py` to add your own phrases.

### Modify Chart Appearance
Edit the `create_base_chart` method to change colors, indicators, or layout.

### Add More Indicators
In the `create_base_chart` method, add additional traces for indicators like:
- Bollinger Bands
- RSI overlay
- MACD
- Volume bars

## How It Works

1. **Analysis Cycle**: The component goes through a 25-step analysis cycle
2. **Steps 0-5**: Initial trend analysis with thought display
3. **Steps 5-10**: Progressive drawing of support/resistance lines
4. **Steps 10-15**: Timeframe switching to get broader perspective
5. **Steps 15-20**: Zoom analysis on recent price action
6. **Steps 20-25**: Final analysis and conclusion
7. **Reset**: After step 25, the cycle restarts

## Troubleshooting

### Chart doesn't appear
- Verify MetaTrader5 is connected
- Check that the symbol exists in MT5
- Ensure all callbacks are registered

### Animation not working
- Check that `bot-state` store exists
- Verify interval component is not disabled
- Check browser console for errors

### No data showing
- Confirm MT5 is returning data for the symbol
- Check that timeframe mapping is correct
- Verify MT5 permissions

## Performance Notes

- The feature updates every 2 seconds by default
- Each update makes one MT5 data request
- CSS animations are GPU-accelerated for smooth performance
- Consider increasing interval if performance is an issue

## Future Enhancements

Potential improvements:
- Add voice narration of analysis thoughts
- Include more complex chart patterns detection
- Add multi-asset comparison view
- Implement user-adjustable analysis speed
- Add export of analysis screenshots

## Support

For issues or questions about this feature:
1. Check the integration guide first
2. Run the test script to isolate issues
3. Check browser console for JavaScript errors
4. Verify all component IDs are unique
