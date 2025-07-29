"""
Dynamic Chart Analyzer - Simulates real-time human-like chart analysis
This module creates a dynamic visualization that mimics how a human trader analyzes charts
"""

import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
import random
import time
from datetime import datetime, timedelta
import json
import MetaTrader5 as mt5

# Analysis response bank for dynamic thoughts
# Analysis response bank for dynamic thoughts - Enhanced and Intelligent
ANALYSIS_THOUGHTS = {
    "trend_analysis": [
        "Analyzing trend structure on {timeframe} - examining multi-timeframe momentum alignment and price action momentum...",
        "Cross-referencing momentum indicators across multiple timeframes to identify confluence zones and divergence patterns...",
        "Evaluating price action patterns and candlestick formations to determine institutional order flow and market sentiment...",
        "Examining support and resistance levels using algorithmic pattern recognition and volume-weighted price analysis...",
        "Scanning for confluence zones where multiple technical factors converge - trend lines, Fibonacci levels, and volume profiles...",
        "Measuring trend strength using ADX derivatives and momentum oscillators to quantify directional bias probability...",
        "Identifying key psychological levels and institutional order blocks through volume profile and market microstructure analysis...",
        "Assessing market volatility using ATR percentiles and volatility clustering patterns to gauge trend sustainability...",
        "Checking for hidden divergences between price action and momentum indicators across multiple timeframe perspectives...",
        "Analyzing volume patterns and order flow dynamics to detect smart money accumulation and distribution phases..."
    ],
    "timeframe_switch": [
        "Switching to {timeframe} for broader perspective - analyzing higher timeframe bias and structural market context...",
        "Zooming out to {timeframe} timeframe to confirm trend continuation patterns and identify potential reversal zones...",
        "Checking {timeframe} for multi-timeframe confirmation - ensuring alignment between micro and macro market structure...",
        "Examining {timeframe} structural framework to understand institutional positioning and long-term directional bias...",
        "Analyzing {timeframe} chart context to identify key inflection points and validate lower timeframe trading signals..."
    ],
    "drawing_lines": [
        "Drawing dynamic trendline connecting recent swing points - identifying potential breakout and breakdown scenarios...",
        "Marking critical support level at {price} based on volume profile analysis and previous institutional activity...",
        "Identifying resistance zone around {price} using Fibonacci confluence and historical turning point analysis...",
        "Connecting significant swing highs to establish trend channel parameters and projected price targets...",
        "Drawing horizontal level at {price} representing key psychological barrier and institutional order concentration...",
        "Marking potential breakout level where multiple technical factors converge for high-probability trade setups..."
    ],
    "zoom_analysis": [
        "Zooming in on recent price action to examine micro-structure and identify institutional footprints in order flow...",
        "Focusing on last {n} candles to analyze candlestick patterns and volume-price relationships for entry timing...",
        "Examining candlestick formations closely - looking for reversal patterns and continuation signals with high statistical probability...",
        "Analyzing market micro-structure to detect algorithmic trading patterns and high-frequency trading impact on price discovery...",
        "Investigating order flow around this critical level to understand supply-demand dynamics and institutional positioning strategies..."
    ],
    "conclusion": [
        "Bullish momentum building across multiple timeframes - confluence of technical factors suggests higher probability upside continuation...",
        "Bearish pressure intensifying with institutional distribution patterns - multiple timeframe analysis confirms downside bias probability...",
        "Market displaying complex consolidation patterns - waiting for directional clarity before committing to high-conviction positions...",
        "Sideways price action with conflicting signals - market structure suggests accumulation phase before next directional move...",
        "Strong trend continuation signals confirmed across timeframes - momentum indicators align with price action for sustained directional movement...",
        "Potential reversal patterns forming at key structural levels - monitoring for confirmation signals before position adjustments...",
        "Market structure integrity maintained - trend remains intact with healthy pullback patterns supporting continuation thesis...",
        "Key support and resistance levels holding with institutional respect - market showing strong structural foundation for next move..."
    ]
}
class DynamicChartAnalyzer:
    def __init__(self):
        self.current_timeframe = "M5"
        self.analysis_state = "idle"
        self.drawn_lines = []
        self.current_zoom_level = 100
        self.analysis_step = 0
        self.timeframes = ["M1", "M5", "M15", "M30", "H1", "H4"]
        self.timeframe_map = {
            "M1": mt5.TIMEFRAME_M1,
            "M5": mt5.TIMEFRAME_M5,
            "M15": mt5.TIMEFRAME_M15,
            "M30": mt5.TIMEFRAME_M30,
            "H1": mt5.TIMEFRAME_H1,
            "H4": mt5.TIMEFRAME_H4
        }
        
    def get_chart_data(self, symbol, timeframe, num_candles=100):
        """Get chart data from MT5"""
        try:
            # Log para debug
           # print(f"🔍 Tentando obter dados: {symbol} - {timeframe}")
            
            # Verificar se MT5 está conectado
            if not mt5.terminal_info():
                print("❌ MT5 não está conectado!")
                if not mt5.initialize():
                    print("❌ Falha ao inicializar MT5")
                    return None
            
            tf_mt5 = self.timeframe_map.get(timeframe, mt5.TIMEFRAME_M5)
            rates = mt5.copy_rates_from_pos(symbol, tf_mt5, 0, num_candles)
            
            if rates is None or len(rates) == 0:
                print(f"❌ Sem dados para {symbol}. Tentando símbolos alternativos...")
                
                # Tentar símbolos alternativos comuns
                alternative_symbols = [
                    symbol,  # Original
                    symbol.replace("/", ""),  # Remove barra (EURUSD)
                    symbol + ".",  # Com ponto (EURUSD.)
                    symbol.replace("EUR", "EUR/"),  # Com barra (EUR/USD)
                    "EURUSD",  # Padrão
                    "EURUSD.",
                    "EUR/USD"
                ]
                
                for alt_symbol in alternative_symbols:
                    rates = mt5.copy_rates_from_pos(alt_symbol, tf_mt5, 0, num_candles)
                    if rates is not None and len(rates) > 0:
                        print(f"✅ Dados encontrados para: {alt_symbol}")
                        df = pd.DataFrame(rates)
                        df['time'] = pd.to_datetime(df['time'], unit='s')
                        return df
                
                print("❌ Nenhum símbolo alternativo funcionou")
                return None
                
           # print(f"✅ Dados obtidos: {len(rates)} candles")
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            return df
        except Exception as e:
            print(f"❌ Erro ao obter dados: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def create_base_chart(self, symbol, timeframe):
        """Create base candlestick chart"""
        df = self.get_chart_data(symbol, timeframe)
        if df is None:
            # Criar gráfico vazio com mensagem
            fig = go.Figure()
            fig.add_annotation(
                text="Sem dados disponíveis<br>Verifique:<br>1. MT5 está conectado?<br>2. Símbolo existe?<br>3. Mercado está aberto?",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=16, color="red")
            )
            fig.update_layout(
                template="plotly_dark",
                height=500,
                plot_bgcolor='rgba(17, 17, 17, 1)',
                paper_bgcolor='rgba(17, 17, 17, 1)'
            )
            return fig, None
        
        fig = go.Figure()
        
        # Add candlestick
        fig.add_trace(go.Candlestick(
            x=df['time'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name="Price",
            increasing_line_color='#26a69a',
            decreasing_line_color='#ef5350'
        ))
        
        # Add EMA
        if 'close' in df.columns:
            ema9 = df['close'].ewm(span=9, adjust=False).mean()
            fig.add_trace(go.Scatter(
                x=df['time'],
                y=ema9,
                name="EMA9",
                line=dict(color='purple', width=2)
            ))
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=f"{symbol} - {timeframe}",
                x=0.5,
                xanchor='center'
            ),
            xaxis_rangeslider_visible=False,
            template="plotly_dark",
            height=500,
            margin=dict(l=50, r=50, t=50, b=50),
            xaxis=dict(
                gridcolor='rgba(128, 128, 128, 0.2)',
                showgrid=True
            ),
            yaxis=dict(
                gridcolor='rgba(128, 128, 128, 0.2)',
                showgrid=True,
                side='right'
            ),
            plot_bgcolor='rgba(17, 17, 17, 1)',
            paper_bgcolor='rgba(17, 17, 17, 1)'
        )
        
              

        return fig, df
    
    def add_support_resistance_lines(self, fig, df):
        """Add dynamic support/resistance lines"""
        if df is None or len(df) < 20:
            return fig
        
        # Find recent highs and lows
        recent_df = df.tail(50)
        
        # Simple peak/trough detection
        highs = recent_df[recent_df['high'] == recent_df['high'].rolling(window=5, center=True).max()]
        lows = recent_df[recent_df['low'] == recent_df['low'].rolling(window=5, center=True).min()]
        
        # Add resistance lines (red)
        for idx, high in highs.iterrows():
            if np.random.random() > 0.5:  # Random selection for variety
                fig.add_hline(
                    y=high['high'],
                    line_dash="dash",
                    line_color="rgba(255, 82, 82, 0.5)",
                    annotation_text=f"R: {high['high']:.4f}",
                    annotation_position="right"
                )
        
        # Add support lines (green)
        for idx, low in lows.iterrows():
            if np.random.random() > 0.5:  # Random selection for variety
                fig.add_hline(
                    y=low['low'],
                    line_dash="dash",
                    line_color="rgba(76, 175, 80, 0.5)",
                    annotation_text=f"S: {low['low']:.4f}",
                    annotation_position="right"
                )
        
        return fig
    
    def add_trend_lines(self, fig, df):
        """Add dynamic trend lines"""
        if df is None or len(df) < 20:
            return fig
        
        # Get recent data for trend analysis
        recent_df = df.tail(30)
        
        # Simple trend line using linear regression on recent lows
        x_numeric = np.arange(len(recent_df))
        
        # Uptrend line (connecting lows)
        if np.random.random() > 0.3:
            z = np.polyfit(x_numeric, recent_df['low'], 1)
            p = np.poly1d(z)
            trend_values = p(x_numeric)
            
            fig.add_trace(go.Scatter(
                x=recent_df['time'],
                y=trend_values,
                mode='lines',
                name='Trend',
                line=dict(color='rgba(255, 255, 255, 0.3)', width=2, dash='dot'),
                showlegend=False
            ))
        
        return fig
    
    def create_zoomed_view(self, fig, df, zoom_start, zoom_end):
        """Create zoomed view of chart"""
        if df is None:
            return fig
            
        # Update x-axis range for zoom effect
        fig.update_xaxes(range=[zoom_start, zoom_end])
        
        return fig
    
    def get_analysis_thought(self, category, **kwargs):
        """Get a random analysis thought from the bank"""
        thoughts = ANALYSIS_THOUGHTS.get(category, ["Analyzing..."])
        thought = random.choice(thoughts)
        
        # Format with any provided kwargs
        try:
            thought = thought.format(**kwargs)
        except:
            pass
            
        return thought
    
    def create_drawing_animation(self, fig, df, step):
        """Simulate drawing lines on chart progressively"""
        if df is None or step < 3:
            return fig
            
        # Progressive line drawing based on step
        if step % 4 == 0:
            fig = self.add_support_resistance_lines(fig, df)
        elif step % 4 == 2:
            fig = self.add_trend_lines(fig, df)
            
        return fig

def create_dynamic_chart_component():
    """Create the dynamic chart analysis component for Dash"""
    
    analyzer = DynamicChartAnalyzer()
    
    component = dbc.Card([
        dbc.CardHeader([
            html.H4(" 🧠 Real time AI Market Analysis", className="d-inline-block me-3"),
            html.Div(
                id="analysis-status-indicator",
                className="d-inline-block",
                children=[
                    html.I(className="fas fa-brain me-2 text-info"),
                    html.Span("Analyzing...", id="analysis-status-text", className="text-info")
                ]
            )
        ]),
        dbc.CardBody([
            # Analysis thoughts display
            html.Div(
                id="analysis-thoughts-container",
                className="mb-3 p-3 bg-dark rounded",
                style={
                    "minHeight": "60px",
                    "border": "1px solid rgba(13, 110, 253, 0.3)",
                    "position": "relative",
                    "overflow": "hidden"
                },
                children=[
                    html.Div(
                        id="analysis-thought-text",
                        className="text-info animate-pulse",
                        style={"fontSize": "14px"},
                        children="Initializing market analysis..."
                    ),
                    # Animated scanning line effect
                    html.Div(
                        className="scanning-line",
                        style={
                            "position": "absolute",
                            "bottom": "0",
                            "left": "0",
                            "height": "2px",
                            "width": "30%",
                            "background": "linear-gradient(to right, transparent, #0dcaf0, transparent)",
                            "animation": "scan 3s infinite"
                        }
                    )
                ]
            ),
            
            # Timeframe selector (visual only, changes automatically)
            html.Div(
                className="mb-3 text-center",
                children=[
                    html.Div(
                        id="timeframe-selector-dynamic",
                        className="btn-group",
                        role="group",
                        children=[
                            html.Button(
                                tf,
                                id=f"tf-btn-{tf}",
                                className=f"btn btn-sm btn-{'primary' if tf == 'M5' else 'outline-secondary'}",
                                style={"transition": "all 0.3s"}
                            ) for tf in ["M1", "M5", "M15", "M30", "H1", "H4"]
                        ]
                    )
                ]
            ),
            
            # Main chart container
            html.Div(
                id="dynamic-chart-container",
                style={
                    "position": "relative",
                    "border": "1px solid rgba(255, 255, 255, 0.1)",
                    "borderRadius": "5px",
                    "overflow": "hidden"
                },
                children=[
                    dcc.Graph(
                        id="dynamic-analysis-chart",
                        config={
                            "displayModeBar": False,
                            "scrollZoom": False
                        },
                        style={"height": "500px"}
                    ),
                    # Overlay effects
                    html.Div(
                        id="chart-overlay-effects",
                        style={
                            "position": "absolute",
                            "top": "0",
                            "left": "0",
                            "right": "0",
                            "bottom": "0",
                            "pointerEvents": "none",
                            "display": "none"
                        },
                        children=[
                            # Zoom effect border
                            html.Div(
                                className="zoom-effect",
                                style={
                                    "position": "absolute",
                                    "border": "2px solid rgba(13, 202, 240, 0.5)",
                                    "borderRadius": "5px",
                                    "animation": "pulse 1.5s infinite"
                                }
                            )
                        ]
                    )
                ]
            ),
            
            # Analysis progress indicator
            dbc.Progress(
                id="analysis-progress",
                value=0,
                max=100,
                striped=True,
                animated=True,
                color="info",
                className="mt-3",
                style={"height": "5px"}
            ),
            
            # Hidden interval for animations
            dcc.Interval(
                id="dynamic-analysis-interval",
                interval=2000,  # Update every 2 seconds
                n_intervals=0,
                disabled=False
            ),
            
            # Store for analysis state
            dcc.Store(id="analysis-state-store", data={
                "step": 0,
                "current_timeframe": "M5",
                "zoom_level": 100,
                "is_analyzing": True
            })
        ])
    ], className="mb-4", style={"backgroundColor": "rgba(33, 37, 41, 0.8)"})
    
    return component, analyzer

# CSS animations to be added to the main app
dynamic_chart_css = """
@keyframes scan {
    0% {
        left: -30%;
    }
    100% {
        left: 100%;
    }
}

@keyframes pulse {
    0% {
        opacity: 1;
        transform: scale(1);
    }
    50% {
        opacity: 0.5;
        transform: scale(1.05);
    }
    100% {
        opacity: 1;
        transform: scale(1);
    }
}

.zoom-effect {
    transition: all 0.5s ease-in-out;
}

.chart-annotation {
    animation: fadeIn 0.5s ease-in;
}

@keyframes fadeIn {
    from {
        opacity: 0;
        transform: translateY(-10px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

.timeframe-switch {
    animation: highlight 0.5s ease;
}

@keyframes highlight {
    0% {
        background-color: rgba(13, 202, 240, 0.3);
    }
    100% {
        background-color: transparent;
    }
}
"""

# Callback functions for the dynamic chart
def register_dynamic_chart_callbacks(app, analyzer, symbol_id='symbol-input'):
    """Register callbacks for dynamic chart analysis"""
    
    @app.callback(
        [Output("dynamic-analysis-chart", "figure"),
         Output("analysis-thought-text", "children"),
         Output("analysis-progress", "value"),
         Output("analysis-state-store", "data"),
         Output("analysis-status-text", "children")] + 
        [Output(f"tf-btn-{tf}", "className") for tf in ["M1", "M5", "M15", "M30", "H1", "H4"]],
        [Input("dynamic-analysis-interval", "n_intervals")],
        [State("analysis-state-store", "data"),
         State(symbol_id, "value")]
    )
    def update_dynamic_analysis(n_intervals, state_data, symbol):
        """Update the dynamic chart analysis visualization"""
        
        if not symbol:
            symbol = "EURUSD"
        
        # Get current state
        step = state_data.get("step", 0)
        current_tf = state_data.get("current_timeframe", "M5")
        zoom_level = state_data.get("zoom_level", 100)
        is_analyzing = state_data.get("is_analyzing", True)
        
        # Create base chart
        fig, df = analyzer.create_base_chart(symbol, current_tf)
        
        # Analysis sequence based on step
        thought = ""
        progress = 0
        status = "Analyzing..."
        
        if step < 5:
            # Initial trend analysis
            thought = analyzer.get_analysis_thought("trend_analysis", timeframe=current_tf)
            progress = step * 20
            
        elif step < 10:
            # Add some lines progressively
            fig = analyzer.create_drawing_animation(fig, df, step)
            thought = analyzer.get_analysis_thought("drawing_lines", price=f"{df['close'].iloc[-1]:.4f}" if df is not None else "N/A")
            progress = step * 10
            
        elif step < 15:
            # Switch timeframes
            timeframe_index = (step - 10) % len(analyzer.timeframes)
            current_tf = analyzer.timeframes[timeframe_index]
            fig, df = analyzer.create_base_chart(symbol, current_tf)
            thought = analyzer.get_analysis_thought("timeframe_switch", timeframe=current_tf)
            progress = step * 6.67
            
        elif step < 20:
            # Zoom analysis
            if df is not None and len(df) > 30:
                zoom_candles = 20 + (step - 15) * 5
                zoom_df = df.tail(zoom_candles)
                fig = analyzer.create_zoomed_view(fig, df, zoom_df['time'].iloc[0], zoom_df['time'].iloc[-1])
                thought = analyzer.get_analysis_thought("zoom_analysis", n=zoom_candles)
                fig = analyzer.create_drawing_animation(fig, zoom_df, step)
            progress = step * 5
            
        elif step < 25:
            # Final analysis
            thought = analyzer.get_analysis_thought("conclusion")
            fig = analyzer.add_support_resistance_lines(fig, df)
            fig = analyzer.add_trend_lines(fig, df)
            progress = 100
            status = "Analysis Complete"
        else:
            # Reset cycle
            step = 0
            thought = "Starting new analysis cycle..."
            progress = 0
            status = "Restarting..."
        
        # Update timeframe button classes
        tf_classes = []
        for tf in ["M1", "M5", "M15", "M30", "H1", "H4"]:
            if tf == current_tf:
                tf_classes.append("btn btn-sm btn-primary timeframe-switch")
            else:
                tf_classes.append("btn btn-sm btn-outline-secondary")
        
        # Update state
        new_state = {
            "step": step + 1,
            "current_timeframe": current_tf,
            "zoom_level": zoom_level,
            "is_analyzing": step < 25
        }
        
        return [fig, thought, progress, new_state, status] + tf_classes
    
    @app.callback(
        Output("dynamic-analysis-interval", "disabled"),
        [Input("bot-state", "data")]
    )
    def toggle_analysis_animation(bot_state):
        """Enable/disable analysis based on bot state"""
        if bot_state:
            return not bot_state.get("running", False)
        return True

# Function to integrate with main app
def add_dynamic_chart_to_layout(layout_children):
    """Add the dynamic chart component to an existing layout"""
    component, analyzer = create_dynamic_chart_component()
    
    # Find a good position to insert (e.g., after the main chart)
    insert_position = 2  # Adjust based on your layout
    
    # Insert the component
    layout_children.insert(insert_position, component)
    
    return layout_children, analyzer
