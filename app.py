# No topo do arquivo
import os
import json
import numpy as np
import pandas as pd
import MetaTrader5 as mt5
import threading
import queue
import re
import warnings
import sys
import time
from datetime import datetime, timedelta

# Dash e Plotly
import dash
from dash import dcc, html, Input, Output, State, callback_context, no_update
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# LangChain
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema.runnable import RunnablePassthrough
from langchain_community.callbacks.manager import get_openai_callback

# Módulos locais
from dotenv import load_dotenv
from frontend_features import create_debug_area, register_debug_callbacks, debug_print, debug_area_css, app_css
from agent_realtime_demo import create_agent_demo_app
from dynamic_chart_analyzer import create_dynamic_chart_component, register_dynamic_chart_callbacks, dynamic_chart_css
from indicators import calculate_atr, calculate_directional_entropy, calculate_ema, calculate_slope
from utils import parse_llm_response, format_trade_for_feedback, calculate_position_performance, detect_price_patterns
from prompts import initial_context_prompt
from ninjatrader_export import save_ninjatrader_command


load_dotenv()  # Carrega .env

#print("🔐 OPENAI_API_KEY:", os.getenv("OPENAI_API_KEY"))
#print("🐍 Python path:", sys.executable)
CONFIDENCE_THRESHOLD = 40  # Threshold mínimo para execução automática

warnings.filterwarnings("ignore")

initial_market_context = None
use_initial_context_enabled = False
token_usage = {"total_tokens": 0, "prompt_tokens": 0, "completion_tokens": 0}
last_token_print_time = time.time()
# Custom CSS para melhorar a aparência da interface

# Adicione o CSS à aplicação
app = dash.Dash(
    __name__, 
    external_stylesheets=[dbc.themes.DARKLY],
    suppress_callback_exceptions=True,
)

app.title = "MetaTrader5 LLM risk management"
app_css_complete = app_css + "\n" + dynamic_chart_css + "\n" + debug_area_css 



# Aplicar o CSS customizado
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
        <style>
''' + app_css_complete + '''
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''
# Global variables
confidence_history = {}
dynamic_analyzer = None  # Para o dynamic chart analyzer
# Criar o componente do dynamic chart
dynamic_chart_component, dynamic_analyzer = create_dynamic_chart_component()
running = False
trade_queue = queue.Queue()
memory = None
llm_chain = None
current_position = 0  # Current number of contracts
max_contracts = 15  # Default max contracts
confidence_level = 0  # Market confidence (-100 to 100)
llm_reasoning = ""
total_pnl = 0.0
market_direction = "Neutral"  # Market direction (Bullish, Bearish, Neutral)
trade_history = []
support_resistance_levels = []
h4_market_context_summary = None
indicator_history_buffer = {
    "atr": [],
    "entropy": [], 
    "slope": [],
    "ema9": [],
    "close": [],
    "timestamps": []  # Compartilhado entre todos os indicadores
}
# Define the timeframe mapping
timeframe_dict = {
    "M1": mt5.TIMEFRAME_M1,
    "M5": mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30,
    "H1": mt5.TIMEFRAME_H1,
    "H4": mt5.TIMEFRAME_H4,
    "D1": mt5.TIMEFRAME_D1,
}



# Set API Key from environment variable
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

from voice_utils import (
    get_available_voices, 
    process_chat_query_with_voice, 
    handle_voice_command_in_chat,
    create_voice_selector_with_tooltip
)
AVAILABLE_VOICES = get_available_voices()

# In the process_chat_query_enhanced function
# Replace your current implementation with this:

def process_chat_query_enhanced(query, symbol, timeframe):
    """
    Wrapper para a função process_chat_query_with_voice em voice_utils.py
    """
    global llm_chain, indicator_history_buffer, h4_market_context_summary, token_usage
    
    return process_chat_query_with_voice(
        query, 
        symbol, 
        timeframe, 
        llm_chain=llm_chain,
        indicator_history_buffer=indicator_history_buffer,
        h4_market_context_summary=h4_market_context_summary,
        get_current_candle_data=get_current_candle_data,
        get_indicator_buffer_summary=get_indicator_buffer_summary,
        get_recent_analysis_summary=get_recent_analysis_summary,
        initialize_llm_chain=initialize_llm_chain,
        token_usage=token_usage
    )
# --- UPDATE_CHAT CALLBACK FUNCTION ---
@app.callback(
    [Output("chat-messages", "children"),
     Output("chat-visual-results", "children"),
     Output("chat-visual-results", "style"),
     Output("chat-history", "data"),
     Output("chat-input", "value"),
     Output("voice-selector", "value", allow_duplicate=True)],
    [Input("send-button", "n_clicks"),
     Input("chat-input", "n_submit"),
     Input("suggestion-1", "n_clicks"),
     Input("suggestion-2", "n_clicks"),
     Input("suggestion-3", "n_clicks"),
     Input("suggestion-voice", "n_clicks"),
     Input("suggestion-4", "n_clicks")],
    [State("chat-input", "value"),
     State("chat-messages", "children"),
     State("chat-history", "data"),
     State("symbol-input", "value"),
     State("timeframe-dropdown", "value")],
    prevent_initial_call=True
)
def update_chat(send_clicks, input_submit, sug1, sug2, sug3, sug_voice, sug4,
               chat_input, chat_messages, chat_history, symbol, timeframe):
    """Updates the chat when the user sends a message"""
    # Check which component triggered the callback
    ctx = callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None
    
    # If no input was triggered or there's no text input, return current state
    if not triggered_id or (triggered_id == "send-button" and not chat_input):
        return chat_messages or [], [], {"display": "none"}, chat_history or {"messages": []}, "", no_update
    
    # Determine the query to be processed
    query = ""
    if triggered_id == "send-button" or triggered_id == "chat-input":
        query = chat_input
    elif triggered_id == "suggestion-1":
        query = "What is the current trend?"
    elif triggered_id == "suggestion-2":
        query = "Show the chart from last week"
    elif triggered_id == "suggestion-3":
        query = "Explain the latest candle pattern"
    elif triggered_id == "suggestion-voice":
        query = "Change voice to nova"
    elif triggered_id == "suggestion-4":
        query = "list the last 5 indicator values?"
    
    if not query:
        return chat_messages or [], [], {"display": "none"}, chat_history or {"messages": []}, "", no_update
    
    # Timestamp para mensagens
    timestamp = datetime.now().strftime("%H:%M")
    
    # Handle voice change commands
    is_voice_command, command_results = handle_voice_command_in_chat(query, timestamp, chat_messages, chat_history)
    if is_voice_command:
        return command_results
    
    # Add user message to the chat
    user_message = html.Div([
        html.Div(query, className="user-message"),
        html.Div(timestamp, className="timestamp")
    ])
    
    updated_messages = (chat_messages or []) + [user_message]
    
    try:
        # Process the query with enhanced context
        response_text, audio_html = process_chat_query_enhanced(query, symbol, timeframe)
    except Exception as e:
        print(f"Error in process_chat_query_enhanced: {e}")
        response_text = f"Error processing your query: {str(e)}"
        audio_html = None
    
    # Check if the query is about a specific date
    has_specific_date = re.search(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', query)
    has_timeframe_request = re.search(r'\b(daily|weekly|monthly|day|week|month)\b', query.lower())
    show_historical_chart = has_specific_date or has_timeframe_request
    date_to_show = None
    
    if has_specific_date:
        date_match = re.search(r'\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b', query)
        if date_match:
            date_to_show = date_match.group(1)
    
    # Create assistant message 
    assistant_message_content = [dcc.Markdown(response_text)]
    
    # Combine content into a single assistant message div
    assistant_message = html.Div([
        html.Div(assistant_message_content, className="assistant-message"),
        html.Div(timestamp, className="timestamp")
    ])
    
    # Add to messages
    final_messages = updated_messages + [assistant_message]
    
    # Prepare visualization if needed
    visual_results = []
    visual_style = {"display": "none"}
    
    if show_historical_chart:
        visual_results, visual_style = create_chat_visualization(symbol, timeframe, date_to_show)
    
    # Update chat history
    messages_history = chat_history.get("messages", []) if chat_history else []
    messages_history.append({
        "role": "user",
        "content": query,
        "timestamp": timestamp
    })
    messages_history.append({
        "role": "assistant",
        "content": response_text,
        "timestamp": timestamp
    })
    
    updated_chat_history = {"messages": messages_history}
    
    return final_messages, visual_results, visual_style, updated_chat_history, "", no_update

def log_candle_data(symbol, timeframe, candle_data, analysis, file_path=None, only_closed_candles=True):
    """
    Log candle data, indicator values, and LLM analysis to an Excel file
    
    Args:
        symbol (str): Trading symbol
        timeframe (str): Timeframe code (e.g. "H4")
        candle_data (str): JSON string containing candle data
        analysis (dict): Dictionary containing LLM analysis
        file_path (str, optional): Path to save the Excel file. If None, a default path is used.
        only_closed_candles (bool): Se True, registra apenas candles fechados
    """
    try:
        # Use default file path if none provided
        if file_path is None:
            file_path = f"metatradebot_log_{symbol}_{timeframe}.xlsx"
        
        # Parse candle data JSON
        try:
            if isinstance(candle_data, str):
                candle_dict = json.loads(candle_data)
            else:
                candle_dict = candle_data  # Already a dict
        except Exception as e:
            print(f"Error parsing candle data: {e}")
            return
        
        # Create a data dictionary for this entry
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        
        # Verificar se o candle está fechado ou em formação
        current_candle = candle_dict.get("current_candle", {})
        is_closed = current_candle.get("is_closed", False)
        
        # Se não houver informação explícita, verificar pelos timestamps
        if not is_closed and "timestamp" in current_candle:
            candle_time = current_candle["timestamp"]
            # Tentar converter para datetime
            try:
                if isinstance(candle_time, str):
                    candle_datetime = datetime.strptime(candle_time, "%Y-%m-%d %H:%M:%S")
                elif isinstance(candle_time, (int, float)):
                    candle_datetime = datetime.fromtimestamp(candle_time)
                else:
                    candle_datetime = candle_time
                
                # Calcular a diferença entre o timestamp atual e o do candle
                current_datetime = datetime.now()
                time_diff = current_datetime - candle_datetime
                
                # Converter timeframe para minutos para comparar com a diferença de tempo
                timeframe_minutes = 0
                if timeframe.startswith("M"):
                    timeframe_minutes = int(timeframe[1:])
                elif timeframe.startswith("H"):
                    timeframe_minutes = int(timeframe[1:]) * 60
                elif timeframe.startswith("D"):
                    timeframe_minutes = int(timeframe[1:]) * 60 * 24
                
                # Se passado mais tempo que o timeframe, consideramos o candle fechado
                if time_diff.total_seconds() / 60 >= timeframe_minutes:
                    is_closed = True
            except Exception as e:
                print(f"Erro ao verificar timestamp do candle: {e}")
        
        # Verificar também se os dados no buffer são de um candle fechado (penúltimo item)
        if 'indicator_history_buffer' in globals() and len(indicator_history_buffer.get("timestamps", [])) > 1:
            # Podemos usar o penúltimo item para obter dados do último candle fechado
            use_previous_candle = True
        else:
            use_previous_candle = False
        
        # Decidir se deve registrar ou não com base no estado do candle
        if only_closed_candles and not is_closed and not use_previous_candle:
            print(f"\n⚠️ Ignorando registro - candle atual ainda está em formação.")
            return
        
        # IMPORTANTE: Obter os valores dos indicadores
        atr_value = 0
        entropy_value = 0 
        ema9_value = 0
        slope_value = 0
        candle_timestamp = timestamp  # Default para o timestamp atual
        
        # Verificar se o buffer global está disponível e tem dados
        if 'indicator_history_buffer' in globals() and len(indicator_history_buffer.get("atr", [])) > 0:
            # Determinar qual índice usar (último ou penúltimo)
            index = -2 if use_previous_candle else -1
            
            # Usar o candle apropriado (fechado ou atual)
            atr_value = indicator_history_buffer["atr"][index]
            entropy_value = indicator_history_buffer["entropy"][index]
            ema9_value = indicator_history_buffer["ema9"][index]
            slope_value = indicator_history_buffer["slope"][index]
            
            # Obter o timestamp do candle para referência
            if len(indicator_history_buffer["timestamps"]) > abs(index):
                candle_timestamp = indicator_history_buffer["timestamps"][index]
            
            print(f"\n✅ Usando valores do indicator_history_buffer ({'-1' if index == -1 else '-2'}):")
            print(f"Candle timestamp: {candle_timestamp}")
            print(f"ATR: {atr_value}")
            print(f"Entropy: {entropy_value}")
            print(f"EMA9: {ema9_value}")
            print(f"Slope: {slope_value}")
        else:
            # Método alternativo - tentar estrutura normal
            indicators = current_candle.get("indicators", {})
            
            # Se os indicadores não estiverem no local esperado, tentar outras localizações
            if not indicators and "indicators" in candle_dict:
                indicators = candle_dict["indicators"]
                
            # Obter valores (com fallback para 0)
            atr_value = indicators.get("atr", 0)
            entropy_value = indicators.get("entropy", 0)
            ema9_value = indicators.get("ema9", 0)
            slope_value = indicators.get("slope", 0)
            
            print(f"\n⚠️ indicator_history_buffer não disponível, usando valores da estrutura:")
            print(f"ATR: {atr_value}")
            print(f"Entropy: {entropy_value}")
            print(f"EMA9: {ema9_value}")
            print(f"Slope: {slope_value}")
        
        # OHLC data
        ohlc = current_candle.get("ohlc", {})
        
        # Combine all data into a single row
        data = {
            "timestamp": timestamp,
            #"candle_timestamp": str(candle_timestamp),
            "symbol": symbol,
            "timeframe": timeframe,
            "candle_status": "Closed" if is_closed or use_previous_candle else "Open",
            
            # OHLC data
            "open": ohlc.get("open", 0),
            "high": ohlc.get("high", 0),
            "low": ohlc.get("low", 0),
            "close": ohlc.get("close", 0),
            "volume": current_candle.get("volume", 0),
            
            # Indicator values
            "atr": atr_value,
            "directional_entropy": entropy_value,
            "ema9": ema9_value,
            "slope": slope_value,
            
            # LLM analysis
            "market_summary": analysis.get("market_summary", ""),
            "confidence_level": analysis.get("confidence_level", 0),
            "direct_confidence": analysis.get("direct_confidence", 0),
            "llm_confidence": analysis.get("llm_confidence", 0),

            "direction": analysis.get("direction", "Neutral"),
            "action": analysis.get("action", "WAIT"),
            "contracts_to_adjust": analysis.get("contracts_to_adjust", 0),
            "reasoning": analysis.get("reasoning", "")[:500]  # Limit reasoning length
           
        }
        
        # Create DataFrame with this row
        df_row = pd.DataFrame([data])
        
        # Check if file exists
        if os.path.exists(file_path):
            # Read existing Excel file
            try:
                existing_df = pd.read_excel(file_path)
                # Append new row
                updated_df = pd.concat([existing_df, df_row], ignore_index=True)
            except Exception as e:
                print(f"Error reading existing Excel file: {e}")
                updated_df = df_row
        else:
            # Create new DataFrame if file doesn't exist
            updated_df = df_row

        # Save to Excel
        updated_df.to_excel(file_path, index=False)
        print(f"✅ Logged {'CLOSED' if is_closed or use_previous_candle else 'CURRENT'} candle data to {file_path} with indicators!")
        
    except Exception as e:
        print(f"Error in log_candle_data: {e}")
        import traceback
        traceback.print_exc()
        
def start_token_monitoring():
    """Inicia monitoramento de tokens em thread separada"""
    def monitor_tokens():
        global token_usage, last_token_print_time
        
        while True:
            current_time = time.time()
            if current_time - last_token_print_time >= 60:
                print(f"\n===== TOKEN USAGE REPORT =====")
                print(f"Total tokens used: {token_usage['total_tokens']}")
                print(f"Prompt tokens: {token_usage['prompt_tokens']}")
                print(f"Completion tokens: {token_usage['completion_tokens']}")
                print(f"Estimated cost (USD): ${(token_usage['prompt_tokens'] * 0.0000015 + token_usage['completion_tokens'] * 0.000002):.4f}")
                print(f"===============================\n")
                last_token_print_time = current_time
            time.sleep(60)  # Check every minute
    
    # Start monitoring in background thread
    monitoring_thread = threading.Thread(target=monitor_tokens, daemon=True)
    monitoring_thread.start()



def get_indicator_buffer_summary():
    """
    Cria um resumo do buffer de indicadores para o chat
    """
    global indicator_history_buffer
    
    if len(indicator_history_buffer["timestamps"]) < 3:
        return "Insufficient indicator history available"
    
    # Pegar últimos 3 valores para tendência
    recent_atr = indicator_history_buffer["atr"][-3:]
    recent_entropy = indicator_history_buffer["entropy"][-3:]
    recent_slope = indicator_history_buffer["slope"][-3:]
    recent_ema9 = indicator_history_buffer["ema9"][-3:]
    recent_close = indicator_history_buffer["close"][-3:]
    
    # Calcular tendências
    atr_trend = "increasing" if recent_atr[-1] > recent_atr[0] else "decreasing"
    entropy_trend = "increasing" if recent_entropy[-1] > recent_entropy[0] else "decreasing"
    slope_trend = "increasing" if recent_slope[-1] > recent_slope[0] else "decreasing"
    price_trend = "rising" if recent_close[-1] > recent_close[0] else "falling"
    
    summary = f"""
Indicator Buffer Summary (Last {len(indicator_history_buffer["timestamps"])} periods):
- ATR: Current {recent_atr[-1]:.4f}, trend {atr_trend}
- Entropy: Current {recent_entropy[-1]:.4f}, trend {entropy_trend}  
- Slope: Current {recent_slope[-1]:.6f}, trend {slope_trend}
- EMA9: Current {recent_ema9[-1]:.4f}
- Price: Current {recent_close[-1]:.4f}, trend {price_trend}
- Price vs EMA9: {"Above" if recent_close[-1] > recent_ema9[-1] else "Below"}
- Buffer spans: {indicator_history_buffer["timestamps"][0]} to {indicator_history_buffer["timestamps"][-1]}
"""
    
    return summary


def get_recent_analysis_summary():
    """
    Obtém resumo da análise mais recente
    """
    global llm_reasoning, confidence_level, market_direction
    
    if not llm_reasoning:
        return "No recent market analysis available"
    
    # Pegar apenas as primeiras linhas do reasoning para economizar tokens
    reasoning_preview = llm_reasoning.split('\n')[:3]
    reasoning_summary = '\n'.join(reasoning_preview)
    
    return f"""
Latest Market Analysis:
- Confidence Level: {confidence_level}
- Market Direction: {market_direction}
- Key Reasoning: {reasoning_summary}...
- Current Position: {current_position} contracts
"""


def create_chat_visualization(symbol, timeframe, date_to_show=None):
    """
    Cria visualização para o chat quando solicitado
    """
    try:
        # Determine the period to get historical data
        if date_to_show:
            df = get_historical_data(symbol, timeframe, target_date=date_to_show)
            period_text = f"Historical data for {date_to_show}"
        else:
            df = get_historical_data(symbol, timeframe, days_ago=7)
            period_text = "Data from last week"
        
        if df is not None and len(df) > 0:
            # Create figure for visualization
            fig = go.Figure()
            
            # Add candlesticks
            fig.add_trace(
                go.Candlestick(
                    x=df['time'],
                    open=df['open'],
                    high=df['high'],
                    low=df['low'],
                    close=df['close'],
                    name="Price"
                )
            )
            
            # Add EMA9
            fig.add_trace(
                go.Scatter(
                    x=df['time'],
                    y=df['ema9'],
                    name="EMA9",
                    line=dict(color='purple', width=1.5)
                )
            )
            
            # Update layout
            fig.update_layout(
                
                title=period_text,
                height=350,
                margin=dict(l=20, r=20, t=40, b=20),
                template="plotly_dark",
                xaxis_rangeslider_visible=False,
            )
            
            # Close button
            close_button = html.Button(
                "×", 
                id="close-historical-chart",
                style={
                    "position": "absolute",
                    "top": "5px",
                    "right": "5px",
                    "background": "none",
                    "border": "none",
                    "color": "white",
                    "fontSize": "20px",
                    "cursor": "pointer"
                }
            )
            
            visual_results = [
                html.Div([
                    html.H5(period_text, className="mb-2"),
                    close_button,
                    dcc.Graph(figure=fig, config={"displayModeBar": False})
                ], className="historical-chart-container")
            ]
            
            visual_style = {"display": "block", "marginBottom": "20px"}
            
            return visual_results, visual_style
        else:
            return [], {"display": "none"}
            
    except Exception as e:
        print(f"Error creating chat visualization: {e}")
        return [], {"display": "none"}
    
    
def get_historical_data(symbol, timeframe_code, target_date=None, days_ago=None):
    """
    Gets historical data for a specific date or X days ago
    
    Args:
        symbol (str): Asset symbol
        timeframe_code (str): Timeframe code (e.g., "M15")
        target_date (str, optional): Target date in "DD/MM/YYYY" format
        days_ago (int, optional): Number of days to look back
        
    Returns:
        pd.DataFrame: DataFrame with historical data
    """
    if target_date is not None:
        # Convert date string to datetime object
        try:
            # Try various possible date formats
            for fmt in ["%d/%m/%Y", "%d-%m-%Y", "%d/%m/%y", "%d-%m-%y"]:
                try:
                    dt = datetime.strptime(target_date, fmt)
                    break
                except ValueError:
                    continue
            else:
                # If no format works
                raise ValueError(f"Unrecognized date format: {target_date}")
        except Exception as e:
            print(f"Error converting date: {e}")
            return None
    elif days_ago is not None:
        # Calculate target date based on days ago
        dt = datetime.now() - timedelta(days=days_ago)
    else:
        # If neither parameter is provided, use current day
        dt = datetime.now()
    
    # Convert to timestamp
    timestamp = int(dt.timestamp())
    
    # Get historical data (30 candles before and 10 after for context)
    timeframe_mt5 = timeframe_dict.get(timeframe_code, mt5.TIMEFRAME_H4)
    
    # Fetch historical data from a specific date using the timestamp
    rates = mt5.copy_rates_from(symbol, timeframe_mt5, timestamp, 40)
    
    if rates is None or len(rates) == 0:
        print(f"No historical data found for {symbol} on {target_date or f'{days_ago} days ago'}")
        return None
    
    # Convert to DataFrame
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    
    # Calculate indicators
    df['atr'] = calculate_atr(df, period=14)
    df['entropy'] = calculate_directional_entropy(df, period=14)
    df['ema9'] = calculate_ema(df['close'], period=9)
    
    return df

# Callback to handle the close button for the historical chart
@app.callback(
    [Output("chat-visual-results", "children", allow_duplicate=True),
     Output("chat-visual-results", "style", allow_duplicate=True)],
    [Input("close-historical-chart", "n_clicks")],
    [State("chat-visual-results", "children")],
    prevent_initial_call=True
)
def close_historical_chart(n_clicks, current_children):
    """Closes the historical chart when the X button is clicked"""
    if n_clicks:
        return [], {"display": "none"}
    return current_children, {"display": "block"}




def initialize_mt5(server="AMPGlobalUSA-Demo", login=1531555, password="_aBlP0Tg"):
    """Initialize connection to MetaTrader5"""
    if not mt5.initialize():
        print("initialize() failed, error code =", mt5.last_error())
        return False
    
    # Try to connect without login credentials first (use already connected account)
    if mt5.account_info():
        print("Already connected to MetaTrader5")
        return True
    
    # If not connected, try with credentials
    authorized = mt5.login(login, password, server)
    if not authorized:
        print(f"Failed to connect to account {login}, error code: {mt5.last_error()}")
        return False
    
    print(f"Connected to account {login}")
    return True


# Modify the existing callback to update reasoning and confidence displays
@app.callback(
    [Output("position-display", "children"),
     Output("pnl-display", "children"),
     Output("direction-display", "children"),
     Output("confidence-bar", "value"),
     Output("confidence-bar", "color"),
     Output("reasoning-display", "children"),
     Output("confidence-value", "children"),
     Output("key-factors", "children"),
     Output("past-market-context", "children"),
     Output("last-llm-analysis", "data"),
     Output("confidence-details", "children"),
     Output("confluence-visualization", "children")],  # Novo output adicionado aqui
    [Input("interval-component", "n_intervals"),
     Input("analyze-button", "n_clicks")],
    [State("last-llm-analysis", "data")]
)
def update_dashboard(n_intervals, n_clicks, last_analysis):
    """Update dashboard elements with enhanced confidence visualization"""
    global initial_market_context, use_initial_context_enabled
    
    # Format position display
    position_text = f"{current_position} Contracts"
    
    # Format P&L display
    pnl_text = f"${total_pnl:.2f}"
    
    # Format direction display
    direction_text = market_direction
    
    # Calculate confidence bar value (convert from -100/100 scale to 0/100 scale)
    confidence_bar_value = (confidence_level + 100) / 2
    
    # Determine bar color based on confidence level
    if confidence_level < -60:
        bar_color = "danger"  # Vermelho escuro para muito bearish
    elif confidence_level < -20:
        bar_color = "warning"  # Laranja para moderadamente bearish
    elif confidence_level < 20:
        bar_color = "info"     # Azul para neutro
    elif confidence_level < 60:
        bar_color = "primary"  # Azul escuro para moderadamente bullish
    else:
        bar_color = "success"  # Verde para muito bullish
    
    # Format reasoning text com syntax highlighting melhorado
    reasoning_html = []
    
    if llm_reasoning:
        # Dividir em parágrafos para processamento
        paragraphs = llm_reasoning.split("\n")
        
        for para in paragraphs:
            if not para.strip():  # Pular parágrafos vazios
                continue
                
            # Destacar menções a indicadores e suas tendências
            highlighted_para = para
            
            # Substituir menções a indicadores e termos-chave por versões destacadas
            highlight_patterns = {
                r'\b(ATR|Average True Range)\b': '<span class="text-info fw-bold">\\1</span>',
                r'\b(Entropy|Directional Entropy)\b': '<span class="text-info fw-bold">\\1</span>',
                r'\b(Slope|angle)\b': '<span class="text-info fw-bold">\\1</span>',
                r'\b(EMA9|EMA|moving average)\b': '<span class="text-info fw-bold">\\1</span>',
                r'\b(increasing|rising|higher|stronger)\b': '<span class="text-success">\\1</span>',
                r'\b(decreasing|falling|lower|weaker)\b': '<span class="text-danger">\\1</span>',
                r'\b(uptrend|bullish|positive)\b': '<span class="text-success fw-bold">\\1</span>',
                r'\b(downtrend|bearish|negative)\b': '<span class="text-danger fw-bold">\\1</span>',
                r'\b(confluence|alignment|agree|confirms)\b': '<span class="text-warning fw-bold">\\1</span>',
                r'\b(divergence|conflict|contradicts)\b': '<span class="text-danger fw-bold">\\1</span>',
                r'\b(volatility|volume)\b': '<span class="text-secondary fw-bold">\\1</span>',
                r'\b(reversal|turning point|shift)\b': '<span class="text-warning fw-bold">\\1</span>',
                r'\b(support|resistance|level|zone)\b': '<span class="text-primary fw-bold">\\1</span>',
            }
            
            # Aplicar os padrões de highlighting
            for pattern, replacement in highlight_patterns.items():
                highlighted_para = re.sub(pattern, replacement, highlighted_para, flags=re.IGNORECASE)
            
            from dash import dcc
            reasoning_html.append(dcc.Markdown(highlighted_para, dangerously_allow_html=True))
    else:
        reasoning_html = html.P("No analysis available yet. Start trading or trigger analysis manually.")
    
    # Preparar texto de nível de confiança com mais detalhes
    abs_confidence = abs(confidence_level)
    
    if abs_confidence >= 80:
        confidence_word = "Very High"
        confidence_badge_color = "success" if confidence_level > 0 else "danger"
        contract_factor = "100%"
    elif abs_confidence >= 60:
        confidence_word = "High"
        confidence_badge_color = "success" if confidence_level > 0 else "danger"
        contract_factor = "75%"
    elif abs_confidence >= 40:
        confidence_word = "Medium"
        confidence_badge_color = "primary" if confidence_level > 0 else "warning"
        contract_factor = "50%"
    elif abs_confidence >= 20:
        confidence_word = "Low"
        confidence_badge_color = "info" if confidence_level > 0 else "warning"
        contract_factor = "25%"
    else:
        confidence_word = "Very Low / Neutral"
        confidence_badge_color = "secondary"
        contract_factor = "0%"
    
    # Format confidence value text with more detail
    if confidence_level > 0:
        confidence_text = f"Bullish Confidence: {confidence_level}"
        confidence_style = {"color": "green"}
        confidence_details = f"{confidence_word} Bullish ({contract_factor} position sizing)"
    elif confidence_level < 0:
        confidence_text = f"Bearish Confidence: {confidence_level}"
        confidence_style = {"color": "red"}
        confidence_details = f"{confidence_word} Bearish ({contract_factor} position sizing)"
    else:
        confidence_text = f"Neutral Confidence: {confidence_level}"
        confidence_style = {"color": "blue"}
        confidence_details = f"Neutral (no position adjustment)"
    
    confidence_value_html = html.Div([
        html.Span(confidence_text, style=confidence_style),
        html.Br(),
        html.Span(confidence_details, className="text-muted small")
    ])
    
    # Extract key factors from the reasoning with enhanced detection
    key_factors_html = extract_key_factors_enhanced(llm_reasoning)
    
    # Format past market context from initial_market_context
    past_market_context_html = []
    if use_initial_context_enabled and initial_market_context:
        try:
            # Try to parse as JSON first
            context_data = json.loads(initial_market_context)
            # Format as a set of key-value pairs
            for key, value in context_data.items():
                formatted_key = key.replace("_", " ").title()
                past_market_context_html.append(html.Div([
                    html.Strong(f"{formatted_key}: ", className="text-info"),
                    html.Span(f"{value}")
                ], className="mb-2"))
        except:
            # If it's not valid JSON, display as text paragraphs
            paragraphs = initial_market_context.split("\n")
            for para in paragraphs:
                if para.strip():
                    past_market_context_html.append(html.P(para))
    else:
        past_market_context_html = [html.P("Initial market context is disabled or not available.", className="text-muted")]

    # Criar um componente visual detalhado da confiança
    confidence_details_html = create_confidence_details(confidence_level, llm_reasoning)

    # Store the latest analysis for reference
    current_analysis = {
        "reasoning": llm_reasoning,
        "confidence_level": confidence_level,
        "direction": market_direction,
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Criar visualização de confluência de indicadores
    # Usamos a função que definimos anteriormente
    confluence_visualization = create_confluence_visualization(current_analysis)
    
    return (position_text, pnl_text, direction_text, confidence_bar_value, 
            bar_color, reasoning_html, confidence_value_html, 
            key_factors_html, past_market_context_html, current_analysis, 
            confidence_details_html, confluence_visualization)


def extract_key_factors_enhanced(reasoning_text):
    """Extract key factors from reasoning text with enhanced pattern recognition"""
    if not reasoning_text:
        return html.P("No factors available", className="text-muted")
    
    # Categorias de fatores
    categories = {
        "Indicator Trends": [],
        "Price Action": [],
        "Market Structure": [],
        "Confluence Patterns": []
    }
    
    # Expressões regulares para diferentes categorias
    indicator_patterns = [
        r"ATR (is |has been |shows |indicating |)\w+",
        r"Entropy (is |has been |shows |indicating |)\w+",
        r"Slope (is |has been |shows |indicating |)\w+",
        r"EMA[0-9]+ (is |has been |shows |indicating |)\w+"
    ]
    
    price_patterns = [
        r"price (is |has |)\w+ (above|below|near|at|approaching)",
        r"candle(s)? (formation|pattern|showing|indicates|suggest)",
        r"(bullish|bearish) candle",
        r"(upper|lower) (wick|shadow)",
        r"(close|open|high|low) (above|below|near)",
        r"price action (suggests|indicates|confirms|shows)"
    ]
    
    structure_patterns = [
        r"(support|resistance) (level|zone|area)",
        r"(key|significant|important) level",
        r"market (structure|framework|context)",
        r"H4 (timeframe|analysis|chart|pattern) (shows|indicates|suggests)",
        r"(higher|lower) (high|low)",
        r"market (is |has been |)(trending|consolidating|ranging)"
    ]
    
    confluence_patterns = [
        r"(confluence|agreement|alignment) (between|of|with)",
        r"(multiple|several) (indicators|factors) (show|suggest|confirm|indicate)",
        r"(indicators|factors) (are |)(aligned|confirming|supporting)",
        r"(strong|clear) (signal|indication|evidence)",
        r"(divergence|disagreement|contradiction) (between|with)"
    ]
    
    # Dividir o texto em frases
    sentences = re.split(r'[.!?]+', reasoning_text)
    
    # Examinar cada frase para as diferentes categorias
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        
        # Verificar padrões de indicadores
        for pattern in indicator_patterns:
            if re.search(pattern, sentence, re.IGNORECASE):
                if sentence not in categories["Indicator Trends"]:
                    categories["Indicator Trends"].append(sentence)
                break
        
        # Verificar padrões de price action
        for pattern in price_patterns:
            if re.search(pattern, sentence, re.IGNORECASE):
                if sentence not in categories["Price Action"]:
                    categories["Price Action"].append(sentence)
                break
        
        # Verificar padrões de estrutura de mercado
        for pattern in structure_patterns:
            if re.search(pattern, sentence, re.IGNORECASE):
                if sentence not in categories["Market Structure"]:
                    categories["Market Structure"].append(sentence)
                break
        
        # Verificar padrões de confluência
        for pattern in confluence_patterns:
            if re.search(pattern, sentence, re.IGNORECASE):
                if sentence not in categories["Confluence Patterns"]:
                    categories["Confluence Patterns"].append(sentence)
                break
    
    # Preparar o HTML para cada categoria
    factors_html = []
    
    for category, factors in categories.items():
        if factors:
            # Adicionar cabeçalho da categoria
            factors_html.append(html.H6(category, className="mt-2 text-info"))
            # Adicionar fatores desta categoria
            factors_html.append(html.Ul([html.Li(factor) for factor in factors[:3]], className="small"))  # Limitar a 3 fatores por categoria
    
    if not any(len(factors) > 0 for factors in categories.values()):
        return html.P("No specific factors detected in the analysis", className="text-muted")
    
    return html.Div(factors_html)

def create_confidence_details(confidence_level, reasoning_text):
    """Create a detailed visual representation of confidence based on indicator confluence"""
    if not reasoning_text:
        return html.Div("No confidence details available")
    
    abs_confidence = abs(confidence_level)
    
    # Determinar o fator de contrato baseado no nível de confiança
    if abs_confidence >= 60:
        contract_factor = "20%"
    elif abs_confidence >= 50:
        contract_factor = "10%"
    elif abs_confidence >= 45:
        contract_factor = "5%"
    elif abs_confidence >= 40: #__________________________________________________________________________________________________________________________________________
        contract_factor = "5%"
    else:
        contract_factor = "0%"
        
    # CORREÇÃO: Inicializar indicator_counts corretamente
    indicator_counts = {
        "ATR": {"bullish": 0, "bearish": 0, "neutral": 0},
        "Entropy": {"bullish": 0, "bearish": 0, "neutral": 0},
        "Slope": {"bullish": 0, "bearish": 0, "neutral": 0},
        "EMA": {"bullish": 0, "bearish": 0, "neutral": 0},
        "H4 Alignment": {"bullish": 0, "bearish": 0, "neutral": 0}
    }
    
    # Padrões para detectar menções de indicadores e suas tendências
    indicator_trend_patterns = {
        "ATR": {
            "bullish": [r"ATR (is |)(increasing|rising|higher|expanding)", r"increased volatility (supports|suggests) (bullish|upward)"],
            "bearish": [r"ATR (is |)(decreasing|falling|lower|contracting)", r"decreased volatility (supports|suggests) (bearish|downward)"],
            "neutral": [r"ATR (is |)(stable|flat|neutral|unchanged)", r"ATR (does not|doesn't) (indicate|suggest|show)"]
        },
        "Entropy": {
            "bullish": [r"Entropy (is |)(decreasing|falling|lower)", r"(low|decreasing) entropy (indicates|suggests) (trend|directional movement|bullish)"],
            "bearish": [r"Entropy (is |)(decreasing|falling|lower)", r"(low|decreasing) entropy (indicates|suggests) (trend|directional movement|bearish)"],
            "neutral": [r"Entropy (is |)(increasing|rising|higher|stable)", r"(high|increasing) entropy (indicates|suggests) (ranging|choppy|sideways)"]
        },
        "Slope": {
            "bullish": [r"Slope (is |)(positive|increasing|becoming more positive)", r"(upward|uptrend|positive) slope"],
            "bearish": [r"Slope (is |)(negative|decreasing|becoming more negative)", r"(downward|downtrend|negative) slope"],
            "neutral": [r"Slope (is |)(near zero|flat|neutral)", r"Slope (does not|doesn't) (indicate|suggest|show)"]
        },
        "EMA": {
            "bullish": [r"(price|close) (is |)(above|crossing above) (the |)EMA", r"EMA (has|shows|with) (a |)(positive|upward|bullish) slope"],
            "bearish": [r"(price|close) (is |)(below|crossing below) (the |)EMA", r"EMA (has|shows|with) (a |)(negative|downward|bearish) slope"],
            "neutral": [r"(price|close) (is |)(near|at|hovering around) (the |)EMA", r"EMA (is |)(flat|stable|horizontal)"]
        },
        "H4 Alignment": {
            "bullish": [r"H4 (timeframe|chart|analysis) (shows|indicates|confirms|aligns with) (bullish|uptrend|upward)", r"(alignment|agreement) between H4 and current timeframe (bullish|upward)"],
            "bearish": [r"H4 (timeframe|chart|analysis) (shows|indicates|confirms|aligns with) (bearish|downtrend|downward)", r"(alignment|agreement) between H4 and current timeframe (bearish|downward)"],
            "neutral": [r"H4 (timeframe|chart|analysis) (shows|indicates) (neutral|sideways|ranging)", r"(conflict|disagreement) between H4 and current timeframe"]
        }
    }
    
    # Verificar cada padrão no texto de raciocínio
    for indicator, trend_patterns in indicator_trend_patterns.items():
        for trend, patterns in trend_patterns.items():
            for pattern in patterns:
                if re.search(pattern, reasoning_text, re.IGNORECASE):
                    indicator_counts[indicator][trend] += 1
    
    # Determinar a tendência dominante para cada indicador
    indicator_trends = {}
    for indicator, counts in indicator_counts.items():
        if counts["bullish"] > counts["bearish"] and counts["bullish"] > counts["neutral"]:
            indicator_trends[indicator] = "bullish"
        elif counts["bearish"] > counts["bullish"] and counts["bearish"] > counts["neutral"]:
            indicator_trends[indicator] = "bearish"
        else:
            indicator_trends[indicator] = "neutral"
    
    # Contar o número de indicadores em concordância com a direção geral
    general_direction = "bullish" if confidence_level > 0 else "bearish" if confidence_level < 0 else "neutral"
    aligned_indicators = sum(1 for trend in indicator_trends.values() if trend == general_direction)
    
    # Criar visualização
    cards = []
    
    # Cabeçalho com informações gerais
    cards.append(html.Div([
        html.H5(f"Confidence Analysis: {abs_confidence}/100", 
                className=f"text-{'success' if confidence_level > 0 else 'danger' if confidence_level < 0 else 'info'}"),
        html.P([
            f"Direction: {general_direction.capitalize()} with ",
            html.Strong(f"{aligned_indicators}/5", 
                        className=f"text-{'success' if aligned_indicators >= 4 else 'warning' if aligned_indicators >= 2 else 'danger'}"),
            " indicators in alignment"
        ])
    ], className="mb-3"))
    
    # Criar cards para cada indicador
    for indicator, trend in indicator_trends.items():
        # Determinar cores e ícones
        if trend == "bullish":
            bg_color = "success"
            icon = "↗️"
            text_color = "white"
        elif trend == "bearish":
            bg_color = "danger"
            icon = "↘️"
            text_color = "white"
        else:
            bg_color = "secondary"
            icon = "↔️"
            text_color = "white"
        
        # Criar card para este indicador
        indicator_card = dbc.Card([
            dbc.CardBody([
                html.H6([icon, " ", indicator], className=f"text-{text_color}"),
                html.P(f"Trend: {trend.capitalize()}", className=f"text-{text_color} mb-0")
            ])
        ], color=bg_color, className="mb-2")
        
        cards.append(indicator_card)
    
    # Adicionar explicação de posicionamento
    position_sizing_card = dbc.Card([
        dbc.CardBody([
            html.H6("Position Sizing Impact", className="text-white"),
            html.P([
                f"Based on {abs_confidence}/100 confidence: ",
                html.Strong(f"{contract_factor} of suggested contracts", 
                           className=f"text-{'success' if abs_confidence >= 60 else 'warning' if abs_confidence >= 20 else 'danger'}")
            ], className="text-white mb-0")
        ])
    ], color="primary", className="mt-3")
    
    cards.append(position_sizing_card)
    
    return html.Div(cards)

def get_h4_market_context_narrative(symbol, num_candles=5):
    """
    Obtém contexto H4 e gera um resumo narrativo usando LLM
    Retorna análise como se fosse feita por um trader experiente
    """
    try:
        # Obter dados H4 históricos
        h4_rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H4, 1, num_candles)
        
        if h4_rates is None or len(h4_rates) < 15:
            return "Dados H4 insuficientes para análise de contexto"
        
        # Converter para DataFrame
        h4_df = pd.DataFrame(h4_rates)
        h4_df['time'] = pd.to_datetime(h4_df['time'], unit='s')
        
        # Calcular indicadores
        h4_df['atr'] = calculate_atr(h4_df, period=14)
        h4_df['entropy'] = calculate_directional_entropy(h4_df, period=14)
        h4_df['ema9'] = calculate_ema(h4_df['close'], period=9)
        h4_df['slope'] = calculate_slope(h4_df['close'], period=14)
        
        # Pegar candles válidos para análise
        valid_df = h4_df.iloc[14:].reset_index(drop=True)
        analysis_df = valid_df.tail(num_candles)
        
        # Preparar dados estruturados para o LLM analisar
        h4_data = {
            # "period_analyzed": f"Últimos {len(analysis_df)} candles H4",
            # "start_time": analysis_df['time'].iloc[0].strftime("%Y-%m-%d %H:%M"),
            # "end_time": analysis_df['time'].iloc[-1].strftime("%Y-%m-%d %H:%M"),
            
            "price_movement": {
                "opening_price": float(analysis_df['open'].iloc[0]),
                "closing_price": float(analysis_df['close'].iloc[-1]),
            #     "highest_point": float(analysis_df['high'].max()),
            #     "lowest_point": float(analysis_df['low'].min()),
            #     "total_range": float(analysis_df['high'].max() - analysis_df['low'].min()),
            #     "net_movement": float(analysis_df['close'].iloc[-1] - analysis_df['open'].iloc[0])
             },
            
            "trend_analysis": {
                "slope_progression": [round(float(x), 6) for x in analysis_df['slope'].tail(5)],
                "ema9_progression": [round(float(x), 4) for x in analysis_df['ema9'].tail(5)],
                "price_vs_ema_current": "ABOVE" if analysis_df['close'].iloc[-1] > analysis_df['ema9'].iloc[-1] else "BELOW",
                "price_vs_ema_start": "ABOVE" if analysis_df['close'].iloc[0] > analysis_df['ema9'].iloc[0] else "BELOW"
            },
            
            "volatility_structure": {
                #"atr_current": round(float(analysis_df['atr'].iloc[-1]), 6),
                "atr_average": round(float(analysis_df['atr'].mean()), 6),
                "atr_trend": "INCREASING" if analysis_df['atr'].iloc[-1] > analysis_df['atr'].iloc[0] else "DECREASING",
                #"entropy_current": round(float(analysis_df['entropy'].iloc[-1]), 4),
                "entropy_average": round(float(analysis_df['entropy'].mean()), 4)
            },
            
            # "market_structure": {
            #     "recent_highs": [round(float(x), 4) for x in analysis_df['high'].tail(3)],
            #     "recent_lows": [round(float(x), 4) for x in analysis_df['low'].tail(3)],
            #     "key_support": round(float(analysis_df['low'].min()), 4),
            #     "key_resistance": round(float(analysis_df['high'].max()), 4)
            # }
        }
        
        # Criar prompt para LLM gerar resumo narrativo
        narrative_prompt = ChatPromptTemplate.from_template("""
        You are an expert market analyst providing essential context for {symbol}.

        Analyze this H4 data and distill it into critical trading insights:

        # H4 DATA FOR ANALYSIS:
        {h4_data}

        # OUTPUT REQUIREMENTS:
        Create ONE concise paragraph (maximum 2-3 sentences) that captures:

        1. **Directional Bias**: The dominant H4 trend direction and any recent reversals or momentum shifts
        2. **Key Structure**: Notable swing highs/lows and current price position relative to EMA9
        3. **Volatility Context**: ATR and entropy readings that impact lower timeframe trading decisions
        4. **Trading Implications**: How this H4 context should influence M5/M15 entries and exits

        # STYLE GUIDE:
        - Be direct and precise - focus on actionable insights only
        - Use exact values for key indicators
        - Maintain a professional but conversational tone
        - Focus on QUALITATIVE assessment (strength, weakness, shifts) over raw data
        - Prioritize recent changes or developing conditions over static information

        """)
        # Inicializar LLM para narrativa
        narrative_llm = ChatOpenAI(
            temperature=0.3,
            api_key=os.getenv("OPENAI_API_KEY"),
            model_name="gpt-4.1-mini"
        )
        
        # Criar chain para narrativa
        narrative_chain = (
            {
                "symbol": lambda x: symbol,
                "h4_data": lambda x: json.dumps(h4_data, indent=2)
            }
            | narrative_prompt
            | narrative_llm
        )
        
        # Gerar narrativa
        with get_openai_callback() as cb:
            narrative_response = narrative_chain.invoke({})
            
            # Log token usage for H4 context
            print(f"\n===== H4 CONTEXT TOKEN USAGE =====")
            print(f"Prompt tokens: {cb.prompt_tokens}")
            print(f"Completion tokens: {cb.completion_tokens}")
            print(f"Total tokens: {cb.total_tokens}")
            print(f"===================================\n")
            
            # Update global token usage
            token_usage["total_tokens"] += cb.total_tokens
            token_usage["prompt_tokens"] += cb.prompt_tokens
            token_usage["completion_tokens"] += cb.completion_tokens
        
        return narrative_response.content
        
    except Exception as e:
        print(f"Erro ao gerar contexto narrativo H4: {e}")
        return f"Erro na análise H4: dados indisponíveis no momento"

def initialize_h4_context(symbol):
    """
    Inicializa o contexto H4 que será usado como referência histórica
    NOVA VERSÃO: Agora gera um resumo narrativo ao invés de JSON
    """
    global h4_market_context_summary
    
    print("🔄 Inicializando contexto H4 com análise narrativa...")
    debug_print("✅ System started, generating analysis")
    h4_market_context_summary = get_h4_market_context_narrative(symbol, num_candles=5)
    print("✅ Contexto H4 narrativo gerado com sucesso")
    print(f"📊 Resumo H4: {h4_market_context_summary[:150]}...")

# Atualizar a função analyze_market para usar o novo contexto narrativo
def analyze_market(symbol, timeframe):
    """
    Análise de mercado utilizando tanto o LLM quanto o cálculo direto de confluência
    """
    global llm_reasoning, confidence_level, market_direction, llm_chain
    global h4_market_context_summary, confidence_history
    print(f"DEBUG: Analisando mercado com posição atual: {current_position} contratos")
    debug_print(f"Analyzing market with current position: {current_position} contracts", "info")
    # Verificar se o contexto H4 narrativo foi inicializado
    if h4_market_context_summary is None:
        print("⚠️  Contexto H4 narrativo não inicializado. Inicializando agora...")
        initialize_h4_context(symbol)
    
    # Obter dados detalhados do timeframe atual
    current_timeframe_data_str = get_current_candle_data(symbol, timeframe)
    
    if "Erro" in current_timeframe_data_str or "insuficiente" in current_timeframe_data_str:
        print(f"⚠️  {current_timeframe_data_str}")
        return {
            "market_summary": current_timeframe_data_str,
            "confidence_level": 0,
            "direction": "Neutral",
            "action": "WAIT",
            "reasoning": "Dados insuficientes para análise",
            "contracts_to_adjust": 0
        }
    
    # Converter a string JSON em dicionário para análise direta
    try:
        current_data_dict = json.loads(current_timeframe_data_str)
        
        # Análise de confluência direta baseada em regras
        # Onde a análise de confluência é processada (provavelmente na função analyze_market)
        confluence_analysis = analyze_indicator_confluence(current_data_dict)
        direct_confidence = confluence_analysis["confidence"]
        direct_direction = confluence_analysis["direction"]
        direct_reason = confluence_analysis["reason"]

        # Adicionar este bloco para mostrar mensagens claras de bloqueio
        if "WAIT FOR DEFINITION" in direct_reason and current_data_dict.get('current_candle', {}).get('indicators', {}).get('entropy', 0) > 0.80:
            # Bloquear apenas se for por entropia alta
            debug_print(f"\n❌ OPERATION BLOCKED BY HIGH ENTROPY❌: {direct_reason}")
            # resto do código...
        elif direct_confidence == 0:
            # Bloqueio por confiança zero, mas com mensagem diferente
            print(f"\n⚠️ OPERAÇÃO COM CONFIANÇA ZERO: {direct_reason}")
            print(f"   Entropia = {current_data_dict.get('current_candle', {}).get('indicators', {}).get('entropy', 'N/A')}")
            # resto do código...
            # Se quiser registrar no log
            log_candle_data(symbol, timeframe, current_timeframe_data_str, {
                "market_summary": "Ordem bloqueada - entropia muito alta",
                "confidence_level": 0,
                "direct_confidence": direct_confidence,  # Adicionar esta linha
                "llm_confidence": 0,    
                "direction": "Neutral",
                "action": "WAIT",
                "reasoning": direct_reason,
                "contracts_to_adjust": 0
            })
            
            # Retornar um valor que indique o bloqueio
            return {
                "market_summary": direct_reason,
                "confidence_level": 0,
                "direction": "Neutral",
                "action": "WAIT",
                "reasoning": "Análise direta bloqueou a operação devido a entropia alta",
                "contracts_to_adjust": 0
            }

        
        debug_print(f"🔍 Direct Confluence Analysis: Confidence={direct_confidence}, Direction={direct_direction}")
        print(f"   Razão: {direct_reason}")
    except Exception as e:
        print(f"❌ Error in direct analysis: {e}")
        direct_confidence = 0
        direct_direction = "Neutral"
        direct_reason = f"Error in direct analysis: {str(e)}"
    
    # Preparar contexto de trading aprimorado
    trading_context_obj = prepare_trading_context(symbol)
    trading_context_text = trading_context_obj["text"]
    position_direction = trading_context_obj["direction"]
    entry_price = trading_context_obj["entry_price"]
    current_price = trading_context_obj["current_price"]
    position_profit_loss = trading_context_obj["profit_loss"]
    
    # Preparar níveis de suporte e resistência
    sr_str = ", ".join([f"{level:.4f}" for level in support_resistance_levels])
    
    print("\n====== starting analissys ======")
    print(f"Symbol: {symbol}")
    print(f"Timeframe: {timeframe}")
    print("Context H4 (Narrative):")
    print(h4_market_context_summary)
    print(f"\nDados {timeframe} Atuais:")
    print(current_timeframe_data_str[:5000] + "..." if len(current_timeframe_data_str) > 5000 else current_timeframe_data_str)
    print("\nPosição Atual:")
    print(f"Direção dominante: {position_direction}")
    print(f"Preço de entrada: {entry_price:.4f}")
    print(f"Preço atual: {current_price:.4f}")
    print(f"P&L: ${position_profit_loss:.2f}")
    print("===============================================\n")
    
    try:
        with get_openai_callback() as cb:
            print("\n===== ENVIANDO DADOS PARA LLM =====")
            print("Dados sendo enviados:")
            input_data = {
                "symbol": symbol,
                "h4_context": h4_market_context_summary,
                "current_timeframe_data": current_timeframe_data_str,
                "timeframe": timeframe,
                "trading_context": trading_context_text,
                "current_position": current_position,
                "max_contracts": max_contracts,
                "support_resistance": sr_str,
                "position_direction": position_direction,
                "entry_price": entry_price,
                "current_price": current_price,
                "position_profit_loss": position_profit_loss
            }
            print(json.dumps(input_data, indent=2))
            
            response = llm_chain.invoke(input_data)
            
            print("\n===== RESPOSTA DO LLM =====")
            print(response.content)
            print("=========FIM DA RESPOSTA==============\n")
            # Log token usage
            print(f"\n===== ANÁLISE PRINCIPAL TOKEN USAGE =====")
            print(f"Prompt tokens: {cb.prompt_tokens}")
            print(f"Completion tokens: {cb.completion_tokens}")
            print(f"Total tokens: {cb.total_tokens}")
            print(f"=========================================\n")
            
            # Update global token usage
            token_usage["total_tokens"] += cb.total_tokens
            token_usage["prompt_tokens"] += cb.prompt_tokens
            token_usage["completion_tokens"] += cb.completion_tokens
        
        # Process response
        response_text = response.content
        llm_analysis = parse_llm_response(response_text)
        
        # Combinar análise do LLM com análise direta (ponderação)
        llm_confidence = llm_analysis.get('confidence_level', 0)
        
        # Podemos dar pesos diferentes para as duas análises
        # 60% análise direta, 40% análise LLM
        DIRECT_WEIGHT = 0.5   
        LLM_WEIGHT    = 0.5  #_____________________________________________________________________________________________________________________
        
        combined_confidence = int(DIRECT_WEIGHT * direct_confidence +
                                LLM_WEIGHT    * llm_confidence)
        debug_print(f"DEBUG: Calculated confidence: {combined_confidence} (Direct: {direct_confidence}, LLM: {llm_confidence})")

        try:
            # Obter timestamp do candle atual
            current_rates = mt5.copy_rates_from_pos(symbol, timeframe_dict[timeframe], 1, 1)
            if current_rates and len(current_rates) > 0:
                candle_timestamp = pd.to_datetime(current_rates[0]['time'], unit='s')
                confidence_history[candle_timestamp] = combined_confidence
                print(f"📊 Armazenando confiança {combined_confidence} para candle {candle_timestamp}")
                
                # Limpar entradas antigas (manter apenas últimos 200 candles)
                if len(confidence_history) > 200:
                    sorted_timestamps = sorted(confidence_history.keys())
                    for old_timestamp in sorted_timestamps[:-200]:
                        del confidence_history[old_timestamp]
        except Exception as e:
            print(f"⚠️ Erro ao armazenar confiança no histórico: {e}")

        # Ajustar para direção consistente
        if combined_confidence > 0:
            combined_direction = "Bullish"
        elif combined_confidence < 0:
            combined_direction = "Bearish"
        else:
            combined_direction = "Neutral"
        
        # Determinar ação baseada na confiança combinada
        if abs(combined_confidence) < 1:
            combined_action = "WAIT"
            contracts_to_adjust = 0
        else:
            combined_action = "ADD_CONTRACTS" if combined_confidence > 0 else "REMOVE_CONTRACTS"
            
            # Ajustar contratos baseado na confiança
            abs_confidence = abs(combined_confidence)
            if abs_confidence >= 60:
                contracts_factor = 0.20  # 100%
            elif abs_confidence >= 50:
                contracts_factor = 0.10  # 75%
            elif abs_confidence >= 45:
                contracts_factor = 0.05  # 50%
            elif abs_confidence >= 40:
                contracts_factor = 0.05  # 25% ______________________________________________________________________________________________________________________________________________
            else:
                contracts_factor = 0.0  # 0%
            
            contracts_to_adjust = int(max_contracts * contracts_factor)        
        # Criar reasoning combinado
        combined_reasoning = f"""
INDICATOR ANALYSIS:
{llm_analysis.get('reasoning', 'No LLM reasoning provided')}

DIRECT CONFLUENCE ANALYSIS:
{direct_reason}

CONFIDENCE CALCULATION:
- Direct rule-based confidence: {direct_confidence} ({direct_direction})
- LLM analysis confidence: {llm_confidence} ({llm_analysis.get('direction', 'Neutral')})
- Combined weighted confidence: {combined_confidence} ({combined_direction})
- Position sizing factor: {abs(combined_confidence)}% -> {int(abs(combined_confidence)/100 * max_contracts)} contracts
        """
        
        # Update global variables
        llm_reasoning = combined_reasoning
        confidence_level = combined_confidence
        market_direction = combined_direction
        
        # Construir resposta final DO PROCESSAMENTO TOTAL____________________________________________________________________________
        final_analysis = {
            "market_summary": llm_analysis.get('market_summary', 'No summary provided'),
            "confidence_level": combined_confidence,
            "direct_confidence": direct_confidence,    # Adicionando direct_confidence ao resultado
            "llm_confidence": llm_confidence, 
            "direction": combined_direction,
            "action": combined_action,
            "reasoning": combined_reasoning,
            "contracts_to_adjust": contracts_to_adjust,
            "indicator_analysis": llm_analysis.get('indicator_analysis', {})
        }
        
        #print(f"✅ Analysis complete - Confidence: {combined_confidence}, Direction: {combined_direction}")
        debug_print(f"✅ Analysis complete - Confidence: {combined_confidence}, Direction: {combined_direction}")
        
        return final_analysis
        
    except Exception as e:
        print(f"❌ Erro na análise de mercado: {e}")
        import traceback
        traceback.print_exc()
        return {
            "market_summary": "Erro na análise",
            "confidence_level": 0,
            "direction": "Neutral",
            "action": "WAIT",
            "reasoning": f"Erro na análise: {str(e)}",
            "contracts_to_adjust": 0
        }
# Atualizar o
def create_confluence_visualization(analysis):
    """
    Cria uma visualização detalhada da confluência dos indicadores
    
    Args:
        analysis (dict): Análise de mercado completa
        
    Returns:
        html.Div: Componente Dash com visualização
    """
    # Extrair dados da análise
    confidence = analysis.get('confidence_level', 0)
    direction = analysis.get('direction', 'Neutral')
    indicator_analysis = analysis.get('indicator_analysis', {})
    
    # Cores e estilos baseados na direção
    if direction == "Bullish":
        main_color = "success"
        direction_icon = "↗️"
    elif direction == "Bearish":
        main_color = "danger"
        direction_icon = "↘️"
    else:
        main_color = "secondary"
        direction_icon = "↔️"
    
    # Criar cartão principal
    header = html.Div([
        html.H4([direction_icon, f" {direction} ", direction_icon], 
                className=f"text-{main_color} text-center mb-3"),
        html.Div([
            html.Span("Confidence: "),
            html.Span(f"{confidence}", 
                     className=f"text-{main_color} fw-bold fs-4")
        ], className="text-center mb-3")
    ])
    
    # Criar cards para cada indicador
    indicator_cards = []
    
    # ATR
    if 'atr' in indicator_analysis:
        atr_card = dbc.Card([
            dbc.CardHeader("ATR (Volatility)", className="fw-bold"),
            dbc.CardBody([
                html.P(indicator_analysis['atr'], className="mb-0")
            ])
        ], className="mb-2", color="light", outline=True)
        indicator_cards.append(atr_card)
    
    # Entropy
    if 'entropy' in indicator_analysis:
        entropy_card = dbc.Card([
            dbc.CardHeader("Entropy (Directionality)", className="fw-bold"),
            dbc.CardBody([
                html.P(indicator_analysis['entropy'], className="mb-0")
            ])
        ], className="mb-2", color="light", outline=True)
        indicator_cards.append(entropy_card)
    
    # Slope
    if 'slope' in indicator_analysis:
        slope_card = dbc.Card([
            dbc.CardHeader("Slope (Momentum)", className="fw-bold"),
            dbc.CardBody([
                html.P(indicator_analysis['slope'], className="mb-0")
            ])
        ], className="mb-2", color="light", outline=True)
        indicator_cards.append(slope_card)
    
    # EMA9
    if 'ema9' in indicator_analysis:
        ema9_card = dbc.Card([
            dbc.CardHeader("EMA9 Relationship", className="fw-bold"),
            dbc.CardBody([
                html.P(indicator_analysis['ema9'], className="mb-0")
            ])
        ], className="mb-2", color="light", outline=True)
        indicator_cards.append(ema9_card)
    
    # Criar explicação de posicionamento baseado na confiança
    abs_confidence = abs(confidence)
    if abs_confidence >= 60:
        sizing_text = "20% position sizing"
        sizing_color = "success"
    elif abs_confidence >= 50:
        sizing_text = "15% position sizing"
        sizing_color = "primary"
    elif abs_confidence >= 45:
        sizing_text = "10% position sizing"
        sizing_color = "info"
    elif abs_confidence >= 40:
        sizing_text = "10% position sizing"
        sizing_color = "warning"
    else:
        sizing_text = "No position (WAIT)"
        sizing_color = "secondary"
    
    position_card = dbc.Card([
        dbc.CardHeader("Position Sizing", className="fw-bold"),
        dbc.CardBody([
            html.Div([
                html.Span(f"{sizing_text}", 
                         className=f"text-{sizing_color} fw-bold")
            ], className="text-center")
        ])
    ], className="mt-3", color=main_color, outline=True)
    
    # Se não houver dados de análise de indicadores, mostrar mensagem
    if not indicator_cards:
        indicator_cards = [html.P("No detailed indicator analysis available.", className="text-muted")]
    
    # Montar visualização completa
    return html.Div([
        header,
        html.Div(indicator_cards),
        position_card
    ])
def initialize_llm_chain():
    """
    Initialize LLM chain otimizada para análise de mercado baseada em confluência de indicadores
    """
    global llm_chain
    
    from prompts import trading_analysis_prompt  # Importar o prompt do arquivo prompts.py
    
    llm = ChatOpenAI(
        temperature=0.2,
        api_key=os.getenv("OPENAI_API_KEY"),
        model_name="gpt-4.1-mini"
    )
    
    # Criar chain
    llm_chain = (
        {
            "symbol": RunnablePassthrough(),
            "h4_context": RunnablePassthrough(),
            "current_timeframe_data": RunnablePassthrough(),
            "timeframe": RunnablePassthrough(),
            "trading_context": RunnablePassthrough(),
            "current_position": RunnablePassthrough(),
            "max_contracts": RunnablePassthrough(),
            "support_resistance": RunnablePassthrough(),
            "position_direction": RunnablePassthrough(),
            "entry_price": RunnablePassthrough(),
            "current_price": RunnablePassthrough(),
            "position_profit_loss": RunnablePassthrough()
        }
        | trading_analysis_prompt
        | llm
    )
    
    return llm_chain
def execute_mt5_order(symbol, action, contracts):
    """
    Executa ordens no MetaTrader5
    
    Args:
        symbol (str): Símbolo de trading
        action (str): ADD_CONTRACTS para comprar, REMOVE_CONTRACTS para vender
        contracts (int): Número de contratos a negociar
    """
    print("\n----- VERIFICAÇÃO DE STATUS DO MT5 -----")
    # Verificar se o MT5 está inicializado
    if not mt5.initialize():
        print(f"❌ MT5 não está inicializado! Erro: {mt5.last_error()}")
        return False
    
    # Verificar se estamos conectados
    if not mt5.terminal_info().connected:
        print("❌ MT5 não está conectado ao servidor!")
        return False
    
    # Verificar se o trading algorítmico está habilitado
    terminal_info = mt5.terminal_info()._asdict()
    if not terminal_info.get('trade_allowed', False):
        print("❌ Trading algorítmico NÃO está habilitado no terminal MT5!")
        print("Por favor, verifique se 'Permitir trading algorítmico' está ativado nas configurações do MT5.")
        return False
    else:
        print("✅ Trading algorítmico está habilitado no terminal MT5.")
    
    # Verificar se o símbolo existe
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        print(f"❌ Símbolo {symbol} não encontrado!")
        return False
    
    # Verificar se o símbolo está disponível para trading
    if not symbol_info.visible:
        print(f"❌ Símbolo {symbol} não está visível na janela de observação de mercado!")
        print("Adicionando o símbolo à janela de observação...")
        if not mt5.symbol_select(symbol, True):
            print(f"❌ Falha ao adicionar {symbol} à janela de observação!")
            return False
    
    # Imprimir detalhes da conta
    account_info = mt5.account_info()._asdict()
    print(f"Conta: {account_info.get('login')}")
    print(f"Servidor: {account_info.get('server')}")
    print(f"Saldo: {account_info.get('balance')}")
    print(f"Margem livre: {account_info.get('margin_free')}")
    
    if action == "WAIT" or contracts == 0:
        print(f"Nenhuma ação a ser executada: {action}, contratos: {contracts}")
        return False
    
    # Configurar ordem
    order_type = mt5.ORDER_TYPE_BUY if action == "ADD_CONTRACTS" else mt5.ORDER_TYPE_SELL
    
    # Obter preços atuais
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        print(f"❌ Não foi possível obter o tick para {symbol}!")
        return False
    
    price = tick.ask if order_type == mt5.ORDER_TYPE_BUY else tick.bid
    print(f"Preço atual: Ask={tick.ask}, Bid={tick.bid}")
    
    # Executar ordem
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": float(contracts),
        "type": order_type,
        "price": price,
        "deviation": 10,
        "magic": 12345,
        "comment": "Ordem enviada por Python",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_RETURN,
    }
    
    # Imprimir detalhes da ordem
    print("\n----- DETALHES DA ORDEM -----")
    print(f"Símbolo: {symbol}")
    print(f"Ação: {action}")
    print(f"Tipo: {'COMPRA' if order_type == mt5.ORDER_TYPE_BUY else 'VENDA'}")
    print(f"Contratos: {contracts}")
    print(f"Preço: {price}")
    
    # Enviar ordem
    print("\nEnviando ordem para o MT5...")
    result = mt5.order_send(request)
    
    if result is None:
        print(f"❌ Falha ao enviar ordem! Erro: {mt5.last_error()}")
        return False
    
    print(f"Resultado da ordem: {result}")
    
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"❌ Erro ao executar ordem: Código {result.retcode}")
        print(f"Descrição do erro: {mt5.last_error()}")
        
        # Códigos de erro comuns
        error_codes = {
            10004: "Requisição rejeitada pelo servidor",
            10006: "Conta desabilitada",
            10007: "Ordem rejeitada pelo servidor",
            10010: "Preço inválido",
            10011: "Volume inválido",
            10014: "Trading não permitido",
            10015: "Mercado fechado",
            10016: "Fundos insuficientes",
            10019: "Ordem modificada",
            10021: "Requisição em processamento",
            10025: "Pedido aceito para processamento",
            10026: "Solicitação aceita para processamento"
        }
        
        if result.retcode in error_codes:
            print(f"Erro conhecido: {error_codes[result.retcode]}")
        
        return False
    else:
        print(f"✅ Ordem executada com sucesso!")
        print(f"Ticket da ordem: {result.order}")
        return True
    
    print("----- FIM DO PROCESSAMENTO DA ORDEM -----\n")
    """
    Analisa a confluência dos indicadores para determinar a confiança de mercado
    baseado em regras específicas.
    
    INDICADORES E SUAS INTERPRETAÇÕES CORRETAS:
    - ATR: Mede volatilidade (alta = mercado volátil/movimentos amplos, baixa = mercado calmo)os valores medios para atr temos sendo o mais alto, indicando mais volatilidade sendo 0.0220 e o mais baixo sendo 0.0074 representando mercado completamente parado, nao queremos operar em mercado parado.
    - Entropy: Mede dispersão direcional (sendo alta >=0.8500 = mercado lateral/sem tendência, baixa <=0.70 = mercado em tendência definida, ambiente otimo para entradas, dentre 0.73 a 0.85 seria inicio de tendencia)
    - Slope: Indica direção e força da tendência (positivo = alta, negativo = baixa, próximo de zero = sem tendência) os valores medios sao comprendidos dentre -0.02118 a 0.02118 
    - EMA9: Média móvel exponencial para confirmar tendência de curto prazo, se o preco esta acima da EMA9, o mercado esta em alta, se o preco esta abaixo da EMA9, o mercado esta em baixa.
    
    LÓGICA DE CONFLUÊNCIA:
    - Quando Entropy está <=0.70, o mercado está em inicio de tendência (bullish ou bearish, entropia nao fala exatamente se e ealta ou baixa, mas sim, se o mercado está em tendência ou não)
    - A direção da tendência é determinada pelo Slope e confirmada pela posição do preço vs EMA9
    - ATR alto indica maior volatilidade, favorável para movimentos direcionais
   
    Elabore uma forma de calcular a confiança de mercado entre -100 (fortemente bearish) e +100 (fortemente bullish) com base nessas regras e suas confluencias.
    por exemplo, se ATR é alto, Entropy é baixa e Slope é positivo, a confiança deve ser alta.
    LEMBRANDO QUE MEU CODIGO FORNECE A LISTA DOS VALORES DE CADA INDICADOR, ENTAO E POSSIVEL RECEBER ESTA LISTA E IDENTIFICAR SE OS INDICADORES ESTAO AUMENTANDO OU DINIMUINDO, E COM ISSO FORTIFICAR AINDA MAIS A ANALIZE, POIS por exemplo a entropia alta nao significa nada, mas a uma media de os ultimos 5 valores ela vem crescendo ou diminuindo, diz muito sobre ela
    
    Args:
        candle_data (dict): Dados do candle atual e histórico de indicadores
        
    Returns:
        dict: Análise de confluência com nível de confiança calculado
    """
def analyze_indicator_confluence(candle_data):
    """
    Analisa a confluência dos indicadores para determinar a confiança de mercado
    baseado em regras específicas e tendências avançadas.
    
    Args:
        candle_data (dict): Dados do candle atual e histórico de indicadores
        
    Returns:
        dict: Análise de confluência com nível de confiança calculado
    """
    try:
        print("\n" + "="*80)
        print("🔍 ANÁLISE DE CONFLUÊNCIA DE INDICADORES")
        print("="*80)
        
        # Extrair valores dos indicadores
        current_indicators = candle_data.get('current_candle', {}).get('indicators', {})
        current_ohlc = candle_data.get('current_candle', {}).get('ohlc', {})
        progression = candle_data.get('progression_analysis', {})
        
        # Valores atuais
        atr = current_indicators.get('atr', 0)
        entropy = current_indicators.get('entropy', 0)
        slope = current_indicators.get('slope', 0)
        ema9 = current_indicators.get('ema9', 0)
        close = current_ohlc.get('close', 0)
        
        print(f"\n📊 VALORES ATUAIS DOS INDICADORES:")
        print(f"   ATR = {atr:.6f} | Entropy = {entropy:.6f} | Slope = {slope:.6f}")
        print(f"   EMA9 = {ema9:.4f} | Close = {close:.4f} | Price/EMA = {'ABOVE' if close > ema9 else 'BELOW'}")
        
        # Tendências do histórico - agora com suporte a tendências fortes
        atr_trend = progression.get('atr_trend', 'STABLE')
        entropy_trend = progression.get('entropy_trend', 'STABLE') 
        slope_trend = progression.get('slope_trend', 'STABLE')
        price_trend = progression.get('price_trend', 'STABLE')
        price_vs_ema = progression.get('price_vs_ema', 'NEUTRAL')
        
        print(f"\n📈 TENDÊNCIAS DOS INDICADORES:")
        print(f"   ATR Trend = {atr_trend}")
        print(f"   Entropy Trend = {entropy_trend}")
        print(f"   Slope Trend = {slope_trend}")
        print(f"   Price Trend = {price_trend}")
        print(f"   Price vs EMA = {price_vs_ema}")
        
        # Inicializar variáveis de análise
        confidence = 0
        direction = "Neutral"
        detailed_reasons = []
        indicator_scores = {}
        
        print("\n🧮 CALCULANDO SCORES DOS INDICADORES:")
        
        # ===== ANÁLISE 1: ENTROPY - QUALIDADE DA TENDÊNCIA =====
        print("\n   📊 ANÁLISE DE ENTROPIA:")
        
        # Verificação de entropia alta - condição de bloqueio
        if entropy >= 0.90:  # Extremamente lateral
            entropy_score = 0
            detailed_reasons.append(f"⚠️ Very high entropy ({entropy:.4f}) = extreme choppy/sideways market")
            detailed_reasons.append("WAIT FOR DEFINITION")  # Sinal para bloquear operações
            debug_print(f"❌ BLOCKING: Very high entropy ({entropy:.4f}) >= 0.90")
            print(f"   🚫 Score de entropia = {entropy_score}")
            print("\n⛔ OPERAÇÃO BLOQUEADA - ENTROPIA EXTREMAMENTE ALTA")
            
            return {
                "confidence": 0,
                "direction": "Neutral",
                "reason": "WAIT FOR DEFINITION - Very high entropy indicates extreme choppy market",
                "detailed_reasons": detailed_reasons,
                "indicator_scores": {"entropy": 0, "slope": 0, "ema9": 0, "atr": 0},
                "details": {
                    "entropy": entropy,
                    "slope": slope,
                    "atr": atr,
                    "ema9": ema9,
                    "close": close,
                    "price_vs_ema": price_vs_ema
                }
            }
        
        # Categorias de entropia para trading
        if entropy >= 0.80:  # Moderadamente lateral
            entropy_score = 0
            detailed_reasons.append(f"⚠️ High entropy ({entropy:.4f}) = choppy/sideways market")
            print(f"⚠️ Entropia alta ({entropy:.4f} >= 0.85): mercado lateral")
        elif entropy >= 0.75:  # Possível formação de tendência
            entropy_score = 10
            detailed_reasons.append(f"📊 Moderate-high entropy ({entropy:.4f}) = potential trend formation")
            debug_print(f"⚠️ Moderate-high entropy ({entropy:.4f} >= 0.75): possible trend formation")
        elif entropy >= 0.60:  # Tendência se estabelecendo
            entropy_score = 30
            detailed_reasons.append(f"✅ Moderate entropy ({entropy:.4f}) = established trend")
            print(f"   ✅ Entropia moderada ({entropy:.4f} >= 0.60): tendência estabelecida")
        else:  # Tendência forte
            entropy_score = 50
            detailed_reasons.append(f"🚀 Low entropy ({entropy:.4f}) = strong, well-defined trend")
            print(f"   🚀 Entropia baixa ({entropy:.4f} < 0.60): tendência forte e bem definida")

        # NOVA ANÁLISE: Tendência da entropia - indicador de formação/término de tendência
        print(f"   📊 Tendência da entropia: {entropy_trend}")
        if entropy_trend == "STRONGLY_INCREASING":
            # Entropia aumentando fortemente = tendência está se quebrando rapidamente
            entropy_score_mod = -20
            entropy_score += entropy_score_mod
            detailed_reasons.append(f"⚠️ Entropy STRONGLY INCREASING = trend breaking down quickly")
            print(f"   ⚠️ Entropia AUMENTANDO FORTEMENTE: tendência quebrando rapidamente ({entropy_score_mod})")
        elif entropy_trend == "INCREASING":
            # Entropia aumentando = tendência pode estar enfraquecendo
            entropy_score_mod = -10
            entropy_score += entropy_score_mod
            detailed_reasons.append(f"⚠️ Entropy INCREASING = trend may be weakening")
            print(f"   ⚠️ Entropia AUMENTANDO: tendência pode estar enfraquecendo ({entropy_score_mod})")
        elif entropy_trend == "STRONGLY_DECREASING":
            # Entropia diminuindo fortemente = tendência se formando rapidamente
            entropy_score_mod = 20
            entropy_score += entropy_score_mod
            detailed_reasons.append(f"🚀 Entropy STRONGLY DECREASING = trend forming rapidly")
            print(f"   🚀 Entropia DIMINUINDO FORTEMENTE: tendência se formando rapidamente ({entropy_score_mod})")
        elif entropy_trend == "DECREASING":
            # Entropia diminuindo = potencial formação de tendência
            entropy_score_mod = 10
            entropy_score += entropy_score_mod
            detailed_reasons.append(f"📈 Entropy DECREASING = potential trend formation")
            print(f"   📈 Entropia DIMINUINDO: potencial formação de tendência ({entropy_score_mod})")
        else:
            print(f"   ℹ️ Entropia estável: sem ajuste de score")
            
        print(f"   🧮 Score final de entropia = {entropy_score}")
        indicator_scores["entropy"] = entropy_score
        
        # ===== ANÁLISE 2: SLOPE - DIREÇÃO E FORÇA DA TENDÊNCIA =====
        print("\n   📊 ANÁLISE DE SLOPE (INCLINAÇÃO):")
        print(f"   📊 Valor atual: {slope:.6f}")
        
        if abs(slope) < 0.00001:  # quando a SLOPE está menor que zero é indicativo de mercado em baixa
            # Market with no clear direction
            slope_score = 0
            direction = "Neutral"
            detailed_reasons.append(
                f"↔️ Neutral slope ({slope:.6f}) = sideways market"
            )
            print(f"   ↔️ Slope neutro (|{slope:.6f}| < 0.00001): mercado lateral")
        elif slope > 0:  # positiva é indicativo de mercado em alta
            # Uptrend
            direction = "Bullish"

            if slope > 0.00010:
                slope_score = 50
                detailed_reasons.append(
                    f"📈 Very positive slope ({slope:.6f}) = STRONG uptrend"
                )
                print(f"   📈 Slope muito positivo ({slope:.6f} > 0.00010): tendência de ALTA FORTE")
            elif slope > 0.00005:
                slope_score = 30
                detailed_reasons.append(
                    f"📈 Positive slope ({slope:.6f}) = MODERATE uptrend"
                )
                print(f"   📈 Slope positivo ({slope:.6f} > 0.00005): tendência de ALTA MODERADA")
            elif slope > 0.00001:
                slope_score = 15
                detailed_reasons.append(
                    f"📈 Slightly positive slope ({slope:.6f}) = WEAK uptrend"
                )
                print(f"   📈 Slope levemente positivo ({slope:.6f} > 0.00001): tendência de ALTA FRACA")
            else:
                slope_score = 5
                detailed_reasons.append(
                    f"📈 Barely positive slope ({slope:.6f}) = nascent uptrend"
                )
                print(f"   📈 Slope quase neutro positivo ({slope:.6f}): tendência de alta nascente")
        else:
            # Downtrend
            direction = "Bearish"

            if slope < -0.00030:
                slope_score = -50
                detailed_reasons.append(
                    f"📉 Very negative slope ({slope:.6f}) = STRONG downtrend"
                )
                print(f"   📉 Slope muito negativo ({slope:.6f} < -0.00030): tendência de BAIXA FORTE")
            elif slope < -0.00005:
                slope_score = -30
                detailed_reasons.append(
                    f"📉 Negative slope ({slope:.6f}) = MODERATE downtrend"
                )
                print(f"   📉 Slope negativo ({slope:.6f} < -0.00005): tendência de BAIXA MODERADA")
            elif slope < -0.00001:
                slope_score = -15
                detailed_reasons.append(
                    f"📉 Slightly negative slope ({slope:.6f}) = WEAK downtrend"
                )
                print(f"   📉 Slope levemente negativo ({slope:.6f} < -0.00001): tendência de BAIXA FRACA")
            else:
                slope_score = -5
                detailed_reasons.append(
                    f"📉 Barely negative slope ({slope:.6f}) = nascent downtrend"
                )
                print(f"   📉 Slope quase neutro negativo ({slope:.6f}): tendência de baixa nascente")

        # NOVA ANÁLISE: Tendência da inclinação - aceleração ou desaceleração da tendência
        print(f"   📊 Tendência do slope: {slope_trend}")
        slope_score_mod = 0
        
        if direction == "Bullish":
            if slope_trend == "STRONGLY_INCREASING":
                # Inclinação positiva e aumentando fortemente = aceleração da alta
                slope_score_mod = 25
                detailed_reasons.append(f"🚀 Slope STRONGLY INCREASING = bullish momentum accelerating")
                print(f"   🚀 Slope AUMENTANDO FORTEMENTE: aceleração do momentum de alta (+{slope_score_mod})")
            elif slope_trend == "INCREASING":
                # Inclinação positiva e aumentando = continuação da alta
                slope_score_mod = 15
                detailed_reasons.append(f"📈 Slope INCREASING = bullish trend continuing")
                print(f"   📈 Slope AUMENTANDO: continuação da tendência de alta (+{slope_score_mod})")
            elif slope_trend == "DECREASING" or slope_trend == "STRONGLY_DECREASING":
                # Inclinação positiva mas diminuindo = desaceleração da alta
                slope_score_mod = -10
                detailed_reasons.append(f"⚠️ Slope DECREASING = bullish momentum slowing")
                print(f"   ⚠️ Slope DIMINUINDO: desaceleração do momentum de alta ({slope_score_mod})")
        elif direction == "Bearish":
            if slope_trend == "STRONGLY_DECREASING":
                # Inclinação negativa e diminuindo fortemente = aceleração da baixa
                slope_score_mod = -25
                detailed_reasons.append(f"🚀 Slope STRONGLY DECREASING = bearish momentum accelerating")
                print(f"   🚀 Slope DIMINUINDO FORTEMENTE: aceleração do momentum de baixa ({slope_score_mod})")
            elif slope_trend == "DECREASING":
                # Inclinação negativa e diminuindo = continuação da baixa
                slope_score_mod = -15
                detailed_reasons.append(f"📉 Slope DECREASING = bearish trend continuing")
                print(f"   📉 Slope DIMINUINDO: continuação da tendência de baixa ({slope_score_mod})")
            elif slope_trend == "INCREASING" or slope_trend == "STRONGLY_INCREASING":
                # Inclinação negativa mas aumentando = desaceleração da baixa
                slope_score_mod = 10
                detailed_reasons.append(f"⚠️ Slope INCREASING = bearish momentum slowing")
                print(f"   ⚠️ Slope AUMENTANDO: desaceleração do momentum de baixa (+{slope_score_mod})")
        
        slope_score += slope_score_mod
        print(f"   🧮 Score final de slope = {slope_score}")
        indicator_scores["slope"] = slope_score

        # ===== ANÁLISE 3: EMA9 - CONFIRMAÇÃO DA TENDÊNCIA =====
        print("\n   📊 ANÁLISE DE EMA9:")
        print(f"   📊 Close = {close:.4f}, EMA9 = {ema9:.4f}, Relação = {'ABOVE' if close > ema9 else 'BELOW'}")
        
        # ── EMA-9 CONFIRMATION ────────────────────────────────────────────────────
        if price_vs_ema == "ABOVE":
            if direction == "Bullish":
                # Perfect confirmation
                ema_score = 30
                detailed_reasons.append(
                    f"✅ Price ABOVE EMA-9 ({close:.4f} > {ema9:.4f}) = CONFIRMS bullish trend"
                )
                print(f"   ✅ Preço ACIMA da EMA-9: CONFIRMA tendência de alta (+{ema_score})")
            elif direction == "Bearish":
                # Divergence – possible upside reversal
                ema_score = 15
                detailed_reasons.append(
                    "⚠️ DIVERGENCE: Price above EMA-9 but slope is negative "
                    "= potential REVERSAL to the upside"
                )
                print(f"   ⚠️ DIVERGÊNCIA: Preço acima da EMA-9 mas slope negativo = possível REVERSÃO para cima (+{ema_score})")
            else:
                # Bullish bias in a neutral market
                ema_score = 10
                detailed_reasons.append(
                    "📊 Price above EMA-9 in a neutral market = slight bullish bias"
                )
                print(f"   📊 Preço acima da EMA-9 em mercado neutro = leve viés de alta (+{ema_score})")
        else:  # BELOW
            if direction == "Bearish":
                # Perfect confirmation
                ema_score = -30
                detailed_reasons.append(
                    f"✅ Price BELOW EMA-9 ({close:.4f} < {ema9:.4f}) = CONFIRMS bearish trend"
                )
                print(f"   ✅ Preço ABAIXO da EMA-9: CONFIRMA tendência de baixa ({ema_score})")
            elif direction == "Bullish":
                # Divergence – possible downside reversal
                ema_score = -15
                detailed_reasons.append(
                    "⚠️ DIVERGENCE: Price below EMA-9 but slope is positive "
                    "= potential REVERSAL to the downside"
                )
                print(f"   ⚠️ DIVERGÊNCIA: Preço abaixo da EMA-9 mas slope positivo = possível REVERSÃO para baixo ({ema_score})")
            else:
                # Bearish bias in a neutral market
                ema_score = -10
                detailed_reasons.append(
                    "📊 Price below EMA-9 in a neutral market = slight bearish bias"
                )
                print(f"   📊 Preço abaixo da EMA-9 em mercado neutro = leve viés de baixa ({ema_score})")

        # NOVA ANÁLISE: Tendência do preço em relação à tendência da EMA
        print(f"   📊 Tendência do preço: {price_trend}")
        ema_score_mod = 0
        
        if price_trend == "STRONGLY_INCREASING" and direction == "Bullish":
            ema_score_mod = 15
            ema_score += ema_score_mod
            detailed_reasons.append(f"🚀 Price STRONGLY INCREASING = powerful bullish momentum")
            print(f"   🚀 Preço AUMENTANDO FORTEMENTE: momentum de alta poderoso (+{ema_score_mod})")
        elif price_trend == "STRONGLY_DECREASING" and direction == "Bearish":
            ema_score_mod = -15
            ema_score += ema_score_mod
            detailed_reasons.append(f"🚀 Price STRONGLY DECREASING = powerful bearish momentum")
            print(f"   🚀 Preço DIMINUINDO FORTEMENTE: momentum de baixa poderoso ({ema_score_mod})")
        
        print(f"   🧮 Score final de EMA9 = {ema_score}")
        indicator_scores["ema9"] = ema_score
        
        # ===== ANÁLISE 4: ATR - VOLATILIDADE E FORÇA DO MOVIMENTO =====
        print("\n   📊 ANÁLISE DE ATR (VOLATILIDADE):")
        print(f"   📊 Valor atual: {atr:.6f}, Tendência: {atr_trend}")
        
        # ── ATR (Average True Range) ──────────────────────────────────────────────
        if atr_trend == "STRONGLY_INCREASING":
            # Volatilidade aumentando rapidamente
            if direction != "Neutral" and entropy < 0.75:
                # Tendência clara + volatilidade aumentando rapidamente = excelente setup
                atr_score = 25
                detailed_reasons.append(
                    f"💥 ATR STRONGLY INCREASING ({atr:.6f}) + clear trend = POWERFUL move expected"
                )
                print(f"   💥 ATR AUMENTANDO FORTEMENTE + tendência clara: movimento PODEROSO esperado (+{atr_score})")
            else:
                # Volatilidade aumentando em mercado lateral = cuidado (pode ser breakout falso)
                atr_score = -5
                detailed_reasons.append(
                    f"⚠️ ATR STRONGLY INCREASING ({atr:.6f}) in sideways market = potential false breakout"
                )
                print(f"   ⚠️ ATR AUMENTANDO FORTEMENTE em mercado lateral: potencial falso breakout ({atr_score})")
        elif atr_trend == "INCREASING":
            # Volatilidade aumentando
            if direction != "Neutral" and entropy < 0.75:
                # Tendência clara + volatilidade aumentando = bom setup
                atr_score = 20
                detailed_reasons.append(
                    f"💥 ATR INCREASING ({atr:.6f}) + clear trend = STRONG move expected"
                )
                print(f"   💥 ATR AUMENTANDO + tendência clara: movimento FORTE esperado (+{atr_score})")
            else:
                # Volatilidade aumentando em mercado lateral = cuidado
                atr_score = 0
                detailed_reasons.append(
                    f"⚠️ ATR INCREASING ({atr:.6f}) in sideways market = directionless volatility"
                )
                print(f"   ⚠️ ATR AUMENTANDO em mercado lateral: volatilidade sem direção (0)")
        elif atr_trend == "STRONGLY_DECREASING":
            # Volatilidade diminuindo rapidamente
            atr_score = -15
            detailed_reasons.append(
                f"😴 ATR STRONGLY DECREASING ({atr:.6f}) = market quickly losing momentum / consolidating"
            )
            print(f"   😴 ATR DIMINUINDO FORTEMENTE: mercado perdendo momentum rapidamente / consolidando ({atr_score})")
        elif atr_trend == "DECREASING":
            # Volatilidade diminuindo
            atr_score = -10
            detailed_reasons.append(
                f"😴 ATR DECREASING ({atr:.6f}) = market losing momentum / consolidating"
            )
            print(f"   😴 ATR DIMINUINDO: mercado perdendo momentum / consolidando ({atr_score})")
        else:
            # ATR estável
            atr_score = 0
            detailed_reasons.append(
                f"📊 ATR STABLE ({atr:.6f}) = normal volatility"
            )
            print(f"   📊 ATR ESTÁVEL: volatilidade normal (0)")

        print(f"   🧮 Score final de ATR = {atr_score}")
        indicator_scores["atr"] = atr_score
        
        # ===== CÁLCULO FINAL DE CONFIANÇA =====
        # Abordagem corrigida que separa magnitude de direção
        print("\n📊 CÁLCULO FINAL DE CONFIANÇA:")
        
        # 1. Determinar a direção baseada apenas nos indicadores direcionais (slope e EMA9)
        directional_score = indicator_scores["slope"] + indicator_scores["ema9"]
        print(f"   📏 Score direcional = {indicator_scores['slope']} (slope) + {indicator_scores['ema9']} (ema9) = {directional_score}")
        
        if directional_score > 0:
            direction = "Bullish"
            base_confidence = directional_score  # Valor positivo para bullish
            print(f"   📈 Direção determinada: BULLISH (base_confidence = {base_confidence})")
        elif directional_score < 0:
            direction = "Bearish"
            base_confidence = abs(directional_score)  # Convertemos para positivo para calcular magnitude
            print(f"   📉 Direção determinada: BEARISH (base_confidence = {base_confidence})")
        else:
            direction = "Neutral"
            base_confidence = 0
            print(f"   ↔️ Direção determinada: NEUTRAL (base_confidence = {base_confidence})")
        
        # 2. Aplicar modificadores de magnitude (entropia e ATR)
        # Estes indicadores afetam a FORÇA da confiança, mas não a direção
        magnitude_modifier = 0
        
        # Entropia (afeta qualidade da tendência)
        entropy_magnitude = abs(indicator_scores["entropy"])  # Usar valor absoluto
        magnitude_modifier += entropy_magnitude
        print(f"   📊 Modificador de magnitude da entropia: +{entropy_magnitude}")
        
        # ATR (afeta força do movimento)
        atr_magnitude = abs(indicator_scores["atr"])  # Usar valor absoluto
        magnitude_modifier += atr_magnitude
        print(f"   📊 Modificador de magnitude do ATR: +{atr_magnitude}")
        
        print(f"   📊 Modificador de magnitude total: {magnitude_modifier}")
        
        # 3. Calcular confiança final
        # A direção vem dos indicadores direcionais
        # A magnitude é baseada em todos os indicadores
        raw_confidence = base_confidence * (1 + magnitude_modifier / 100)
        print(f"   🧮 Confiança bruta = {base_confidence} * (1 + {magnitude_modifier}/100) = {raw_confidence:.2f}")
        
        # Converter de volta para positivo/negativo com base na direção
        if direction == "Bullish":
            confidence = int(round(raw_confidence))
            print(f"   📈 Confiança final (BULLISH): +{confidence}")
        elif direction == "Bearish":
            confidence = -int(round(raw_confidence))
            print(f"   📉 Confiança final (BEARISH): {confidence}")
        else:
            confidence = 0
            print(f"   ↔️ Confiança final (NEUTRAL): {confidence}")
        
        # Limitar confiança ao intervalo -100 a 100
        old_confidence = confidence
        confidence = max(-100, min(100, confidence))
        if old_confidence != confidence:
            print(f"   ⚠️ Confiança limitada ao intervalo -100 a 100: {old_confidence} → {confidence}")
        
        # Formatar razão final para um resumo conciso
        final_summary = f"{direction} market with {abs(confidence)} confidence: "
        if direction == "Bullish":
            final_summary += f"Uptrend with {'strong' if abs(confidence) > 60 else 'moderate' if abs(confidence) > 30 else 'weak'} momentum."
        elif direction == "Bearish":
            final_summary += f"Downtrend with {'strong' if abs(confidence) > 60 else 'moderate' if abs(confidence) > 30 else 'weak'} momentum."
        else:
            final_summary += "No clear directional bias."
        
        print(f"\n✅ RESUMO FINAL: {final_summary}")
        print("="*80 + "\n")
            
        # Retornar análise completa, mantendo a estrutura original
        return {
            "confidence": confidence,
            "direction": direction,
            "reason": final_summary,
            "detailed_reasons": detailed_reasons,
            "indicator_scores": indicator_scores,
            "details": {
                "entropy": entropy,
                "slope": slope,
                "atr": atr,
                "ema9": ema9,
                "close": close,
                "price_vs_ema": price_vs_ema
            }
        }
        
        
    except Exception as e:
        print(f"❌ Erro em analyze_indicator_confluence: {e}")
        import traceback
        traceback.print_exc()
        return {
            "confidence": 0,
            "direction": "Neutral",
            "reason": f"Erro na análise: {str(e)}"
        }

def update_trade_history(n_intervals):
    """Update trade history display"""
    if not trade_history:
        return html.P("No trades executed yet", className="text-muted")
    
    # Create trade history cards
    history_cards = []
    
    for i, trade in enumerate(reversed(trade_history)):  # Mostrar mais recentes primeiro
        # Determine card color based on action
        card_color = "success" if trade["action"] == "ADD_CONTRACTS" else "danger"
        
        # Format timestamp
        timestamp = trade["timestamp"]
        
        # Format PnL with color
        pnl_color = "text-success" if trade["pnl_change"] >= 0 else "text-danger"
        pnl_text = f"+${trade['pnl_change']:.2f}" if trade["pnl_change"] >= 0 else f"-${abs(trade['pnl_change']):.2f}"
        
        # Create card
        card = dbc.Card([
            dbc.CardHeader(f"Trade #{len(trade_history) - i} - {timestamp}"),
            dbc.CardBody([
                html.H5(f"{trade['action']}", className=f"text-{card_color}"),
                html.P([
                    f"Contracts: {trade['contracts']} @ {trade['price']:.5f}",
                    html.Br(),
                    html.Span(f"P&L: {pnl_text}", className=pnl_color)
                ])
            ])
        ], className="mb-2")
        
        history_cards.append(card)
    
    return history_cards
def get_initial_context(symbol, timeframe, num_candles=5):
    """Get initial market context for LLM"""
    if timeframe not in timeframe_dict:
        return "Invalid timeframe"
    
    # Get historical data
    rates = mt5.copy_rates_from_pos(symbol, timeframe_dict[timeframe], 0, num_candles)
    
    if rates is None or len(rates) == 0:
        return "No data available"
    
    # Convert to dataframe
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    
    # Calculate indicators
    df['atr'] = calculate_atr(df, period=14)
    df['entropy'] = calculate_directional_entropy(df, period=14)
    df['ema9'] = calculate_ema(df['close'], period=9)
    
    # Detect price patterns
    #patterns = detect_price_patterns(df)
    
    # Prepare context summary
    market_summary = {
        "period_analyzed": f"{num_candles} {timeframe} candles",
        "start_date": df['time'].iloc[0].strftime("%Y-%m-%d %H:%M"),
        "end_date": df['time'].iloc[-1].strftime("%Y-%m-%d %H:%M"),
        "price_range": {
            "high": float(df['high'].max()),
            "low": float(df['low'].min()),
            "current": float(df['close'].iloc[-1])
        },
        "trend_summary": {
            "direction": "Bullish" if df['close'].iloc[-1] > df['open'].iloc[0] else "Bearish",
            "strength": abs(float(df['close'].iloc[-1] - df['open'].iloc[0])) / float(df['atr'].iloc[-1]) if not np.isnan(df['atr'].iloc[-1]) and df['atr'].iloc[-1] != 0 else 0,
        },
        "volatility": {
            "average_atr": float(df['atr'].mean()) if not np.isnan(df['atr'].mean()) else 0,
            "entropy": float(df['entropy'].mean()) if not np.isnan(df['entropy'].mean()) else 0
        },
        "key_levels": {
            # "recent_high": float(df['high'].max()),
            # "recent_low": float(df['low'].min()),
            "ema9": float(df['ema9'].iloc[-1]) if not np.isnan(df['ema9'].iloc[-1]) else 0
        },
       # "price_patterns": patterns
    }
    
    return json.dumps(market_summary, indent=2)

def update_indicator_history(symbol, timeframe, max_history=20):
    """
    Atualiza o buffer de histórico de indicadores de forma dinâmica
    Mantém os últimos max_history valores para progressão temporal
    """
    global indicator_history_buffer
    
    try:
        # Obter apenas o candle mais recente + buffer para indicadores
        total_needed = 20  # Buffer para calcular indicadores
        rates = mt5.copy_rates_from_pos(symbol, timeframe_dict[timeframe], 0, total_needed)
        
        if rates is None or len(rates) < 15:
            return False
        
        # Converter para DataFrame
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        
        # Calcular indicadores
        df['atr'] = calculate_atr(df, period=14)
        df['entropy'] = calculate_directional_entropy(df, period=14)
        df['ema9'] = calculate_ema(df['close'], period=9)
        df['slope'] = calculate_slope(df['close'], period=14)
        
        # Ordenar por tempo
        df = df.sort_values('time').reset_index(drop=True)
        
        # Pegar apenas candles com indicadores válidos
        valid_df = df.iloc[14:].reset_index(drop=True)
        
        if len(valid_df) == 0:
            return False
        
        # Pegar o candle mais recente (último)
        latest_candle = valid_df.iloc[-1]
        latest_timestamp = latest_candle['time'].strftime("%Y-%m-%d %H:%M")
        
        # Verificar se este timestamp já existe no buffer (evitar duplicatas)
        if latest_timestamp in indicator_history_buffer["timestamps"]:
            print(f"⚠️  Timestamp {latest_timestamp} já existe no buffer. Pulando...")
            return False
        
        # Adicionar novos valores ao buffer
        indicator_history_buffer["atr"].append(round(float(latest_candle['atr']), 4))
        indicator_history_buffer["entropy"].append(round(float(latest_candle['entropy']), 4))
        indicator_history_buffer["slope"].append(round(float(latest_candle['slope']), 5))  # Slope precisa mais precisão
        indicator_history_buffer["ema9"].append(round(float(latest_candle['ema9']), 4))
        indicator_history_buffer["close"].append(round(float(latest_candle['close']), 4))
            
            # Manter apenas os últimos max_history valores
        for key in indicator_history_buffer:
            if len(indicator_history_buffer[key]) > max_history:
                indicator_history_buffer[key] = indicator_history_buffer[key][-max_history:]
        
        print(f"📈 Buffer atualizado: {len(indicator_history_buffer['timestamps'])} candles no histórico")
        print(f"   Último timestamp: {latest_timestamp}")
        print(f"   ATR: {latest_candle['atr']:.6f}, Entropy: {latest_candle['entropy']:.6f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro ao atualizar histórico: {e}")
        return False

def initialize_indicator_history(symbol, timeframe, initial_candles=15):
    """
    Inicializa o buffer de histórico com os candles iniciais
    Deve ser chamado apenas uma vez no início
    """
    global indicator_history_buffer
    
    print(f"🔄 Initializing indicator buffer for {symbol} {timeframe}...")
    debug_print(f"🔄 Initializing indicator buffer for {symbol} {timeframe}...")
    
    # Limpar buffer existente
    indicator_history_buffer = {
        "atr": [],
        "entropy": [], 
        "slope": [],
        "ema9": [],
        "close": [],
        "timestamps": []
    }
    
    try:
        # Obter dados históricos suficientes
        total_needed = initial_candles + 14  # + buffer para indicadores
        rates = mt5.copy_rates_from_pos(symbol, timeframe_dict[timeframe], 0, total_needed)
        
        if rates is None or len(rates) < 15:
            print("❌ Dados insuficientes para inicializar buffer")
            return False
        
        # Converter para DataFrame
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        
        # Calcular indicadores
        df['atr'] = calculate_atr(df, period=14)
        df['entropy'] = calculate_directional_entropy(df, period=14)
        df['ema9'] = calculate_ema(df['close'], period=9)
        df['slope'] = calculate_slope(df['close'], period=14)
        
        # Ordenar por tempo
        df = df.sort_values('time').reset_index(drop=True)
        
        # Pegar apenas candles com indicadores válidos
        valid_df = df.iloc[14:].reset_index(drop=True)
        
        # Pegar os últimos initial_candles para inicializar o buffer
        initial_data = valid_df.tail(initial_candles)
        
  
        for i in range(len(initial_data)):
            candle = initial_data.iloc[i]
            
            indicator_history_buffer["atr"].append(round(float(candle['atr']), 4))
            indicator_history_buffer["entropy"].append(round(float(candle['entropy']), 4))
            indicator_history_buffer["slope"].append(round(float(candle['slope']), 6))  # Slope precisa mais precisão
            indicator_history_buffer["ema9"].append(round(float(candle['ema9']), 4))
            indicator_history_buffer["close"].append(round(float(candle['close']), 4))
            indicator_history_buffer["timestamps"].append(candle['time'].strftime("%Y-%m-%d %H:%M"))

        print(f"✅ Buffer inicializado com {len(indicator_history_buffer['timestamps'])} candles")
        print(f"   Período: {indicator_history_buffer['timestamps'][0]} a {indicator_history_buffer['timestamps'][-1]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro ao inicializar buffer: {e}")
        return False
    

def calculate_trend(values, min_periods=10, significance_threshold=0.015, indicator_name="indicator"):
    """
    Realiza análise de tendência avançada usando múltiplas técnicas
    MODIFICADA para retornar análise qualitativa para LLM
    
    Args:
        values: Lista de valores do indicador
        min_periods: Número mínimo de períodos para análise (default: 6)
        significance_threshold: Limiar percentual para considerar uma mudança significativa (default: 1.5%)
        indicator_name: Nome do indicador para contexto qualitativo
        
    Returns:
        dict: {"trend": str, "analysis": str} com tendência e análise qualitativa
    """
    import numpy as np
    
    # Verificar se temos dados suficientes
    if len(values) < min_periods:
        return {
            "trend": "INSUFFICIENT_DATA",
            "analysis": f"Insufficient data for {indicator_name} analysis. Need at least {min_periods} periods."
        }
    
    # Pegar os últimos valores para análise
    recent_values = np.array(values[-min_periods:])
    
    # 1. Análise de regressão linear para tendência
    x = np.arange(len(recent_values))
    slope, intercept = np.polyfit(x, recent_values, 1)
    
    # Calcular o coeficiente de determinação (R²) para medir a qualidade da tendência
    y_pred = slope * x + intercept
    ss_total = np.sum((recent_values - np.mean(recent_values))**2)
    ss_residual = np.sum((recent_values - y_pred)**2)
    r_squared = 1 - (ss_residual / ss_total) if ss_total != 0 else 0
    
    # 2. Cálculo de mudança percentual do início ao fim
    percent_change = (recent_values[-1] - recent_values[0]) / recent_values[0] if recent_values[0] != 0 else 0
    
    # 3. Análise de consistência direcional
    # Calcular as mudanças entre períodos consecutivos
    period_changes = np.diff(recent_values)
    up_periods = np.sum(period_changes > 0)
    down_periods = np.sum(period_changes < 0)
    flat_periods = np.sum(period_changes == 0)
    
    # Consistência direcional (proporção dos períodos na direção predominante)
    consistency = max(up_periods, down_periods) / (len(period_changes)) if len(period_changes) > 0 else 0
    
    # 4. Análise de aceleração (segunda derivada)
    # Verificar se a tendência está acelerando ou desacelerando
    if len(period_changes) > 1:
        acceleration = np.diff(period_changes)
        accel_up = np.sum(acceleration > 0)
        accel_down = np.sum(acceleration < 0)
        is_accelerating = accel_up > accel_down
    else:
        is_accelerating = False
    
    # Determinar tendência
    if abs(percent_change) < significance_threshold:
        trend = "STABLE"
    elif percent_change > 0 and (consistency > 0.6 or r_squared > 0.5):
        trend = "STRONGLY_INCREASING" if is_accelerating and consistency > 0.8 else "INCREASING"
    elif percent_change < 0 and (consistency > 0.6 or r_squared > 0.5):
        trend = "STRONGLY_DECREASING" if is_accelerating and consistency > 0.8 else "DECREASING"
    else:
        trend = "STABLE"
    
    # Formatar valores para análise qualitativa
    values_str = ", ".join([f"{v:.6f}" for v in recent_values])
    percent_str = f"{percent_change * 100:.1f}"
    consistency_str = f"{consistency * 100:.0f}"
    
    # Criar análise qualitativa baseada no tipo de indicador
    if indicator_name == "atr":
        # ATR mede volatilidade - contexto sobre amplitude de movimento
        if trend == "STRONGLY_INCREASING":
            analysis = f"The ATR increased {percent_str}% over the last {min_periods} candles [{values_str}], with {consistency_str}% directional consistency. This indicates rapidly expanding volatility and stronger price movements, suggesting a powerful trend is developing or a breakout is occurring."
        elif trend == "INCREASING":
            analysis = f"The ATR increased {percent_str}% over the last {min_periods} candles [{values_str}], with {consistency_str}% directional consistency. This shows growing volatility and expanding price ranges, indicating the market is becoming more active."
        elif trend == "DECREASING":
            analysis = f"The ATR decreased {abs(float(percent_str))}% over the last {min_periods} candles [{values_str}], with {consistency_str}% directional consistency. This shows contracting volatility and narrowing price ranges, suggesting market consolidation or reduced trading activity."
        elif trend == "STRONGLY_DECREASING":
            analysis = f"The ATR decreased {abs(float(percent_str))}% over the last {min_periods} candles [{values_str}], with {consistency_str}% directional consistency. This indicates rapidly contracting volatility, suggesting the market is entering a tight consolidation phase or losing momentum significantly."
        else:
            analysis = f"The ATR remained stable around {recent_values[-1]:.6f} over the last {min_periods} candles [{values_str}], with only {percent_str}% change. This indicates consistent volatility levels without significant expansion or contraction."
        debug_print(analysis)
    elif indicator_name == "entropy":
        # Entropy mede qualidade/clareza da tendência - contexto sobre direção do mercado
        if trend == "STRONGLY_INCREASING":
            analysis = f"The Entropy increased {percent_str}% over the last {min_periods} candles [{values_str}], with {consistency_str}% directional consistency. This indicates the market is rapidly becoming more chaotic and directionless, with the trend breaking down quickly - a very choppy/sideways market is forming."
        elif trend == "INCREASING":
            analysis = f"The Entropy increased {percent_str}% over the last {min_periods} candles [{values_str}], with {consistency_str}% directional consistency. This shows the market is becoming more random and less trendy, suggesting the current trend may be weakening."
        elif trend == "DECREASING":
            analysis = f"The Entropy decreased {abs(float(percent_str))}% over the last {min_periods} candles [{values_str}], with {consistency_str}% directional consistency. This shows the market is becoming more ordered and directional, suggesting a trend is forming or strengthening."
        elif trend == "STRONGLY_DECREASING":
            analysis = f"The Entropy decreased {abs(float(percent_str))}% over the last {min_periods} candles [{values_str}], with {consistency_str}% directional consistency. This indicates the market is rapidly becoming more directional and organized - a strong, clear trend is emerging quickly."
        else:
            analysis = f"The Entropy remained stable around {recent_values[-1]:.6f} over the last {min_periods} candles [{values_str}], with only {percent_str}% change. This indicates the market maintains its current state of organization/randomness without significant changes."
        debug_print(analysis)
    elif indicator_name == "slope":
        # Slope mede direção e força da tendência - contexto sobre momentum
        if trend == "STRONGLY_INCREASING":
            analysis = f"The Slope increased {percent_str}% over the last {min_periods} candles [{values_str}], with {consistency_str}% directional consistency. This indicates rapidly accelerating upward momentum - the bullish trend is gaining significant strength and speed."
        elif trend == "INCREASING":
            analysis = f"The Slope increased {percent_str}% over the last {min_periods} candles [{values_str}], with {consistency_str}% directional consistency. This shows strengthening upward momentum, with the trend gradually becoming more bullish."
        elif trend == "DECREASING":
            analysis = f"The Slope decreased {percent_str}% over the last {min_periods} candles [{values_str}], with {consistency_str}% directional consistency. This shows weakening momentum or increasing bearish pressure in the market."
        elif trend == "STRONGLY_DECREASING":
            analysis = f"The Slope decreased {percent_str}% over the last {min_periods} candles [{values_str}], with {consistency_str}% directional consistency. This indicates rapidly accelerating downward momentum - the bearish trend is gaining significant strength."
        else:
            analysis = f"The Slope remained stable around {recent_values[-1]:.6f} over the last {min_periods} candles [{values_str}], with only {percent_str}% change. This indicates consistent directional momentum without acceleration or deceleration."
        debug_print(analysis)
    elif indicator_name == "price":
        # Price trend - contexto sobre movimento de preço
        if trend == "STRONGLY_INCREASING":
            analysis = f"The Price increased {percent_str}% over the last {min_periods} candles [{values_str}], with {consistency_str}% directional consistency. This shows a strong bullish price movement with accelerating upward momentum."

        elif trend == "INCREASING":
            analysis = f"The Price increased {percent_str}% over the last {min_periods} candles [{values_str}], with {consistency_str}% directional consistency. This indicates a steady upward price movement."
        elif trend == "DECREASING":
            analysis = f"The Price decreased {abs(float(percent_str))}% over the last {min_periods} candles [{values_str}], with {consistency_str}% directional consistency. This shows a steady downward price movement."
        elif trend == "STRONGLY_DECREASING":
            analysis = f"The Price decreased {abs(float(percent_str))}% over the last {min_periods} candles [{values_str}], with {consistency_str}% directional consistency. This shows a strong bearish price movement with accelerating downward momentum."
        else:
            analysis = f"The Price remained stable around {recent_values[-1]:.4f} over the last {min_periods} candles [{values_str}], with only {percent_str}% change. This indicates a sideways/ranging market without clear directional bias."
            debug_print(analysis)
    else:
        # Análise genérica para outros indicadores
        analysis = f"The {indicator_name} changed {percent_str}% over the last {min_periods} candles [{values_str}], with {consistency_str}% directional consistency. Current trend: {trend}."
        debug_print(analysis)
    return {
        "trend": trend,
        "analysis": analysis
    }
def get_current_candle_data(symbol, timeframe):
    """
    NOVA VERSÃO: Obtém dados do candle atual + histórico dinâmico de indicadores
    """
    global indicator_history_buffer
    
    try:
        # Atualizar o buffer com o novo candle (se houver)
        update_indicator_history(symbol, timeframe)
        
        # Verificar se temos dados suficientes
        if len(indicator_history_buffer["timestamps"]) < 10:  # Aumentado para 10 para suportar a nova análise de tendência
            return "Histórico insuficiente - aguardando mais candles"
        
        # Obter o candle que acabou de fechar 
        current_atr = indicator_history_buffer["atr"][-2]
        current_entropy = indicator_history_buffer["entropy"][-2]
        current_slope = indicator_history_buffer["slope"][-2]
        current_ema9 = indicator_history_buffer["ema9"][-2]
        current_close = indicator_history_buffer["close"][-2]
        current_timestamp = indicator_history_buffer["timestamps"][-2]
        
        # Obter histórico (todos exceto o último)
        historical_atr = indicator_history_buffer["atr"][:-2]
        historical_entropy = indicator_history_buffer["entropy"][:-2]
        historical_slope = indicator_history_buffer["slope"][:-2]
        historical_ema9 = indicator_history_buffer["ema9"][:-2]
        historical_close = indicator_history_buffer["close"][:-2]
        historical_timestamps = indicator_history_buffer["timestamps"][:-2]
        
        # Obter dados OHLC do candle atual
        current_rates = mt5.copy_rates_from_pos(symbol, timeframe_dict[timeframe], 1, 1)
        if current_rates is None or len(current_rates) == 0:
            return "Erro ao obter dados do candle atual"
        
        current_ohlc = current_rates[0]
        
        # Detectar padrões nos últimos candles
        recent_closes = indicator_history_buffer["close"][-5:]  # Últimos 5 candles
        patterns = []
        
        if len(recent_closes) >= 3:
            if recent_closes[-1] > recent_closes[-2] > recent_closes[-3]:
                patterns.append("BULLISH_SEQUENCE")
            elif recent_closes[-1] < recent_closes[-2] < recent_closes[-3]:
                patterns.append("BEARISH_SEQUENCE")
        
        # Análise de tendências com contexto qualitativo para LLM
        # IMPORTANTE: calculate_trend deve estar definida FORA desta função
        atr_analysis = calculate_trend(indicator_history_buffer["atr"], indicator_name="atr")
        entropy_analysis = calculate_trend(indicator_history_buffer["entropy"], indicator_name="entropy")
        slope_analysis = calculate_trend(indicator_history_buffer["slope"], indicator_name="slope")
        price_analysis = calculate_trend(indicator_history_buffer["close"], indicator_name="price")
        
        # Preparar dados estruturados
        candle_data = {
            "current_candle": {
                "timestamp": current_timestamp,
                "timeframe": timeframe,
                "ohlc": {
                    "open": float(current_ohlc['open']),
                    "high": float(current_ohlc['high']),
                    "low": float(current_ohlc['low']),
                    "close": float(current_ohlc['close'])
                },
                "volume": float(current_ohlc['tick_volume']),
                # "indicators": {
                #     "atr": round(current_atr, 4),
                #     "entropy": round(current_entropy, 4),
                #     "ema9": round(current_ema9, 4),
                #     "slope": round(current_slope, 5)
                # },
                # "patterns": patterns  # Adicionar padrões detectados
            },
            
            # HISTÓRICO DINÂMICO - CHAVE DO SISTEMA
            # "indicator_progression": {
            #     "atr_values": [round(val, 4) for val in historical_atr],
            #     "entropy_values": [round(val, 4) for val in historical_entropy],
            #     "slope_values": [round(val, 5) for val in historical_slope],
            #     "ema9_values": [round(val, 4) for val in historical_ema9],
            #     "close_values": [round(val, 4) for val in historical_close],
            #     "timestamps": historical_timestamps,
            #     "total_periods": len(historical_timestamps)
            # },
            
            # Análise de tendências com resultados simples
            "progression_analysis": {
                "atr_trend": atr_analysis["trend"],
                "entropy_trend": entropy_analysis["trend"],
                "slope_trend": slope_analysis["trend"],
                "price_trend": price_analysis["trend"],
                "price_vs_ema": "ABOVE" if current_close > current_ema9 else "BELOW"
            },
            
            # NOVO: Análise qualitativa detalhada para LLM
            "trend_analysis_details": {
                "atr": atr_analysis["analysis"],
                "entropy": entropy_analysis["analysis"],
                "slope": slope_analysis["analysis"],
                "price": price_analysis["analysis"],
                "price_vs_ema_context": f"The current price ({current_close:.4f}) is {'above' if current_close > current_ema9 else 'below'} the EMA9 ({current_ema9:.4f}) by {abs(current_close - current_ema9):.4f} points ({abs((current_close - current_ema9) / current_ema9 * 100):.2f}%), indicating {'bullish' if current_close > current_ema9 else 'bearish'} short-term sentiment."
            }
        }
        
        return json.dumps(candle_data, indent=2)
        
    except Exception as e:
        print(f"❌ Erro em get_current_candle_data: {e}")
        import traceback
        traceback.print_exc()
        return f"Erro ao obter dados: {str(e)}"


def prepare_trading_context(symbol):
    """lida apenas com informações de posição do MT5 Prepara contexto adicional de trading com informações detalhadas da posição atual"""
    
    # Verificar posições existentes
    positions = mt5.positions_get(symbol=symbol)
    position_info = "No open positions"
    position_direction = "NEUTRAL"
    position_entry_price = 0
    position_profit_loss = 0
    position_duration = "0 minutes"
    current_price = 0
    
    # Verificar preço atual
    tick = mt5.symbol_info_tick(symbol)
    if tick:
        current_price = (tick.bid + tick.ask) / 2
    
    if positions and len(positions) > 0:
        # Identificar posição inicial (mais antiga)
        earliest_pos = min(positions, key=lambda p: p.time)
        initial_entry_price = earliest_pos.price_open
        initial_entry_time = datetime.fromtimestamp(earliest_pos.time)
        
        # Determinar direção dominante da posição
        buy_volume = sum(p.volume for p in positions if p.type == mt5.ORDER_TYPE_BUY)
        sell_volume = sum(p.volume for p in positions if p.type == mt5.ORDER_TYPE_SELL)
        
        if buy_volume > sell_volume:
            position_direction = "LONG"
        elif sell_volume > buy_volume:
            position_direction = "SHORT"
        else:
            position_direction = "NEUTRAL"
        
        # Calcular duração da posição
        now = datetime.now()
        duration = now - initial_entry_time
        hours, remainder = divmod(duration.total_seconds(), 3600)
        minutes, _ = divmod(remainder, 60)
        position_duration = f"{int(hours)} hours {int(minutes)} minutes"
        
        # Calcular profit/loss atual
        position_profit_loss = sum(p.profit for p in positions)
        
        # Formatar informação detalhada da posição
        position_info = f"Open {position_direction} position since {initial_entry_time.strftime('%Y-%m-%d %H:%M')} at {initial_entry_price:.4f}\n"
        position_info += f"Position duration: {position_duration}\n"
        position_info += f"Current P&L: ${position_profit_loss:.2f}"
        
        # Calcular movimento de preço desde a entrada
        if current_price > 0 and initial_entry_price > 0:
            price_change_pct = (current_price - initial_entry_price) / initial_entry_price * 100
            position_info += f"\nPrice movement: {price_change_pct:.2f}% from entry"
    
    # Histórico de ajustes recentes
    recent_adjustments = ""
    if trade_history and len(trade_history) > 0:
        last_3_trades = trade_history[-3:]
        recent_adjustments = "Recent adjustments: " + "; ".join([
            f"{trade['action']} {trade['contracts']} @ {trade['price']:.4f}"
            for trade in last_3_trades
        ])
    
    # Retornar contexto detalhado
    return {
        "text": f"""Current Position Status: {position_info}\n{recent_adjustments}\nRisk Management: Using confidence-based position sizing""",
        "direction": position_direction,
        "entry_price": initial_entry_price if positions and len(positions) > 0 else 0,
        "current_price": current_price,
        "profit_loss": position_profit_loss,
        "duration": position_duration
    }

  
def execute_trade(symbol, action, contracts_to_adjust):
    """Execute trade on MetaTrader5 with improved validation and retry logic"""
    print(f"\n🚀 === execute_trade INICIADA ===")
    print(f"Symbol: {symbol}")
    print(f"Action: {action}")
    print(f"Contracts: {contracts_to_adjust}")
    global current_position, total_pnl, trade_history, confidence_level
    
    # Check if we need to adjust position
    if action == "WAIT" or contracts_to_adjust == 0:
        print(f"❌ BLOQUEADO: action={action}, contracts={contracts_to_adjust}")
        return "No trade executed"
    
    # Obter informações sobre a posição atual
    trading_context_obj = prepare_trading_context(symbol)
    position_direction = trading_context_obj["direction"]
    position_profit_loss = trading_context_obj["profit_loss"]
    entry_price = trading_context_obj["entry_price"]
    current_price = trading_context_obj["current_price"]
    
    print(f"✅ PASSOU dos filtros iniciais")
    print(f"📊 Confidence Level: {confidence_level}")
    print(f"📈 Posição atual: {position_direction} com P&L ${position_profit_loss:.2f}")
    
    # VALIDAÇÃO 1: Verificar conexão MT5
    if not mt5.terminal_info():
        print("❌ MetaTrader5 não está conectado")
        if not initialize_mt5():
            return "Failed to connect to MetaTrader5"
    
    # VALIDAÇÃO 2: Verificar informações da conta
    account_info = mt5.account_info()
    if account_info is None:
        return "Failed to get account information"
    
    print(f"💰 Account Balance: ${account_info.balance:.2f}")
    print(f"💰 Free Margin: ${account_info.margin_free:.2f}")
    
    # Get current symbol info
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        return f"Symbol {symbol} not found"
    
    # Verificar se o símbolo está habilitado para trading
    if not symbol_info.visible:
        print(f"Symbol {symbol} is not visible, trying to enable...")
        if not mt5.symbol_select(symbol, True):
            return f"Failed to select symbol {symbol}"
    
    # VALIDAÇÃO 3: Verificar se o mercado está aberto
    if not symbol_info.trade_mode == mt5.SYMBOL_TRADE_MODE_FULL:
        return f"Market is closed or restricted for {symbol}"
    
    # Determinar o tipo de instrumento e fazer conversão apropriada
    contract_size = symbol_info.trade_contract_size
    min_volume = symbol_info.volume_min
    volume_step = symbol_info.volume_step
    max_volume = symbol_info.volume_max
    
    print(f"📈 Symbol specifications:")
    print(f"   Contract size: {contract_size}")
    print(f"   Min volume: {min_volume}")
    print(f"   Volume step: {volume_step}")
    print(f"   Max volume: {max_volume}")
    
    # Converter contratos para volume MT5
    if "Forex" in symbol_info.path or symbol in ["EURUSD", "GBPUSD", "USDJPY", "EURJPY"]:
        # Para Forex, assumir que 1 contrato = 0.01 lot
        base_volume = contracts_to_adjust * 0.01
    else:
        # Para outros instrumentos (futuros), usar diretamente
        base_volume = contracts_to_adjust * min_volume
    
    # Arredondar para o volume_step mais próximo
    volume = round(base_volume / volume_step) * volume_step
    
    # Garantir que está dentro dos limites
    volume = max(min_volume, min(volume, max_volume))
    
    print(f"📊 Volume calculation:")
    print(f"   Requested contracts: {contracts_to_adjust}")
    print(f"   Final volume: {volume}")
    
    # VALIDAÇÃO 4: Verificar margem necessária
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        return "Failed to get current prices"
    
    # Verificar spread
    spread = tick.ask - tick.bid
    spread_points = spread / symbol_info.point
    print(f"📊 Current spread: {spread_points:.1f} points")
    
    # VALIDAÇÃO 5: Verificar spread máximo aceitável
    MAX_SPREAD_POINTS = 50
    if spread_points > MAX_SPREAD_POINTS:
        return f"Spread too high: {spread_points:.1f} points (max: {MAX_SPREAD_POINTS})"
    
    # Verificar posições existentes
    positions = mt5.positions_get(symbol=symbol)
    print(f"Current positions for {symbol}: {len(positions) if positions else 0}")
    
    # Calcular exposição total atual
    current_exposure = 0
    if positions:
        for position in positions:
            current_exposure += position.volume
    
    # VALIDAÇÃO 6: Verificar limite máximo de exposição
    MAX_EXPOSURE = max_contracts
    if current_exposure + volume > MAX_EXPOSURE:
        return f"Maximum exposure would be exceeded. Current: {current_exposure}, Requested: {volume}, Max: {MAX_EXPOSURE}"
    debug_print(f"Maximum exposure would be exceeded. Current: {current_exposure}, Requested: {volume}, Max: {MAX_EXPOSURE}")
    # Prepare trade request
    trade_type = mt5.ORDER_TYPE_BUY if action == "ADD_CONTRACTS" else mt5.ORDER_TYPE_SELL
    
    # Obter preço atual
    if trade_type == mt5.ORDER_TYPE_BUY:
        price = tick.ask
    else:
        price = tick.bid
    
    print(f"Current prices - Bid: {tick.bid}, Ask: {tick.ask}")
    
    # Create trade request
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": float(volume),
        "type": trade_type,
        "price": price,
        "deviation": 20,
        "magic": 123456,
        "comment": f"LLM Bot {'Buy' if trade_type == mt5.ORDER_TYPE_BUY else 'Sell'}",
        "type_time": mt5.ORDER_TIME_GTC,
    }
    
    # Determinar o tipo de preenchimento
    filling_type = None
    filling_modes = symbol_info.filling_mode
    
    if filling_modes & mt5.ORDER_FILLING_FOK:
        filling_type = mt5.ORDER_FILLING_FOK
    elif filling_modes & mt5.ORDER_FILLING_IOC:
        filling_type = mt5.ORDER_FILLING_IOC
    elif filling_modes & mt5.ORDER_FILLING_RETURN:
        filling_type = mt5.ORDER_FILLING_RETURN
    
    if filling_type is not None:
        request["type_filling"] = filling_type
    
    print(f"Enviando pedido de trade: {request}")
    
    # IMPLEMENTAR RETRY LOGIC
    MAX_RETRIES = 3
    RETRY_DELAY = 1
    
    result = None
    last_error = ""
    
    for attempt in range(MAX_RETRIES):
        if attempt > 0:
            print(f"🔄 Tentativa {attempt + 1} de {MAX_RETRIES}...")
            time.sleep(RETRY_DELAY)
            
            # Atualizar preço antes de nova tentativa
            tick = mt5.symbol_info_tick(symbol)
            if tick:
                request["price"] = tick.ask if trade_type == mt5.ORDER_TYPE_BUY else tick.bid
        
        # Send trade order
        result = mt5.order_send(request)
        
        if result is None:
            error_code = mt5.last_error()
            last_error = f"Falha ao enviar ordem: Erro {error_code[0]} - {error_code[1]}"
            print(f"❌ Tentativa {attempt + 1}: {last_error}")
            continue
        

        else:
            error_code = result.retcode
            comment = result.comment if hasattr(result, 'comment') else "No comment"
            last_error = f"Order failed, retcode={error_code}, comment={comment}"
            print(f"❌ Tentativa {attempt + 1}: {last_error}")
            
            if error_code in [mt5.TRADE_RETCODE_INVALID_PRICE, mt5.TRADE_RETCODE_PRICE_CHANGED]:
                continue
            elif error_code == mt5.TRADE_RETCODE_INVALID_VOLUME:
                request["volume"] = min_volume
                continue
            else:
                break
    
    # Verificar se teve sucesso
    if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"❌ FALHA FINAL após {MAX_RETRIES} tentativas: {last_error}")
        return last_error
    
    # ✅ SUCESSO NO MT5 - AGORA EXPORTAR PARA NINJATRADER
    print(f"✅ Order executed successfully in MT5")
    print(f"   Deal: {result.deal}, Order: {result.order}")
    print(f"   Executed volume: {result.volume}")
    
    # IMPORTANTE: Exportar para NinjaTrader ANTES do return
# IMPORTANTE: Exportar para NinjaTrader ANTES do return
    try:
        print("\n📤 Exportando ordem para NinjaTrader...")
        
        # Garantir que contracts_to_adjust seja um inteiro
        int_contracts = int(contracts_to_adjust)
        print(f"DEBUG NINJA: Convertendo contratos de {contracts_to_adjust} para {int_contracts}")
        
        ninja_result = save_ninjatrader_command(
            action=action,
            contracts=int_contracts,  # Explicitamente convertido para inteiro
            symbol=symbol
        )
        
        if ninja_result["status"] == "success":
            print(f"✅ NinjaTrader Export SUCCESS")
            print(f"   Command: {ninja_result['command']}")
            print(f"   File: {ninja_result['file_path']}")
        else:
            print(f"⚠️ NinjaTrader Export FAILED: {ninja_result.get('reason', 'Unknown error')}")
            
    except Exception as e:
        print(f"❌ ERRO ao exportar para NinjaTrader: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # Atualizar rastreamento de posição
    executed_volume = result.volume if hasattr(result, 'volume') else volume
    
    if action == "ADD_CONTRACTS":
        current_position += executed_volume
    else:  # REMOVE_CONTRACTS
        current_position -= executed_volume
    
    print(f"📊 Position updated: {current_position} lots")
    
    # Calcular P&L
    pnl_change = 0
    positions = mt5.positions_get(symbol=symbol)
    if positions and len(positions) > 0:
        pnl_change = sum(pos.profit for pos in positions)
        total_pnl = pnl_change
        print(f"Actual P&L from positions: ${pnl_change:.2f}")
    
    # Adicionar ao histórico
    new_trade = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "action": action,
        "contracts": contracts_to_adjust,  # Guardar contratos originais
        "mt5_volume": executed_volume,      # Guardar volume MT5
        "price": result.price,
        "pnl_change": pnl_change,
        "ninja_exported": ninja_result["status"] == "success" if 'ninja_result' in locals() else False
    }
    trade_history.append(new_trade)
    
    success_message = f"✅ Trade Complete: {action} {contracts_to_adjust} contracts (MT5: {executed_volume} lots) @ {result.price}"
    print(f"\n{success_message}")
    
    return success_message



def check_ninjatrader_response():
    """
    Verifica se o NinjaTrader executou as ordens
    """
    response_file = "C:/ProgramData/NinjaTrader 8/templates/Order Export/ninja_responses.txt"
    
    if os.path.exists(response_file):
        try:
            with open(response_file, "r") as f:
                responses = f.readlines()
            
            # Processar últimas respostas
            for response in responses[-5:]:  # Últimas 5 respostas
                print(f"📋 Resposta NT: {response.strip()}")
                
        except Exception as e:
            print(f"Erro ao ler respostas: {e}")
    else:
        print("⚠️ Arquivo de resposta do NinjaTrader não encontrado")

def trading_loop(symbol, timeframe):
    """Main trading loop with dynamic history"""
    global running, trade_history, current_position, total_pnl
    
    print(f"Starting trading loop for {symbol} on {timeframe} timeframe")
    
    # Keep track of the last processed candle time
    last_candle_time = None
    
    # Verify existing positions at startup
    positions = mt5.positions_get(symbol=symbol)
    if positions:
        print(f"Found {len(positions)} existing positions for {symbol}")
        current_position = sum(pos.volume for pos in positions)
        print(f"Setting current position to {current_position} contracts")
    
    while running:
        try:
            # Check if buffer is initialized and has enough data
            if len(indicator_history_buffer["timestamps"]) < 5:
                print("⚠️  Aguardando buffer de indicadores ser preenchido...")
                time.sleep(10)
                continue
            
            # Check if a new candle has formed
            current_rates = mt5.copy_rates_from_pos(symbol, timeframe_dict.get(timeframe, mt5.TIMEFRAME_H4), 1, 1)
            if current_rates is not None and len(current_rates) > 0:
                current_candle_time = pd.to_datetime(current_rates[0]['time'], unit='s')
                
                # If this is a new candle, analyze the market
                if last_candle_time is None or current_candle_time > last_candle_time:
                    print(f"New candle detected at {current_candle_time}")
                    last_candle_time = current_candle_time
                    
                    # Get current candle data (now with dynamic history)
                    candle_data = get_current_candle_data(symbol, timeframe)
                    
                    # Skip analysis if data is insufficient
                    if "insuficiente" in candle_data or "Erro" in candle_data:
                        print(f"⚠️  {candle_data}")
                        time.sleep(5)
                        continue
                    
                    # Perform market analysis
                    analysis = analyze_market(symbol, timeframe)
                    print(f"Market analysis: {analysis['market_summary']}")
                    print(f"Confidence: {analysis['confidence_level']}, Direction: {analysis['direction']}")
                    print(f"DEBUG: Verificando condição de entrada automática - Confiança: {analysis['confidence_level']}, Threshold: {CONFIDENCE_THRESHOLD}")

                    # Log candle data and analysis to Excel
                    try:
                        log_candle_data(symbol, timeframe, candle_data, analysis)
                    except Exception as e:
                        print(f"Error logging candle data: {e}")
                    # If confidence exceeds threshold, execute automatically
                    if abs(analysis['confidence_level']) >= CONFIDENCE_THRESHOLD:
                        print(f"Confidence level {analysis['confidence_level']} exceeds threshold ({CONFIDENCE_THRESHOLD}). Executing trade...")
                  
                        contracts_to_adjust = min(
                            analysis['contracts_to_adjust'],
                            max_contracts - current_position if analysis['action'] == "ADD_CONTRACTS" else current_position
                        )
                        
                        if contracts_to_adjust > 0:
                            result = execute_trade(symbol, analysis['action'], contracts_to_adjust)
                            print(result)
                    else:
                        print(f"Confidence level {analysis['confidence_level']} below threshold. Waiting for user input.")
            
            # Check if there are user inputs in the queue
            try:
                user_input = trade_queue.get_nowait()
                print(f"Processing user input: {user_input}")
                
                # Get current candle data
                candle_data = get_current_candle_data(symbol, timeframe)
                
                # Analyze market
                analysis = analyze_market(symbol, timeframe)
                
                # Log candle data and analysis to Excel (for manual analysis as well)
                try:
                    log_candle_data(symbol, timeframe, candle_data, analysis, 
                                file_path=f"metatradebot_manual_{symbol}_{timeframe}.xlsx")
                except Exception as e:
                    print(f"Error logging manual candle data: {e}")
                
                # Execute trade based on analysis
                if analysis['action'] != "WAIT":
                    # Limit contract adjustments to respect max_contracts
                    contracts_to_adjust = min(
                        analysis['contracts_to_adjust'],
                        max_contracts - current_position if analysis['action'] == "ADD_CONTRACTS" else current_position
                    )
                    
                    if contracts_to_adjust > 0:
                        result = execute_trade(symbol, analysis['action'], contracts_to_adjust)
                        print(result)
                        
                        # Calculate position performance
                        symbol_info = mt5.symbol_info(symbol)
                        if symbol_info:
                            current_price = symbol_info.ask
                            performance = calculate_position_performance(trade_history, current_price)
                            total_pnl = performance['total_pnl']
                            
                            print(f"Position size: {performance['position_size']}")
                            print(f"Average entry: {performance['average_entry']:.5f}")
                            print(f"Unrealized P&L: ${performance['unrealized_pnl']:.2f}")
                            print(f"Total P&L: ${performance['total_pnl']:.2f}")
                    else:
                        print(f"Skipping trade - contracts to adjust ({contracts_to_adjust}) <= 0")
                else:
                    print("Analysis recommends waiting - no trade executed")
            
            except queue.Empty:
                # No user input, continue with regular updates
                pass
            
            # Update positions every 30 seconds
            if int(time.time()) % 10 == 0:
                # Check positions in MetaTrader5
                positions = mt5.positions_get(symbol=symbol)
                if positions:
                    mt_position_size = sum(pos.volume for pos in positions)
                    if mt_position_size != current_position:
                        print(f"Position discrepancy detected: MT5={mt_position_size}, Bot={current_position}")
                        print(f"Updating internal position to match MT5: {mt_position_size}")
                        current_position = mt_position_size
                        
                    # Update P&L with real data
                    mt_pnl = sum(pos.profit for pos in positions)
                    if abs(mt_pnl - total_pnl) > 0.01:  # If there's a significant difference
                        print(f"P&L discrepancy detected: MT5=${mt_pnl:.2f}, Bot=${total_pnl:.2f}")
                        print(f"Updating internal P&L to match MT5: ${mt_pnl:.2f}")
                        total_pnl = mt_pnl
                else:
                    # If there are no positions, and the bot thinks it still has, adjust
                    if current_position != 0:
                        print(f"No positions found in MT5, but bot thinks it has {current_position}. Resetting to 0.")
                        current_position = 0
                        total_pnl = 0
            
            # Sleep to avoid excessive polling
            time.sleep(1)
            
        except Exception as e:
            print(f"Error in trading loop: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(5) 
            

    # ─────────────────────────  LAYOUT CABEÇALHO EM DUAS COLUNAS  ─────────────────────────
app.layout = dbc.Container(fluid=True, children=[
        dbc.Row([
            dbc.Col([
                dbc.Navbar(
                    dbc.Container([
                        # Container flexível com classes de alinhamento
                        html.Div([

                            # Logotipo encostado à esquerda
                            html.Img(src="/assets/Ninjalogo.png", height="80px", 
                                    style={"marginLeft": "0px"}),  # Margem negativa para encostar à esquerda
                            
                            # Texto centralizado (usando flex-grow e justify-content-center)
                            html.Div([
                                html.H2("NinjaTrader LLM Analysis", 
                                        className="text-center fw-bold text-white m-0",
                                        style={"width": "100%"})
                            ], className="d-flex justify-content-center flex-grow-1"),
                            
                            # Elemento vazio para balancear (mesmo espaço do logo)
                            html.Div(style={"width": "80px"})
                        ], className="d-flex align-items-center w-100"),
                    ], fluid=True, className="px-2"),  # Container fluido com padding horizontal reduzido
                    color="black",  # Cor de fundo preta
                    dark=True,
                    className="mb-4 border-0",  # Remover borda
                    style={"backgroundColor": "#000000"},  # Garantir que seja preto puro
                ),
            ], className="p-0"),  # Remover padding da coluna
        ], className="mb-4 g-0"),  # Remover gutters da linha

    # Somewhere in your main layout
    html.Button(
        id="close-historical-chart", 
        style={"display": "none"}  # Hidden initially
    ),
    
    # Reorganized layout to put Trading Setup on left and Trading Assistant on right
    dbc.Row([
        # Trading Setup on the left side
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Trading Setup", 
               style={"padding": "0.5rem 1rem", "fontSize": "0.9rem"}),

                    
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Label("Asset Symbol:"),
                            dbc.Input(id="symbol-input", type="text", value="NGEU25", placeholder="Enter symbol (e.g., EURUSD)"),
                        ], width=6),
                        dbc.Col([
                            html.Label("Timeframe:"),
                            dcc.Dropdown(
                                id="timeframe-dropdown",
                                options=[
                                    {"label": "1 Minute (M1)", "value": "M1"},
                                    {"label": "5 Minutes (M5)", "value": "M5"},
                                    {"label": "15 Minutes (M15)", "value": "M15"},
                                    {"label": "30 Minutes (M30)", "value": "M30"},
                                    {"label": "1 Hour (H1)", "value": "H1"},
                                    {"label": "4 Hours (H4)", "value": "H4"},
                                    {"label": "1 Day (D1)", "value": "D1"},
                                ],
                                value="H4",
                                clearable=False,
                                style={                         
                                    "color": "black",
                                    "backgroundColor": "white"
                                },
                                className="timeframe-dd" 
                            ),
                        ], width=6),
                    ]),
                    
                    html.Br(),
                    
                    dbc.Row([
                        dbc.Col([
                            html.Label("Support & Resistance Levels:"),
                            dbc.Input(id="sr-input", type="text", placeholder="Enter levels separated by commas (e.g., 1.1000, 1.1050)"),
                            html.Div(id="sr-output", className="mt-2")
                        ], width=6),
                        dbc.Col([
                            html.Label("Max Contracts:"),
                            dbc.Input(id="max-contracts-input", type="number", value=5, min=1, max=100),
                        ], width=6),
                    ]),
                    
                    html.Br(),
                    
                    dbc.Row([
                        dbc.Col([
                            dbc.Checkbox(id="use-context-checkbox", label="Use Initial Market Context", value=False),
                            dbc.Tooltip(
                                "Enabling this will analyze previous candles to provide context for the LLM, but will use more tokens.",
                                target="use-context-checkbox",
                            ),
                        ], width=6),
                        dbc.Col([
                            dbc.Button("Start", id="start-button", color="success", className="me-2"),
                            dbc.Button("Stop", id="stop-button", color="danger", className="me-2"),
                            dbc.Button("Trigger Analysis", id="analyze-button", color="primary", className="me-2"),
                        ], width=6, className="d-flex justify-content-end align-items-end"),
                    ]),
                ]),
            ], className="mb-4"),
            create_debug_area(),
            # Botão Web Scraping com estilo personalizado
        dbc.Button(
            [html.I(className="fas fa-globe me-2"), "🌐 Real time ", html.Span("Web scraping", style={"fontStyle": "italic"})],
            id="open-agent-demo",
            color="dark",  # Fundo preto
            className="mb-4 w-100",  # Margem inferior e largura total
            style={
                "borderColor": "#FF8C00",  # Borda laranja
                "color": "#FF8C00",  # Texto laranja
                "fontSize": "1.1rem",  # Tamanho do texto um pouco maior
                "padding": "10px",  # Padding maior
                "fontWeight": "bold",  # Texto em negrito
            }
        ),
        ], width=6),
        
        # Trading Assistant on the right side
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.H4("Trading Assistant", className="d-inline me-2"),
                    html.Span("Ask about the asset, technical analysis, or specific periods", 
                             className="text-muted small")
                ]),
                dbc.CardBody([
                    # Chat display area
                    html.Div(
                        id="chat-messages",
                        className="chat-container mb-3",
                        style={
                            "height": "250px",
                            "overflowY": "auto",
                            "display": "flex",
                            "flexDirection": "column",
                            "padding": "10px",
                            "border": "1px solid rgba(255, 255, 255, 0.1)",
                            "borderRadius": "8px",
                            "backgroundColor": "rgba(0, 0, 0, 0.2)"
                        },
                        children=[
                            html.Div([
                                html.Div(dcc.Markdown("Hello! I'm your trading assistant with voice capabilities.")),
                                html.Div(datetime.now().strftime("%H:%M"), className="timestamp")
                            ])
                        ]
                    ),
                    
                    # Area for visual results
                    html.Div(
                        id="chat-visual-results",
                        className="mb-3",
                        style={"display": "none"}
                    ),
                    
                    # User input area
                    dbc.InputGroup([
                        dbc.Input(
                            id="chat-input",
                            placeholder="Type your question about the asset...",
                            type="text",
                            className="border-primary",
                            n_submit=0
                        ),
                        dbc.Button(
                            html.Div([
                                "Send ",
                                html.I(className="fas fa-arrow-right", style={"marginLeft": "5px"})
                            ], style={"display": "flex", "alignItems": "center"}) ,
                            id="send-button",
                            color="primary",
                            className="px-3",
                            style={"fontWeight": "bold", "borderTopLeftRadius": "0", "borderBottomLeftRadius": "0"},
                            n_clicks=0,
                        )
                    ]),
                    
                    # Voice selector
                    create_voice_selector_with_tooltip(),    
                    # Suggestion buttons
                    html.Div([
                        html.P("Suggestions:", className="mt-2 mb-1 text-muted small"),
                        html.Div([
                            dbc.Button("Could you provide me with an analysis of how the market was 5 hours ago?", 
                                     id="suggestion-1", 
                                     color="light", 
                                     size="sm", 
                                     className="me-2 mb-2 suggestion-btn"),
                            dbc.Button("Show the chart from last week", 
                                     id="suggestion-2", 
                                     color="light", 
                                     size="sm", 
                                     className="me-2 mb-2 suggestion-btn"),
                            dbc.Button("What is the current trend?", 
                                     id="suggestion-3", 
                                     color="light", 
                                     size="sm", 
                                     className="me-2 mb-2 suggestion-btn"),
                            dbc.Button("How is the confluence of indicators now?", 
                                     id="suggestion-voice", 
                                     color="light", 
                                     size="sm", 
                                     className="me-2 mb-2 suggestion-btn"),
                            dbc.Button("What does this support level mean?", 
                                     id="suggestion-4", 
                                     color="light", 
                                     size="sm", 
                                     className="me-2 mb-2 suggestion-btn"),
                        ], className="d-flex flex-wrap")
                    ])
                ])
            ], className="mb-4 chat-card shadow")
        ], width=6)
    ]),

    # Add this Div to store chat history
    dcc.Store(id="chat-history", data={"messages": []}),

    # Fix duplicated chat-history Store
    # (removed duplicate dcc.Store with id="chat-history")
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Market Dashboard"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.H5("Current Position:"),
                            html.H3(id="position-display", children="0 Contracts"),
                        ], width=4),
                        dbc.Col([
                            html.H5("Total P&L:"),
                            html.H3(id="pnl-display", children="$0.00"),
                        ], width=4),
                        dbc.Col([
                            html.H5("Market Direction:"),
                            html.H3(id="direction-display", children="Neutral"),
                        ], width=4),
                    ]),
                    
                    html.Hr(),
                                    dbc.Row([
                    # Primeiro gráfico - ocupa metade da largura
                    dbc.Col([
                        html.H5("Price Chart:"),
                        dcc.Graph(id="price-chart", style={"height": "800px"}),
                    ], md=6),
                    
                    # Segundo gráfico (Dynamic Chart Component) - ocupa a outra metade
                    dbc.Col([
                        html.H5("Dynamic Chart:"),
                        html.Div(
                            dynamic_chart_component,
                            style={"height": "800px"}
                        ),
                    ], md=6),
                ]),
   
                    
                    # Nova seção para exibir o raciocínio do LLM e nível de confiança
                    dbc.Card([
                        dbc.CardHeader("LLM Analysis for Current Candle"),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    html.H5("Confidence Level:", className="mb-2"),
                                    # Barra de confiança com gradiente de cores
                                    html.Div([
                                        dbc.Progress(
                                            id="confidence-bar",
                                            value=50,
                                            color="info",
                                            className="mb-1",
                                            style={"height": "30px"}
                                        ),
                                        
                                        html.Div(
                                            id="confidence-scale",
                                            className="d-flex justify-content-between mt-1",
                                            children=[
                                                html.Span("Bearish Confidence", className="text-danger"),
                                                html.Span("Neutral", className="text-info"),
                                                html.Span("Bullish Confidence", className="text-success")
                                            ]
                                        )
                                    ], className="mb-3"),
                                    html.Div(
                                        id="confidence-value",
                                        className="text-center mb-3 h7"
                                    ),
                                ], width=12),
                                html.Div([
                                html.H5("Confidence Factors Analysis", className="mt-3 mb-2"),
                                html.Div(id="confidence-details", className="confidence-details")
                            ], className="mt-4")
                            ]),
                            dbc.Row([
                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardHeader("Indicator Confluence Analysis"),
                                        dbc.CardBody([
                                            html.Div(id="confluence-visualization")
                                        ])
                                    ], className="mb-4 dashboard-card")
                                ], width=12)
                            ], className="mt-3"),
                            dbc.Row([
                                dbc.Col([
                                    html.H5("LLM Reasoning:"),
                                    dbc.Card(
                                        dbc.CardBody(
                                            html.Div(
                                                id="reasoning-display",
                                                className="border-0 text-white"
                                            )
                                        ),
                                        className="bg-dark text-light mt-2 mb-3" 
                                    ),
                                ], width=12),
                            ]),
                            dbc.Row([
                                dbc.Col([
                                    html.H5("Key Factors Influencing Decision:"),
                                    html.Div(
                                        id="key-factors",
                                        className="mt-2"
                                    ),
                                ], width=12),
                            ]),
                        ]),
                    ], style={"marginTop": "200px"},className="mb-4"),
                    
                ]),
            ], className="mb-4"),
        ], width=12),
    ]),
    dbc.Row([
    dbc.Col([
        html.H5("Past Market Context (H4):"),
        html.Div(
            id="past-market-context",
            className="mt-2"
        ),
    ], width=12),
    ]),
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Trade History"),
                dbc.CardBody([
                    html.Div(id="trade-history-display", style={"maxHeight": "300px", "overflow": "auto"})
                ]),
            ]),
        ], width=12),
    ]),
    
    # Add interval component for regular updates
    dcc.Interval(
        id='interval-component',
        interval=2*1000,  # in milliseconds (2 seconds)
        n_intervals=0
    ),
    
    # Store de dados para o último análise do LLM
    dcc.Store(id="last-llm-analysis", data={}),
        # Add the missing bot-state component
    dcc.Store(id="bot-state", data={"running": False}),
    
    # Add dynamic-analysis-interval if it's also missing
    dcc.Interval(
        id='dynamic-analysis-interval',
        interval=10*1000,  # 10 seconds
        n_intervals=0,
        disabled=True
    ),


    # Modal para exibir o Agent Demo
    dbc.Modal(
        [
            dbc.ModalHeader(
                dbc.ModalTitle([html.I(className="fas fa-brain me-2"), "Intelligence Agent - Natural Gas Markets"]),
                close_button=True,
            ),
            dbc.ModalBody(id="agent-demo-content", style={"height": "85vh", "padding": "0"}),
        ],
        id="agent-demo-modal",
        size="xl",
        is_open=False,
        centered=True,
        style={"maxWidth": "95vw", "width": "1400px"},
    )
])
    # Registrar callbacks do dynamic chart analyzer
register_dynamic_chart_callbacks(app, dynamic_analyzer, symbol_id='symbol-input')

@app.callback(
    [Output("sr-output", "children")],
    [Input("sr-input", "value")]
)
def update_sr_levels(sr_input):
    """Update support and resistance levels"""
    global support_resistance_levels
    
    if not sr_input:
        support_resistance_levels = []
        return [html.P("No levels defined", className="text-muted")]
    
    try:
        # Parse comma-separated values
        levels = [float(level.strip()) for level in sr_input.split(",") if level.strip()]
        support_resistance_levels = sorted(levels)
        
        # Create output display
        level_displays = []
        for i, level in enumerate(support_resistance_levels):
            level_displays.append(html.Span(f"{level:.4f}", className="badge bg-primary me-2"))
        
        return [html.Div(level_displays)]
    
    except ValueError:
        return [html.P("Invalid input. Use comma-separated numbers.", className="text-danger")]


@app.callback(
    Output("price-chart", "figure"),
    [Input("interval-component", "n_intervals")],
    [State("symbol-input", "value"),
     State("timeframe-dropdown", "value")]
)

def update_price_chart(n_intervals, symbol, timeframe):

    """Update price chart with current market data """
    if not symbol or not timeframe:
        # Return empty chart if no symbol or timeframe
        return go.Figure()
    
    # Define the timeframe mapping
    timeframe_dict = {
        "M1": mt5.TIMEFRAME_M1,
        "M5": mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "M30": mt5.TIMEFRAME_M30,
        "H1": mt5.TIMEFRAME_H1,
        "H4": mt5.TIMEFRAME_H4,
        "D1": mt5.TIMEFRAME_D1,
    }
    
    try:
        # Check if mt5 is initialized
        if not mt5.terminal_info():
            # Return placeholder chart if MT5 not initialized
            fig = go.Figure()
            fig.add_annotation(
                text="Start trading to view chart.",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=14)
            )
            return fig
        
        
        rates = mt5.copy_rates_from_pos(symbol, timeframe_dict.get(timeframe, mt5.TIMEFRAME_H4), 0, 100)
        
        if rates is None or len(rates) == 0:
            # Return empty chart if no data
            fig = go.Figure()
            fig.add_annotation(
                text=f"No data available for {symbol} on {timeframe} timeframe",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=14)
            )
            return fig
        
        # Convert to dataframe
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        
        # Calculate indicators
        df['atr'] = calculate_atr(df, period=14)
        df['entropy'] = calculate_directional_entropy(df, period=14)
        df['ema9'] = calculate_ema(df['close'], period=9)
        df['slope'] = calculate_slope(df['close'], period=14)
        df['combined_confidence'] = combined_confidence = 0  # Placeholder, will be set later
        # Create subplots - AGORA COM 5 ROWS PARA SEPARAR OS INDICADORES
        fig = make_subplots(
            rows=5, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.4, 0.15, 0.15, 0.15, 0.15],  # Price chart maior, indicadores menores
            subplot_titles=(
                f"{symbol} - {timeframe} Price", 
                "ATR (Average True Range)", 
                "Directional Entropy",
                "Slope",
                "Volume"
            )
        )
        
        # Create custom hover text with indicators included
        global confidence_level
        global confidence_history
        hovertext = []
        for i in range(len(df)):
            date_str = df['time'].iloc[i].strftime('%b %d, %Y, %H:%M')
            candle_time = df['time'].iloc[i]
            current_confidence = confidence_history.get(candle_time, 0)  # 0 se não houver análise
            hover_info = (
                f"Date: {date_str}<br>" +
                f"Open: {df['open'].iloc[i]:.4f}<br>" +
                f"High: {df['high'].iloc[i]:.4f}<br>" +
                f"Low: {df['low'].iloc[i]:.4f}<br>" +
                f"Close: {df['close'].iloc[i]:.4f}<br>" +
                f"ATR: {df['atr'].iloc[i]:.4f}<br>" + 
                f"Entropy: {df['entropy'].iloc[i]:.4f}<br>" +
                f"Slope: {df['slope'].iloc[i]:.6f}<br>" +
                #f"EMA9: {df['ema9'].iloc[i]:.4f}" 
                f"Confidence: {current_confidence}"
            )
            hovertext.append(hover_info)
        
        # 1. PRICE CHART (Row 1) - Mantendo todas as funcionalidades existentes
        fig.add_trace(
            go.Candlestick(
                x=df['time'],
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name="Price",
                hoverinfo="text",
                hovertext=hovertext,
            ),
            row=1, col=1
        )
        
        # Add EMA to price chart
        fig.add_trace(
            go.Scatter(
                x=df['time'],
                y=df['ema9'],
                name="EMA9",
                line=dict(color='purple', width=2),
                hovertemplate='EMA9: %{y:.4f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Add support and resistance lines to price chart
        for level in support_resistance_levels:
            fig.add_shape(
                type="line",
                x0=df['time'].iloc[0],
                x1=df['time'].iloc[-1],
                y0=level,
                y1=level,
                line=dict(color="rgba(255, 0, 0, 0.5)", width=2, dash="dash"),
                row=1, col=1
            )
            # Add annotation for S/R level
            fig.add_annotation(
                x=df['time'].iloc[-1],
                y=level,
                text=f"{level:.4f}",
                showarrow=False,
                xanchor="left",
                bgcolor="rgba(255, 0, 0, 0.3)",
                bordercolor="red",
                row=1, col=1
            )
        
        # 2. ATR CHART (Row 2) - APENAS A LINHA DO ATR
        # Verificar se ATR tem valores válidos
        atr_valid = df['atr'].dropna()
        if len(atr_valid) > 0:
            fig.add_trace(
                go.Scatter(
                    x=df['time'],
                    y=df['atr'],
                    name="ATR (14)",
                    line=dict(color='orange', width=2),
                    mode='lines',
                    hovertemplate='ATR: %{y:.4f}<extra></extra>',
                    showlegend=True
                ),
                row=2, col=1
            )
        else:
            # Se não houver dados ATR válidos, adicionar anotação
            fig.add_annotation(
                text="ATR data not available",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=12, color="orange"),
                row=2, col=1
            )
        
        # 3. ENTROPY CHART (Row 3) - APENAS LINHA
        fig.add_trace(
            go.Scatter(
                x=df['time'],
                y=df['entropy'],
                name="Entropy",
                line=dict(color='blue', width=2),
                hovertemplate='Entropy: %{y:.4f}<extra></extra>'
            ),
            row=3, col=1
        )
        
        # Add critical entropy levels
        fig.add_hline(y=0.6, line_dash="dash", line_color="white", line_width=0.5,   
                     annotation_text="0.6", annotation_position="right", row=3, col=1)
        fig.add_hline(y=0.8, line_dash="dash", line_color="white",line_width=0.5,   
                     annotation_text="0.8", annotation_position="right", row=3, col=1)
        
        # 4. SLOPE CHART (Row 4) - APENAS LINHA
        fig.add_trace(
            go.Scatter(
                x=df['time'],
                y=df['slope'],
                name="Slope",
                line=dict(color='green', width=2),
                hovertemplate='Slope: %{y:.6f}<extra></extra>'
            ),
            row=4, col=1
        )
        
        # Add zero line for slope
        fig.add_hline(y=0, line_dash="solid", line_color="gray", 
                     line_width=1, row=4, col=1)
        
        # 5. VOLUME CHART (Row 5)
        colors = ['green' if df['close'].iloc[i] > df['open'].iloc[i] else 'red' 
                  for i in range(len(df))]
        
        fig.add_trace(
            go.Bar(
                x=df['time'],
                y=df['tick_volume'],
                name="Volume",
                marker=dict(color=colors, opacity=0.7),
                hovertemplate='Volume: %{y}<extra></extra>'
            ),
            row=5, col=1
        )
        
        # Add trade entry points (mantendo funcionalidade existente)
        for trade in trade_history:
            timestamp = datetime.strptime(trade['timestamp'], "%Y-%m-%d %H:%M:%S")
            
            # Find the closest candle to this timestamp
            closest_idx = np.abs(df['time'] - timestamp).argmin()
            
            if closest_idx >= 0 and closest_idx < len(df):
                # Add marker for trade entry on price chart
                marker_color = 'green' if trade['action'] == 'ADD_CONTRACTS' else 'red'
                
                fig.add_trace(
                    go.Scatter(
                        x=[df['time'].iloc[closest_idx]],
                        y=[df['high'].iloc[closest_idx] * 1.002 if trade['action'] == 'ADD_CONTRACTS' else df['low'].iloc[closest_idx] * 0.998],
                        mode='markers+text',
                        marker=dict(
                            symbol='triangle-down' if trade['action'] == 'REMOVE_CONTRACTS' else 'triangle-up',
                            size=15,
                            color=marker_color,
                            line=dict(width=2, color='black')
                        ),
                        text=f"{trade['contracts']}",
                        textposition="top center" if trade['action'] == 'ADD_CONTRACTS' else "bottom center",
                        name=f"{trade['action']} ({trade['contracts']})",
                        showlegend=False,
                        hovertemplate=f"{trade['action']}<br>Contracts: {trade['contracts']}<br>Price: {trade['price']:.4f}<extra></extra>"
                    ),
                    row=1, col=1
                )
        
        # Update layout with dark theme and proper styling
        fig.update_layout(
            height=1000,  # Aumentado para acomodar todos os subplots ______________________________________________________________________
            margin=dict(l=50, r=50, t=80, b=50),
            template="plotly_dark",
            showlegend=True,
            legend=dict(
                orientation="h", 
                yanchor="bottom", 
                y=1.06, 
                xanchor="center", 
                x=0.5,
                bgcolor="rgba(0,0,0,0.5)",
                bordercolor="gray",
                borderwidth=1
            ),
            hovermode="x unified",  # IMPORTANTE: Isso cria a linha vertical em todos os subplots
            plot_bgcolor='rgba(0,0,0,1)',  # Fundo preto sólido
            paper_bgcolor='rgba(0,0,0,1)',  # Fundo preto sólido
            xaxis_rangeslider_visible=False
        )
         # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        
        # Skip Saturdays and Sundays on every subplot
        for r in range(1, 6):      # rows 1-5
            fig.update_xaxes(
                rangebreaks=[dict(bounds=["sat", "mon"])],
                row=r, col=1
            )
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        
        # Configure x-axes (only show on bottom plot)
        for i in range(1, 5):
            fig.update_xaxes(showticklabels=False, row=i, col=1)
        fig.update_xaxes(title_text="Time", row=5, col=1)
        
        # Configure y-axes with appropriate titles
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="ATR", row=2, col=1)
        fig.update_yaxes(title_text="Entropy", row=3, col=1)
        fig.update_yaxes(title_text="Slope", row=4, col=1)
        fig.update_yaxes(title_text="Volume", row=5, col=1)
        
        # Add grid to all subplots
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
       

        
        return fig
    
    except Exception as e:
        print(f"Error updating chart: {e}")
        
        # Return error message in chart
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error updating chart: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=14, color="red")
        )
        return fig
    
@app.callback(
    [Output("max-contracts-input", "disabled")],
    [Input("start-button", "n_clicks")],
    [State("max-contracts-input", "value")]
)
def update_max_contracts(n_clicks, max_contracts_value):
    """Update max contracts setting"""
    global max_contracts
    
    if n_clicks is not None and n_clicks > 0:
        max_contracts = max(1, min(100, max_contracts_value))
        return [True]  # Disable input after starting
    
    return [False]

@app.callback(
    [Output("start-button", "disabled"),
     Output("stop-button", "disabled")],
    [Input("start-button", "n_clicks"),
     Input("stop-button", "n_clicks")],
    [State("symbol-input", "value"),
     State("timeframe-dropdown", "value"),
     State("max-contracts-input", "value"),
     State("use-context-checkbox", "value")]
)
def handle_trading_controls(start_clicks, stop_clicks, symbol, timeframe, max_contracts_input, use_context):
    """Handle start and stop trading buttons with dynamic history initialization"""
    global running, max_contracts, use_initial_context_enabled
    
    ctx = callback_context
    if not ctx.triggered:
        return False, True
    
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if triggered_id == "start-button" and start_clicks and symbol and timeframe:
        max_contracts = max_contracts_input or 20
        use_initial_context_enabled = use_context
        
        if initialize_mt5():
            # 1. Inicializar contexto H4 (uma vez)
            initialize_h4_context(symbol)
            
            # 2. Inicializar buffer dinâmico de indicadores (uma vez)
            if not initialize_indicator_history(symbol, timeframe, initial_candles=15):
                print("❌ Falha ao inicializar histórico de indicadores")
                return False, True
            
            # 3. Inicializar LLM chain
            initialize_llm_chain()
            
            # 4. Iniciar trading loop
            running = True
            trading_thread = threading.Thread(
                target=trading_loop,
                args=(symbol, timeframe),
                daemon=True
            )
            trading_thread.start()
            
            print(f"✅ System started with dynamic history for {symbol} {timeframe}")
            debug_print(f"✅ System started with dynamic history for {symbol} {timeframe}")
            return True, False  # Start disabled, Stop enabled
        else:
            return False, True
    
    elif triggered_id == "stop-button" and stop_clicks:
        running = False
        print("🛑 Trading stopped by user")
        return False, True  # Start enabled, Stop disabled
    
    return False, True

def control_trading(start_clicks, stop_clicks, symbol, timeframe, use_context):
    """Control trading process"""
    global running, llm_chain, trade_queue, initial_market_context, use_initial_context_enabled
    
    # Get the button that triggered the callback
    ctx = callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None
    
    if triggered_id == "start-button" and start_clicks is not None and start_clicks > 0:
        # Initialize components
        if not initialize_mt5():
            return [False, True, False]
        
        # Initialize LLM chain
        if llm_chain is None:
            initialize_llm_chain()
        
        # Set the global flag for using initial context
        use_initial_context_enabled = bool(use_context)
        
        # If enabled, get and analyze initial context ONCE at startup
        if use_initial_context_enabled:
            try:
                print("Getting initial market context...")
                # Get historical data for H4 timeframe
                historical_data = get_initial_context(symbol, "H4", num_candles=10)#__________________________________________________________ quantia de candles para a analize H4 do langchain
                print("\n====== EXACT DATA BEING SENT TO LLM FOR H4 ANALYSIS ======")
                print(historical_data)
                print("\n====== END OF EXACT DATA ======\n")
               
        
                # Also print the full prompt being sent
                full_prompt = initial_context_prompt.format(historical_data=historical_data)
                print("\n====== FULL PROMPT BEING SENT TO LLM ======")
                print(full_prompt)
                print("\n====== END OF FULL PROMPT ======\n")
        # Use your existing initial_context_prompt from prompts.py
                
                
                # Create LLM for historical analysis
                history_llm = ChatOpenAI(
                    temperature=0.1,
                    api_key=os.getenv("OPENAI_API_KEY"),
                    model_name="gpt-4.1-mini"
                )
                
                # Create a chain for historical analysis using your existing prompt
                historical_chain = (
                    {"historical_data": lambda x: historical_data}
                    | initial_context_prompt 
                    | history_llm
                )
                
                # Run the historical analysis
                response = historical_chain.invoke({})
                
                # Store the result in the global variable
                if isinstance(response.content, str):
                    initial_market_context = response.content
                else:
                    # Format the JSON response for display in the prompt
                    initial_market_context = json.dumps(response.content, indent=2)
                
                debug_print("Initial market context analysis completed and stored.")
                
            except Exception as e:
                print(f"Error getting initial context: {e}")
                import traceback
                traceback.print_exc()
                initial_market_context = None
        else:
            # Reset the initial context if not using it
            initial_market_context = None
        
        # Start trading loop
        running = True
        trading_thread = threading.Thread(target=trading_loop, args=(symbol, timeframe))
        trading_thread.daemon = True
        trading_thread.start()
        
        return [True, False, False]
    
    elif triggered_id == "stop-button" and stop_clicks is not None and stop_clicks > 0:
        # Stop trading loop
        running = False
        
        # Shutdown MT5
        mt5.shutdown()
        
        return [False, True, True]
    
    # Default state
    return [False, True, True]
@app.callback(
    Output("analyze-button", "n_clicks"),
    [Input("analyze-button", "n_clicks")],
    [State("symbol-input", "value"),
     State("timeframe-dropdown", "value")]
)
def trigger_analysis(n_clicks, symbol, timeframe):
    """Trigger manual market analysis"""
    if n_clicks is not None and n_clicks > 0:
        # Add to the queue
        trade_queue.put("ANALYZE")
    
    return 0  # Reset clicks
# Inside analyze_market function after getting the response
def start_token_reporting():
    """Start token usage reporting thread"""
    token_thread = threading.Thread(target=print_token_usage)
    token_thread.daemon = True
    token_thread.start()
    print("Token usage reporting started")

@app.callback(
    Output("agent-demo-modal", "is_open"),
    [Input("open-agent-demo", "n_clicks")],
    [State("agent-demo-modal", "is_open")],
)
def toggle_agent_demo_modal(n_clicks, is_open):
    if n_clicks:
        return not is_open
    return is_open

@app.callback(
    Output("agent-demo-content", "children"),
    [Input("agent-demo-modal", "is_open")],
)
def load_agent_demo_content(is_open):
    if is_open:
        # Criar um iframe que carrega a rota /agent-demo/ do servidor
        return html.Iframe(
            src="/agent-demo/",
            style={
                "width": "100%",
                "height": "100%",
                "border": "none",
                "borderRadius": "5px",
                "overflow": "hidden"
            }
        )
    return None

@app.callback(
    Output("voice-selector", "value"),
    [Input("voice-selector", "value")]
)
def update_selected_voice(voice_name):
    """Update the selected voice for text-to-speech"""
    if voice_name:
        from voice_utils import set_voice, get_selected_voice
        set_voice(voice_name)
        #print(f"Voice changed to: {voice_name}")
        # Double-check that the voice was actually set
        current_voice = get_selected_voice()
        if current_voice != voice_name:
            print(f"Warning: Voice was not changed to {voice_name}, still using {current_voice}")
            return current_voice
    return voice_name

register_debug_callbacks(app)

# No final do arquivo, onde você inicia o app
if __name__ == "__main__":
    # Initialize MT5 connection
    if initialize_mt5():
        print("✅ MetaTrader5 connected successfully")
        #debug_print("✅ System started, generating analysis")
    else:
        print("❌ Failed to connect to MetaTrader5")
    #print("Inicializando Agent de Inteligência Artificial - Foco: Gás Natural...")
    
    # Criar a aplicação do agente no mesmo servidor
    agent_app, agent = create_agent_demo_app(server=app.server)
    # Start token monitoring
    start_token_monitoring()
    
    # Start the app
    app.run(debug=True, port=8050)

# Callback to display initial welcome message
@app.callback(
    Output("chat-messages", "children"),
    [Input("initial-message", "data")]
)
def display_welcome_message(message_data):
    """Display welcome message when app loads"""
    if message_data and message_data.get("show", False):
        timestamp = datetime.now().strftime("%H:%M")
        return [html.Div([
            html.Div(message_data.get("content", ""), className="assistant-message"),
            html.Div(timestamp, className="timestamp")
        ])]
    return []