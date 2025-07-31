"""
Debug Area implementation for NinjaTradeV2 with Auto-Clear Timer (Fixed Version)
"""
import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
from datetime import datetime, timedelta
import time

# Initialize a global log buffer to store debug messages with expiration
debug_log_buffer = []
MAX_LOG_ENTRIES = 10000  # Maximum number of log entries to keep in buffer
MESSAGE_LIFETIME_SECONDS = 10000  # Time in seconds before a message is automatically removed

def create_debug_area():
    """
    Creates a debug area component for displaying log messages
    
    Returns:
        dbc.Card: A card component containing the debug area
    """
    return dbc.Card([
        dbc.CardHeader([
            html.H5("Debug Area", className="d-inline me-2"),
            dbc.Badge("Live", color="success", className="me-1"),

            html.Button(
                "Clear", 
                id="clear-debug-btn",
                className="btn btn-sm btn-outline-secondary float-end"
            ),
        ]),
        dbc.CardBody([
            html.Div(
                id="debug-content",
                className="debug-log-area",
                style={
                    "height": "150px",
                    "overflowY": "auto",
                    "fontFamily": "monospace",
                    "fontSize": "0.85rem",
                    "backgroundColor": "#1e1e1e",
                    "color": "#d4d4d4",
                    "padding": "10px",
                    "borderRadius": "4px",
                    "whiteSpace": "pre-wrap",
                    "wordWrap": "break-word"
                }
            )
        ])
    ], className="mt-3 mb-4")

def register_debug_callbacks(app):
    """
    Registers callbacks for the debug area
    
    Args:
        app: Dash application instance
    """
    @app.callback(
        dash.Output("debug-content", "children"),
        [dash.Input("debug-update-trigger", "n_intervals"),
         dash.Input("clear-debug-btn", "n_clicks")]
    )
    def update_debug_content(n_intervals, clear_clicks):
        """Updates the debug content area with new log messages and removes expired ones"""
        global debug_log_buffer  # Declare global at the beginning of the function
        
        ctx = dash.callback_context
        triggered_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None
        
        if triggered_id == "clear-debug-btn" and clear_clicks:
            # Clear the debug log buffer when clear button is clicked
            debug_log_buffer.clear()
            return []
        
        # Get current time to check for expired messages
        current_time = datetime.now()
        
        # Remove expired messages
        debug_log_buffer = [
            entry for entry in debug_log_buffer 
            if current_time < entry.get('expiry_time', current_time + timedelta(days=1))
        ]
        
        # Convert the remaining log buffer entries to HTML components
        log_components = []
        for entry in debug_log_buffer:
            # Calculate time remaining before auto-clear
            seconds_remaining = (entry['expiry_time'] - current_time).total_seconds()
            seconds_remaining = max(0, int(seconds_remaining))
            
            # Create a log entry with timestamp, message, and time remaining
            log_entry = html.Div([
                html.Span(f"[{entry['timestamp']}] ", style={"color": "#858585"}),
                html.Span(f"[{entry['level'].upper()}] ", style={"color": get_level_color(entry['level'])}),
                html.Span(str(entry['message'])),
                html.Span(
                    f" ({seconds_remaining}s)", 
                    style={"color": "#858585", "fontSize": "0.8em", "marginLeft": "5px"},
                    className="auto-clear-timer"
                ) if seconds_remaining > 0 else ""
            ], className=f"debug-entry")
            
            log_components.append(log_entry)
        
        return log_components

    # Add a hidden interval component to trigger debug updates
    interval = dcc.Interval(
        id="debug-update-trigger",
        interval=1000,  # Update every second for countdown timer
        n_intervals=0
    )
    
    # Check if it already exists before adding
    if not any(getattr(comp, 'id', None) == "debug-update-trigger" for comp in app.layout.children):
        app.layout.children.append(interval)

def get_level_color(level):
    """
    Returns a color for the given log level
    """
    level_colors = {
        "info": "#569cd6",     # Blue
        "warning": "#ce9178",  # Orange
        "error": "#f14c4c",    # Red
        "success": "#6a9955",  # Green
        "debug": "#b5cea8"     # Light green
    }
    return level_colors.get(level, "#d4d4d4")  # Default to white

def debug_print(message, level="info", lifetime=MESSAGE_LIFETIME_SECONDS):
    """
    Custom print function that adds messages to the debug log buffer with expiry time
    
    Args:
        message (str): The message to log
        level (str): Log level - "info", "warning", "error", or "success"
        lifetime (int): Number of seconds before the message is auto-cleared (0 = never clear)
    """
    global debug_log_buffer  # Declare global at the beginning of the function
    
    # Format timestamp
    current_time = datetime.now()
    timestamp = current_time.strftime("%H:%M:%S.%f")[:-3]
    
    # Calculate expiry time
    expiry_time = current_time + timedelta(seconds=lifetime) if lifetime > 0 else None
    
    # Add to buffer
    debug_log_buffer.append({
        "timestamp": timestamp,
        "message": str(message),
        "level": level,
        "expiry_time": expiry_time,
        "created_time": current_time
    })
    
    # Keep buffer size limited
    if len(debug_log_buffer) > MAX_LOG_ENTRIES:
        debug_log_buffer.pop(0)
    
    # Also print to console for normal logging
    print(f"[{level.upper()}] {message}")

# Debug area CSS
debug_area_css = """
.debug-log-area {
    scrollbar-width: thin;
    scrollbar-color: #3a3a3a #1e1e1e;
}

.debug-log-area::-webkit-scrollbar {
    width: 8px;
}

.debug-log-area::-webkit-scrollbar-track {
    background: #1e1e1e;
}

.debug-log-area::-webkit-scrollbar-thumb {
    background: #3a3a3a;
    border-radius: 4px;
}

.debug-log-area::-webkit-scrollbar-thumb:hover {
    background: #555;
}

.debug-entry {
    border-bottom: 1px solid #333;
    padding: 4px 0;
}

.auto-clear-timer {
    opacity: 0.7;
}

@keyframes fadeOut {
    from { opacity: 1; }
    to { opacity: 0.3; }
}

.debug-entry.expiring {
    animation: fadeOut 1s ease-in-out forwards;
}
"""
# CSS principal da aplicação
app_css = """
/* Chat styles */
.chat-container {
    scrollbar-width: thin;
    scrollbar-color: #6c757d #343a40;
}

.chat-container::-webkit-scrollbar {
    width: 6px;
}

.chat-container::-webkit-scrollbar-track {
    background: #343a40;
}

.chat-container::-webkit-scrollbar-thumb {
    background-color: #6c757d;
    border-radius: 3px;
}

.user-message, .assistant-message {
    max-width: 85%;
    padding: 8px 12px;
    margin-bottom: 10px;
    border-radius: 12px;
    word-wrap: break-word;
}

.user-message {
    background-color: #0d6efd;
    color: white;
    align-self: flex-end;
    border-bottom-right-radius: 2px;
}

.assistant-message {
    background-color: #2a2e32;
    color: #f8f9fa;
    align-self: flex-start;
    border-bottom-left-radius: 2px;
}

.assistant-message .dash-markdown {
    width: 100%;
}

.timestamp {
    font-size: 0.7rem;
    color: rgba(255, 255, 255, 0.5);
    margin-top: 2px;
    text-align: right;
}

.suggestion-btn {
    transition: all 0.2s;
    border-radius: 15px;
    font-size: 0.8rem;
}

.suggestion-btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
}

.chat-card {
    border-left: 4px solid #0d6efd;
}

.historical-date-marker {
    text-align: center;
    color: #adb5bd;
    font-size: 0.8rem;
    margin: 15px 0;
    position: relative;
}

.historical-date-marker::before, 
.historical-date-marker::after {
    content: "";
    display: inline-block;
    width: 25%;
    height: 1px;
    background-color: rgba(173, 181, 189, 0.5);
    vertical-align: middle;
    margin: 0 10px;
}

.timeframe-selector {
    display: flex;
    justify-content: center;
    gap: 8px;
    margin: 10px 0;
}

.timeframe-btn {
    font-size: 0.8rem;
    padding: 3px 8px;
    border-radius: 12px;
}

.animate-pulse {
    animation: pulse 1.5s infinite;
}

@keyframes pulse {
    0% {
        opacity: 1;
    }
    50% {
        opacity: 0.5;
    }
    100% {
        opacity: 1;
    }
}

/* Melhorias estéticas para o dashboard */
.reasoning-text p {
    line-height: 1.5;
    margin-bottom: 0.8rem;
}

/* Gradiente para a barra de confiança */
.confidence-gradient {
    background: linear-gradient(to right, #dc3545, #ffc107, #0dcaf0, #0d6efd, #198754);
    height: 6px;
    width: 100%;
    margin-top: -6px;
    border-radius: 0 0 0.25rem 0.25rem;
}

/* Estilos para os cards de fatores */
.factor-card {
    transition: transform 0.2s;
}
.factor-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 4px 8px rgba(0,0,0,0.2);
}

/* Estilo para o botão de enviar */
#send-button {
    transition: all 0.2s ease;
}
#send-button:hover {
    transform: translateX(2px);
    box-shadow: 0 0 10px rgba(13, 110, 253, 0.5);
}

/* Melhorias para o histórico de trades */
.trade-card {
    transition: all 0.2s;
}
.trade-card:hover {
    transform: scale(1.02);
}

/* Destaque para o header */
.app-header {
    background: linear-gradient(to right, #343a40, #495057);
    border-radius: 0.5rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}

/* Melhorias para os cards */
.dashboard-card {
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    transition: all 0.3s;
}
.dashboard-card:hover {
    box-shadow: 0 6px 10px rgba(0,0,0,0.15);
}

/* Estilo para valores de confiança */
.confidence-value {
    padding: 0.25rem 0.5rem;
    border-radius: 0.25rem;
    display: inline-block;
    font-weight: bold;
}
.confidence-bullish {
    background-color: rgba(25, 135, 84, 0.1);
    color: #198754;
}
.confidence-bearish {
    background-color: rgba(220, 53, 69, 0.1);
    color: #dc3545;
}
.confidence-neutral {
    background-color: rgba(13, 202, 240, 0.1);
    color: #0dcaf0;
}
/* Estilos para o modal do Agent Demo */
.modal-xl .modal-content {
    background-color: #343a40;
    border-radius: 10px;
}

.modal-body {
    padding: 0;
}

#agent-demo-content iframe {
    background-color: #000;
    border-radius: 0 0 10px 10px;
}

.modal-header {
    border-bottom: 1px solid #444;
}

.modal-title {
    color: #0dcaf0;
    font-weight: bold;
}

.assistant-message-text {
    margin-bottom: 8px;
}

.assistant-message-audio {
    border-top: 1px solid rgba(255, 255, 255, 0.1);
    padding-top: 8px;
}

/* Voice selector styles */
.voice-selector {
    margin-bottom: 0px;
}

.voice-selector .Select-control {
    background-color: #343a40;
    border-color: #495057;
}

.voice-selector .Select-menu-outer {
    background-color: #343a40;
    border-color: #495057;
}

.voice-selector .Select-option {
    background-color: #343a40;
    color: white;
}

.voice-selector .Select-option.is-focused {
    background-color: #495057;
}
"""

# Monkey patch the print function to log messages to the debug area
original_print = print

def debug_print_wrapper(*args, **kwargs):
    """
    Wrapper for the built-in print function that also logs to the debug area
    """
    # Call the original print function
    original_print(*args, **kwargs)
    
    # Convert args to a single string
    message = " ".join(str(arg) for arg in args)
    
    # Determine log level based on message content
    level = "info"
    if any(keyword in message.lower() for keyword in ["error", "exception", "fail", "❌"]):
        level = "error"
    elif any(keyword in message.lower() for keyword in ["warning", "warn", "⚠️"]):
        level = "warning"
    elif any(keyword in message.lower() for keyword in ["success", "completed", "✅"]):
        level = "success"
    
    # Add to debug buffer
    debug_print(message, level)

# Uncomment to replace the global print function with our wrapper
# print = debug_print_wrapper