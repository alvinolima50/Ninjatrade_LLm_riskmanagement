"""
Financial Market News Scraping Agent Simulation - Natural Gas Focus
Professional Version with Human-like Navigation

This module creates a visual demonstration of an AI agent actively searching
for natural gas futures market news and information. It displays real websites
with realistic mouse movement, scrolling, and human-like interaction patterns.
"""

import dash
from dash import dcc, html, callback_context
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import time
import random
import webbrowser
import threading
import datetime
import pandas as pd
import plotly.graph_objects as go
import json

# URLs focused on Natural Gas futures and energy markets
FINANCIAL_URLS = [
    {
        "url": "https://tradingeconomics.com/commodity/natural-gas",
        "description": "Analyzing Natural Gas Future Prices",
        "scroll_targets": [300, 600, 900, 1200],
        "hover_areas": ["#aspnetForm", ".table-responsive", ".chart-container"]
    },
    {
        "url": "https://tradingeconomics.com/commodity/natural-gas/forecast",
        "description": "Checking forecasts for Natural Gas",
        "scroll_targets": [400, 800, 1000],
        "hover_areas": [".forecast-table", ".chart-area"]
    },
    {
        "url": "https://tradingeconomics.com/commodity/natural-gas/news",
        "description": "Looking for recent news about Natural Gas",
        "scroll_targets": [200, 500, 800, 1100, 1400],
        "hover_areas": [".news-item", ".headline", ".timestamp"]
    },
    {
        "url": "https://tradingeconomics.com/commodity/natural-gas/technical",
        "description": "Analyzing Natural Gas Technical Indicators",
        "scroll_targets": [350, 700, 1050],
        "hover_areas": [".technical-indicators", ".signal-strength"]
    },
    {
        "url": "https://tradingeconomics.com/commodity/natural-gas/historical",
        "description": "Studying Natural Gas Price History",
        "scroll_targets": [250, 550, 850, 1150],
        "hover_areas": [".historical-data", ".price-chart"]
    },
    {
        "url": "https://tradingeconomics.com/stream?i=markets&c=commodity",
        "description": "Monitoring commodity flows in real time",
        "scroll_targets": [300, 600, 900],
        "hover_areas": [".stream-item", ".commodity-row"]
    },
    {
        "url": "https://tradingeconomics.com/commodity/uk-natural-gas",
        "description": "Comparing UK Natural Gas Prices",
        "scroll_targets": [400, 800],
        "hover_areas": [".price-widget", ".market-data"]
    },
    {
        "url": "https://tradingeconomics.com/commodity/eu-natural-gas",
        "description": "Checking the European Natural Gas Market",
        "scroll_targets": [350, 750, 1100],
        "hover_areas": [".eu-gas-price", ".market-analysis"]
    },
    {
        "url": "https://tradingeconomics.com/forecast/commodity",
        "description": "Analyzing general commodity forecasts",
        "scroll_targets": [200, 400, 600, 800, 1000],
        "hover_areas": [".commodity-forecast", ".natural-gas-row"]
    },
    {
        "url": "https://tradingeconomics.com/commodity/gasoline",
        "description": "Compared to the gasoline market",
        "scroll_targets": [300, 700],
        "hover_areas": [".gasoline-data", ".correlation-chart"]
    }
]

# Agent thoughts focused on Natural Gas market

AGENT_THOUGHTS = [
    "Analyzing volatility patterns in natural gas futures...",
    "Verifying correlation between weather patterns and natural gas prices...",
    "Processing natural gas storage data (EIA reports)...",
    "Evaluating impact of shale gas production on price dynamics...",
    "Monitoring seasonal demand patterns for natural gas...",
    "Calculating spread differentials between natural gas futures contracts...",
    "Analyzing LNG export flow metrics...",
    "Detecting accumulation patterns in natural gas contracts...",
    "Verifying support and resistance levels in NG futures...",
    "Processing natural gas inventory reports...",
    "Evaluating impact of extreme weather events on consumption...",
    "Monitoring US natural gas production rates...",
    "Analyzing industrial demand for natural gas resources...",
    "Calculating historical seasonality patterns for natural gas...",
    "Verifying positioning of commercial traders in futures market...",
    "Detecting technical divergences in natural gas charts...",
    "Processing residential consumption data across regions...",
    "Analyzing spread relationships between Henry Hub and TTF benchmarks...",
    "Monitoring available storage capacity utilization rates...",
    "Evaluating geopolitical risk factors in global gas markets...",
    "Calculating implied volatility in natural gas options chain...",
    "Verifying international pipeline flow volumes...",
    "Analyzing natural gas demand for power generation...",
    "Detecting anomalies in trading volume distributions...",
    "Processing long-range weather forecast models...",
    "Monitoring natural gas crack spread relationships...",
    "Evaluating impact of environmental regulations on production...",
    "Analyzing energy substitution trends across sectors...",
    "Examining AECO basis differential fluctuations...",
    "Mapping Canadian storage injection and withdrawal rates...",
    "Correlating rig count data with forward production estimates...",
    "Quantifying contango and backwardation curve structures...",
    "Modeling potential freeze-off impacts on wellhead production...",
    "Assessing cross-fuel switching thresholds for utilities...",
    "Calculating heat-rate dispatch curves for power plants...",
    "Analyzing Northeast basis blowouts during demand spikes...",
    "Measuring pipeline capacity utilization across key hubs...",
    "Evaluating storage cycle efficiency metrics year-over-year...",
    "Processing EIA Form 914 production survey data...",
    "Modeling Canadian export flows to Midwest markets...",
    "Calculating net-long positioning changes among non-commercials...",
    "Analyzing Marcellus/Utica production constraint factors...",
    "Monitoring Mexican export demand variability...",
    "Quantifying cash-to-futures basis relationships at major hubs...",
    "Evaluating Western Canadian takeaway capacity limitations...",
    "Processing floating storage and regasification unit (FSRU) utilization data...",
    "Analyzing Asian premium/discount relationships to Henry Hub...",
    "Calculating probability distributions for winter withdrawal scenarios...",
    "Modeling Dutch TTF and JKM spread relationships to US benchmarks...",
    "Examining producer hedging ratio changes quarter-over-quarter...",
    "Assessing deliverability constraints during peak demand periods...",
    "Analyzing Permian associated gas production growth rates...",
    "Calculating technical momentum oscillator divergences on multiple timeframes...",
    "Processing residential/commercial/industrial consumption ratios...",
    "Evaluating impact of carbon policy shifts on gas-to-coal switching...",
    "Modeling Alberta production response to AECO price signals...",
    "Analyzing hydrogen blending impact on natural gas infrastructure...",
    "Calculating coal-to-gas switching economics for power generators...",
    "Examining Northeast winter basis risk premium patterns...",
    "Processing European storage adequacy metrics pre-heating season...",
    "Modeling Northeast pipeline constraint scenarios during polar vortex events...",
    "Calculating technical fibonacci retracement levels across multiple timeframes...",
    "Analyzing Japanese utility procurement strategy shifts...",
    "Evaluating Appalachian production decline rates by vintage...",
    "Processing compression ratio data at major pipeline interconnects...",
    "Modeling liquefaction capacity utilization rates globally...",
    "Calculating Renewable Natural Gas (RNG) premium valuations...",
    "Analyzing Haynesville production response to price signals...",
    "Examining LNG vessel tracking data for supply-chain disruptions...",
    "Processing prompt-month volatility skew metrics...",
    "Modeling Canadian storage capacity utilization trends...",
    "Calculating technical Elliott wave pattern progressions...",
    "Analyzing Northeast demand elasticity during cold snaps...",
    "Evaluating Mountain Valley Pipeline capacity impacts on regional basis...",
    "Processing propane-to-natural gas price ratio relationships...",
    "Modeling Gulf Coast export capacity ramp rates...",
    "Calculating natural gas producer free cash flow sensitivity to price changes...",
    "Analyzing salt dome vs. depleted reservoir withdrawal rate capabilities...",
    "Examining technical MACD histogram divergence patterns...",
    "Processing UK NBP spread relationships to continental benchmarks...",
    "Modeling pipeline capacity release market liquidity metrics...",
    "Calculating technical RSI overbought/oversold conditions across contracts...",
    "Analyzing Calcasieu Pass LNG ramp-up schedule impacts...",
    "Evaluating technical Ichimoku cloud support/resistance levels...",
    "Processing Sabine Pass maintenance schedule implications...",
    "Modeling Western Canadian production freeze-off probabilities...",
    "Calculating technical Bollinger Band compression/expansion cycles...",
    "Analyzing storage delta vs. 5-year average correlations to price...",
    "Examining Northeast winter basis volatility patterns...",
    "Processing technical Gann fan angle projections...",
    "Modeling Southern California storage adequacy scenarios...",
    "Calculating technical Andrews' pitchfork channel boundaries...",
    "Analyzing Pacific Northwest hydro generation displacement effects...",
    "Evaluating Rockies production decline offset by Permian growth...",
    "Processing technical volume-weighted average price support/resistance levels..."
]

# Market insights specific to Natural Gas
MARKET_INSIGHTS = [
    "Possible declining inventory levels might affect winter supply dynamics",
    "Asian market sentiment appears to be shifting toward increased LNG demand",
    "Early indicators suggest potential stabilization in North American shale production",
    "Winter/summer spread differentials showing intriguing patterns worth monitoring",
    "January contract activity displaying unusual volume characteristics",
    "Historical correlation patterns with crude oil markets possibly breaking down",
    "Speculative positioning reportedly shifting in natural gas futures",
    "Long-range weather models hinting at potential colder-than-average winter scenarios",
    "Storage capacity utilization rates approaching noteworthy levels",
    "LNG export infrastructure seemingly operating near capacity limits",
    "Option market dynamics suggesting heightened price movement expectations",
    "Technical support zones emerging around key psychological price levels",
    "Divergence patterns forming in momentum indicators on daily timeframes",
    "Eastern European pipeline flow metrics showing potential irregularities",
    "Industrial consumption figures potentially exceeding analyst consensus",
    "Intercontinental price differentials widening beyond typical seasonal patterns",
    "Institutional order flow analysis indicating possible accumulation patterns",
    "Market structure potentially conducive to short-term squeezes",
    "Regime change indicators flashing across multiple technical frameworks",
    "Technical confluence suggesting possible directional alignment",
    "Currency correlation coefficients potentially shifting from historical norms",
    "Weekly chart formations hinting at possible reversal scenarios",
    "Residential heating demand projections trending above 5-year averages",
    "Gulf production infrastructure facing operational challenges",
    "Canadian AECO basis differential showing curious deviations from seasonal norms",
    "Alberta production metrics possibly influencing continental supply expectations",
    "Western Canadian storage dynamics creating interesting market implications",
    "TransCanada pipeline capacity utilization trends worth monitoring closely",
    "British Columbia production regions showing unexpected output signatures",
    "Canadian regulatory developments potentially impacting cross-border dynamics",
    "Northeast market premium to AECO hub presenting arbitrage considerations",
    "Eastern Canadian demand patterns displaying seasonal anomalies",
    "Ontario storage withdrawal rates deviating from historical precedent",
    "Montney formation productivity metrics warranting closer examination",
    "Canadian export capacity constraints possibly affecting market dynamics",
    "Potential shift in Canadian producers' hedging strategies being observed",
    "Dawn hub pricing mechanisms showing interesting behavioral patterns",
    "Weather pattern predictions for Western Canada suggesting monitoring opportunities",
    "LNG Canada project timeline developments potentially influencing futures curve",
    "Changing correlations between Canadian and Henry Hub pricing structures",
    "Quebec consumption trends displaying noteworthy seasonal adjustments",
    "Atlantic Canadian import facility utilization rates showing curious patterns",
    "Mackenzie Delta exploration sentiment possibly shifting among key players",
    "Northern Canadian pipeline project discussions potentially affecting long-dated contracts",
    "Western Canadian Sedimentary Basin production efficiency metrics raising questions",
    "Canadian natural gas liquid content ratios showing subtle changes",
    "Maritime provinces energy transition policies possibly impacting demand outlook",
    "Saskatchewan production-to-reserve ratios warranting closer analysis",
    "Canadian gas-to-power demand elasticity possibly transitioning to new paradigm",
    "Cross-border capacity auction results suggesting evolving market dynamics",
    "Canadian storage injection rates displaying intriguing seasonal variations",
    "Northwest Territories exploration sentiment possibly changing among participants",
    "Yukon energy infrastructure development talks potentially affecting regional outlook",
    "Canadian wellhead freeze-off probability models suggesting increased vigilance",
    "Manitoba consumption elasticity coefficients possibly shifting from historical patterns",
    "Pre-hedging activity among Canadian producers showing interesting trends",
    "Northeast gateway capacity constraints potentially affecting Canadian export routes",
    "Canadian drilling rig deployment patterns suggesting evolution in production strategy",
    "British Columbia regulatory climate potentially influencing development timeline expectations"
]
# Mouse movement patterns
MOUSE_PATTERNS = [
    {"name": "reading", "speed": "slow", "path": "linear"},
    {"name": "scanning", "speed": "medium", "path": "zigzag"},
    {"name": "clicking", "speed": "fast", "path": "direct"},
    {"name": "hovering", "speed": "slow", "path": "circular"},
    {"name": "scrolling", "speed": "medium", "path": "vertical"}
]

class MouseSimulator:
    """Simulates realistic mouse movements and interactions"""
    
    def __init__(self):
        self.x = random.randint(100, 800)
        self.y = random.randint(100, 600)
        self.target_x = self.x
        self.target_y = self.y
        self.speed = 0.1
        self.pattern = "reading"
        self.last_movement = time.time()
        self.click_animation = False
        self.hover_time = 0
        self.reading_pause = False
        
    def set_target(self, x, y, pattern="direct"):
        """Set new target position with movement pattern"""
        self.target_x = x
        self.target_y = y
        self.pattern = pattern
        
        # Add some randomness to make it more human-like
        self.target_x += random.randint(-20, 20)
        self.target_y += random.randint(-20, 20)
        
    def update(self):
        """Update mouse position with realistic movement"""
        current_time = time.time()
        dt = current_time - self.last_movement
        self.last_movement = current_time
        
        # Calculate distance to target
        dx = self.target_x - self.x
        dy = self.target_y - self.y
        distance = (dx**2 + dy**2)**0.5
        
        if distance > 5:
            # Add micro-movements for realism
            micro_x = random.uniform(-2, 2)
            micro_y = random.uniform(-2, 2)
            
            # Different movement patterns
            if self.pattern == "reading":
                # Slow, horizontal movement while reading
                self.x += (dx * 0.05 + micro_x) * dt * 60
                self.y += (dy * 0.02 + micro_y) * dt * 60
            elif self.pattern == "scanning":
                # Faster, more erratic movement
                self.x += (dx * 0.15 + micro_x * 3) * dt * 60
                self.y += (dy * 0.15 + micro_y * 3) * dt * 60
            elif self.pattern == "direct":
                # Direct movement to target
                self.x += dx * 0.2 * dt * 60
                self.y += dy * 0.2 * dt * 60
            else:
                # Default smooth movement
                self.x += (dx * 0.1 + micro_x) * dt * 60
                self.y += (dy * 0.1 + micro_y) * dt * 60
                
        else:
            # Add small idle movements when at target
            if random.random() < 0.1:
                self.x += random.uniform(-1, 1)
                self.y += random.uniform(-1, 1)
                
        # Keep within bounds
        self.x = max(10, min(1200, self.x))
        self.y = max(10, min(800, self.y))
        
        return self.x, self.y
    
    def simulate_click(self):
        """Simulate a click animation"""
        self.click_animation = True
        
    def simulate_hover(self):
        """Simulate hovering behavior"""
        self.hover_time = time.time()
        
    def get_cursor_style(self):
        """Get appropriate cursor style based on current action"""
        if self.click_animation:
            return "pointer"
        elif self.hover_time and time.time() - self.hover_time < 2:
            return "pointer"
        else:
            return "default"

class AgentSimulation:
    """Enhanced agent simulation with realistic browsing behavior"""
    
    def __init__(self):
        self.current_url_index = 0
        self.logs = []
        self.active = False
        self.last_thought_time = time.time()
        self.insights_generated = []
        self.agent_thread = None
        self.mouse = MouseSimulator()
        self.current_scroll = 0
        self.target_scroll = 0
        self.scroll_speed = 0
        self.last_interaction = time.time()
        self.reading_content = False
        self.focus_areas = []
        
    def start(self):
        """Start the agent simulation"""
        if not self.active:
            self.active = True
            self.logs = []
            self.add_log("AI agent initialized. Starting Natural Gas market analysis...", "system")
            self.add_log("Main focus: Natural Gas Futures and related news", "system")
            
            self.agent_thread = threading.Thread(target=self._run_agent)
            self.agent_thread.daemon = True
            self.agent_thread.start()
            
            return True
        return False
            
    def stop(self):
        """Stop the agent simulation"""
        if self.active:
            self.active = False
            self.add_log("Agent finalizing analysis. Compiling insights on Natural Gas...", "system")
            return True
        return False
    
    def _run_agent(self):
        """Enhanced agent behavior with realistic interactions"""
        try:
            while self.active:
                current_time = time.time()
                
                # Update mouse position
                self.mouse.update()
                
                # Simulate scrolling
                if abs(self.target_scroll - self.current_scroll) > 10:
                    scroll_diff = self.target_scroll - self.current_scroll
                    self.current_scroll += scroll_diff * 0.1
                    
                # Generate thoughts about Natural Gas
                if current_time - self.last_thought_time > random.uniform(3, 7):
                    self._generate_thought()
                    self.last_thought_time = current_time
                
                # Generate insights
                if random.random() < 0.15:
                    self._generate_insight()
                
                # Simulate realistic browsing patterns
                if current_time - self.last_interaction > random.uniform(2, 5):
                    self._simulate_interaction()
                    self.last_interaction = current_time
                
                time.sleep(0.05)  # 20 FPS for smooth animation
                
        except Exception as e:
            self.add_log(f"Erro no processamento do agent: {str(e)}", "error")
    
    def _simulate_interaction(self):
        """Simulate realistic user interactions"""
        interaction_type = random.choice([
            "scroll", "hover", "click", "read", "pause"
        ])
        
        url_data = self.get_current_url()
        
        if interaction_type == "scroll":
            # Simulate scrolling to areas of interest
            if url_data.get("scroll_targets"):
                target = random.choice(url_data["scroll_targets"])
                self.target_scroll = target
                self.add_log(f"Navegando pela página... posição: {target}px", "interaction")
                
                # Move mouse during scroll
                self.mouse.set_target(
                    random.randint(400, 800),
                    random.randint(300, 500),
                    "scanning"
                )
                
        elif interaction_type == "hover":
            # Simulate hovering over important elements
            self.mouse.simulate_hover()
            self.mouse.set_target(
                random.randint(300, 900),
                random.randint(200, 600),
                "reading"
            )
            self.add_log("Analisando elementos da página...", "interaction")
            
        elif interaction_type == "click":
            # Simulate clicking behavior
            self.mouse.simulate_click()
            self.add_log("Interagindo com dados de Gás Natural...", "interaction")
            
        elif interaction_type == "read":
            # Simulate reading behavior
            self.reading_content = True
            self.mouse.pattern = "reading"
            self.add_log("Lendo informações detalhadas sobre mercado de gás...", "interaction")
            
        elif interaction_type == "pause":
            # Simulate thinking/processing pause
            self.add_log("Processando informações coletadas...", "interaction")
    
    def get_current_url(self):
        """Get the current URL data"""
        return FINANCIAL_URLS[self.current_url_index]
    
    def move_to_next_url(self):
        """Move to next URL with realistic transition"""
        url_data = self.get_current_url()
        self.add_log(f"Navegando para: {url_data['description']}", "navigation")
        self.add_log(f"URL: {url_data['url']}", "navigation")
        
        # Reset scroll and mouse for new page
        self.current_scroll = 0
        self.target_scroll = 0
        self.mouse.set_target(
            random.randint(400, 800),
            random.randint(100, 300),
            "direct"
        )
        
        self.current_url_index = (self.current_url_index + 1) % len(FINANCIAL_URLS)
        return url_data
    
    def _generate_thought(self):
        """Generate Natural Gas focused thoughts"""
        thought = random.choice(AGENT_THOUGHTS)
        self.add_log(thought, "thought")
        
        # Add data extraction message
        extraction_targets = [
            'tabela de preços', 'gráfico de futuros', 'notícia sobre gás natural',
            'relatório de inventário', 'análise técnica', 'previsão de demanda',
            'dados de produção', 'informações de LNG', 'spreads de contratos'
        ]
        self.add_log(f"Extraindo dados de {random.choice(extraction_targets)}", "extraction")
    
    def _generate_insight(self):
        """Generate Natural Gas market insights"""
        available_insights = [i for i in MARKET_INSIGHTS if i not in self.insights_generated[-5:]]
        
        if not available_insights:
            available_insights = MARKET_INSIGHTS
            
        insight = random.choice(available_insights)
        self.insights_generated.append(insight)
        
        confidence = random.randint(65, 99)
        self.add_log(f"INSIGHT (confiança: {confidence}%): {insight}", "insight")
        
        # Add follow-up action
        if confidence > 85:
            self.add_log("⚠️ Highly relevant signal detected for Natural Gas", "signal")
    
    def add_log(self, message, log_type):
        """Add a log entry with timestamp"""
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.logs.append({
            "timestamp": timestamp,
            "message": message,
            "type": log_type
        })
        
        if len(self.logs) > 100:
            self.logs = self.logs[-100:]
    
    def get_logs(self):
        """Get all logs"""
        return self.logs
    
    def get_mouse_position(self):
        """Get current mouse position"""
        return self.mouse.x, self.mouse.y
    
    def get_scroll_position(self):
        """Get current scroll position"""
        return self.current_scroll
    
    def is_active(self):
        """Check if simulation is active"""
        return self.active

# Create the enhanced Dash app
def create_agent_demo_app(server=None):
    """Create the professional agent demo dashboard"""
    
    if server:
        app = dash.Dash(
            __name__,
            server=server,
            url_base_pathname='/agent-demo/',
            external_stylesheets=[dbc.themes.DARKLY],
            suppress_callback_exceptions=True
        )
    else:
        app = dash.Dash(
            __name__,
            external_stylesheets=[dbc.themes.DARKLY],
            suppress_callback_exceptions=True
        )
    
    app.title = "Agent IA - Análise de Gás Natural"
    
    # Create agent instance
    agent = AgentSimulation()
    
    # Initial setup
    initial_url = FINANCIAL_URLS[0]["url"]
    initial_description = FINANCIAL_URLS[0]["description"]
    
    # Enhanced layout with professional styling
    app.layout = dbc.Container([
        dbc.Row([
            dbc.Col([
                html.Div([
                    # Professional header
                    html.Div([
                        html.H2([
                            html.I(className="fas fa-brain me-2"),
                            "Intelligence Agent - Natural Gas Markets"
                        ], className="text-primary mb-0"),
                        html.P("Real-Time Analysis of Natural Gas Futures",
                               className="text-secondary mt-1")
                    ], className="text-center mb-4"),
                    
                    html.Hr(style={"borderColor": "#444"}),
                    
                    # Status panel with animated elements
                    dbc.Card([
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    html.Div([
                                        html.Div([
                                            html.I(className="fas fa-circle text-success me-2 blink"),
                                            html.Span("ONLINE", className="text-success fw-bold")
                                        ]),
                                        html.Small("Active Agent", className="text-muted")
                                    ])
                                ], width=3),
                                dbc.Col([
                                    html.Div([
                                        html.Div([
                                            html.I(className="fas fa-fire me-2 text-warning"),
                                            html.Span("NATURAL GAS", className="text-warning fw-bold")
                                        ]),
                                        html.Small("Main Focus", className="text-muted")
                                    ])
                                ], width=3),
                                dbc.Col([
                                    html.Div([
                                        html.Div([
                                            html.I(className="fas fa-chart-line me-2 text-info"),
                                            html.Span("Futures market", className="text-info fw-bold")
                                        ]),
                                        html.Small("Target Market", className="text-muted")
                                    ])
                                ], width=3),
                                dbc.Col([
                                    html.Div([
                                        html.Div(id="session-timer", className="text-white fw-bold"),
                                        html.Small("Session Time", className="text-muted")
                                    ])
                                ], width=3)
                            ])
                        ])
                    ], className="mb-4", style={"backgroundColor": "#1a1a1a"}),
                    
                    # Metrics dashboard
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardBody([
                                    html.H6([
                                        html.I(className="fas fa-globe me-2"),
                                        "Sources Analyzed"
                                    ], className="text-center text-muted mb-2"),
                                    html.H2("0", id="sources-count", 
                                            className="text-center text-info mb-0 counter"),
                                    dbc.Progress(id="sources-progress", value=0, 
                                                color="info", className="mt-2", style={"height": "5px"})
                                ])
                            ], style={"backgroundColor": "#1a1a1a", "border": "1px solid #333"})
                        ], width=4),
                        dbc.Col([
                            dbc.Card([
                                dbc.CardBody([
                                    html.H6([
                                        html.I(className="fas fa-lightbulb me-2"),
                                        "Insights Generated"
                                    ], className="text-center text-muted mb-2"),
                                    html.H2("0", id="insights-count", 
                                            className="text-center text-warning mb-0 counter"),
                                    dbc.Progress(id="insights-progress", value=0, 
                                                color="warning", className="mt-2", style={"height": "5px"})
                                ])
                            ], style={"backgroundColor": "#1a1a1a", "border": "1px solid #333"})
                        ], width=4),
                        dbc.Col([
                            dbc.Card([
                                dbc.CardBody([
                                    html.H6([
                                        html.I(className="fas fa-exclamation-triangle me-2"),
                                        "Critical Signals"
                                    ], className="text-center text-muted mb-2"),
                                    html.H2("0", id="signals-count", 
                                            className="text-center text-danger mb-0 counter"),
                                    dbc.Progress(id="signals-progress", value=0, 
                                                color="danger", className="mt-2", style={"height": "5px"})
                                ])
                            ], style={"backgroundColor": "#1a1a1a", "border": "1px solid #333"})
                        ], width=4)
                    ], className="mb-4"),
                    
                    # Enhanced browser simulation with mouse cursor
                    dbc.Card([
                        dbc.CardHeader([
                            dbc.Row([
                                dbc.Col([
                                    html.Div([
                                        html.I(className="fas fa-satellite-dish me-2 text-primary"),
                                        html.Span("Live Intelligence Gathering", className="fw-bold")
                                    ])
                                ], width=6),
                                dbc.Col([
                                    html.Div([
                                        html.I(className="fas fa-map-marker-alt me-2 text-info"),
                                        html.Span(id="site-description", className="text-info")
                                    ], className="text-end")
                                ], width=6)
                            ])
                        ], style={"backgroundColor": "#2a2a2a"}),
                        dbc.CardBody([
                            # Browser chrome with professional styling
                            html.Div([
                                dbc.Row([
                                    dbc.Col([
                                        html.Div([
                                            html.Button("◀", className="browser-btn me-1"),
                                            html.Button("▶", className="browser-btn me-1"),
                                            html.Button("⟳", className="browser-btn me-3"),
                                        ], className="d-inline-block")
                                    ], width="auto"),
                                    dbc.Col([
                                        html.Div([
                                            html.I(className="fas fa-lock text-success me-2"),
                                            html.Span(initial_url, id="url-display", 
                                                     className="font-monospace")
                                        ], className="url-bar")
                                    ])
                                ])
                            ], className="mb-3"),
                            
                            # Browser viewport with mouse cursor overlay
                            html.Div([
                                # Loading bar
                                html.Div([
                                    html.Div(className="loading-bar", id="loading-bar")
                                ], className="loading-container"),
                                
                                # Mouse cursor simulation
                                html.Div([
                                    html.Div(id="mouse-cursor", className="mouse-cursor"),
                                    html.Div(id="mouse-trail", className="mouse-trail")
                                ], className="mouse-container"),
                                
                                # Scroll indicator
                                html.Div([
                                    html.I(className="fas fa-chevron-down scroll-indicator")
                                ], id="scroll-indicator", style={"display": "none"}),
                                
                                # Actual iframe
                                html.Iframe(
                                    id="browser-iframe",
                                    src=initial_url,
                                    className="browser-viewport",
                                    sandbox="allow-same-origin allow-scripts"
                                )
                            ], className="browser-container", id="browser-container")
                        ], style={"backgroundColor": "#1a1a1a"})
                    ], className="mb-4", style={"border": "1px solid #333"}),
                    
                    # Enhanced activity log with categorization
                    dbc.Card([
                        dbc.CardHeader([
                            dbc.Row([
                                dbc.Col([
                                    html.H5([
                                        html.I(className="fas fa-terminal me-2"),
                                        "Agent Activity Monitor"
                                    ], className="mb-0")
                                ], width=6),
                                dbc.Col([
                                    html.Div([
                                        html.Span("Logs: ", className="text-muted"),
                                        html.Span("0", id="log-count", className="text-white"),
                                        html.Span(" | ", className="text-muted mx-2"),
                                        html.I(className="fas fa-sync-alt me-1 text-success spin"),
                                        html.Span("Live", className="text-success small")
                                    ], className="text-end")
                                ], width=6)
                            ])
                        ], style={"backgroundColor": "#2a2a2a"}),
                        dbc.CardBody([
                            html.Div(id="agent-log", className="agent-log-container")
                        ], style={"backgroundColor": "#0a0a0a"})
                    ], className="mb-4", style={"border": "1px solid #333"}),
                    
                    # Natural Gas specific visualizations
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader([
                                    html.H6([
                                        html.I(className="fas fa-chart-area me-2"),
                                       "Sentiment Analysis - Natural Gas"
                                    ], className="mb-0")
                                ], style={"backgroundColor": "#2a2a2a"}),
                                dbc.CardBody([
                                    dcc.Graph(
                                        id="sentiment-graph",
                                        config={"displayModeBar": False}
                                    )
                                ], style={"backgroundColor": "#1a1a1a"})
                            ], style={"border": "1px solid #333"})
                        ], width=6),
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader([
                                    html.H6([
                                        html.I(className="fas fa-tachometer-alt me-2"),
                                       "Market Indicators"
                                    ], className="mb-0")
                                ], style={"backgroundColor": "#2a2a2a"}),
                                dbc.CardBody([
                                    dcc.Graph(
                                        id="market-indicators",
                                        config={"displayModeBar": False}
                                    )
                                ], style={"backgroundColor": "#1a1a1a"})
                            ], style={"border": "1px solid #333"})
                        ], width=6)
                    ], className="mb-4"),
                    
                    # Hidden components for state management
                    dcc.Store(id="simulation-active", data=True),
                    dcc.Store(id="mouse-position", data={"x": 400, "y": 300}),
                    dcc.Store(id="scroll-position", data=0),
                    dcc.Interval(
                        id="update-interval",
                        interval=50,  # 20 FPS for smooth animations
                        n_intervals=0
                    ),
                    dcc.Interval(
                        id="browser-update-interval",
                        interval=15000,  # 15 seconds between URL changes
                        n_intervals=0
                    ),
                    dcc.Interval(
                        id="slow-update-interval",
                        interval=1000,  # 1 second for counters
                        n_intervals=0
                    )
                ], className="p-4", style={"backgroundColor": "#0f0f0f"})
            ], width=12)
        ])
    ], fluid=True, className="bg-dark text-light", style={"backgroundColor": "#000"})
    
    # Enhanced CSS with professional animations and styling
    app.index_string = '''
    <!DOCTYPE html>
    <html>
        <head>
            {%metas%}
            <title>{%title%}</title>
            {%favicon%}
            {%css%}
            <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.3/css/all.min.css">
            <style>
                @import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@300;400;700&display=swap');
                
                body {
                    background-color: #000;
                    color: #f8f9fa;
                    font-family: 'Roboto Mono', monospace;
                    overflow-x: hidden;
                }
                
                /* Animations */
                @keyframes blinker {
                    50% { opacity: 0.3; }
                }
                
                @keyframes spin {
                    from { transform: rotate(0deg); }
                    to { transform: rotate(360deg); }
                }
                
                @keyframes pulse {
                    0% { transform: scale(1); }
                    50% { transform: scale(1.05); }
                    100% { transform: scale(1); }
                }
                
                @keyframes slideIn {
                    from { 
                        opacity: 0;
                        transform: translateX(-20px);
                    }
                    to {
                        opacity: 1;
                        transform: translateX(0);
                    }
                }
                
                @keyframes glow {
                    0% { box-shadow: 0 0 5px rgba(0, 123, 255, 0.5); }
                    50% { box-shadow: 0 0 20px rgba(0, 123, 255, 0.8); }
                    100% { box-shadow: 0 0 5px rgba(0, 123, 255, 0.5); }
                }
                
                .blink {
                    animation: blinker 1.5s linear infinite;
                }
                
                .spin {
                    animation: spin 2s linear infinite;
                }
                
                .counter {
                    animation: pulse 2s ease-in-out infinite;
                }
                
                /* Browser styling */
                .browser-btn {
                    background: #333;
                    border: 1px solid #444;
                    color: #999;
                    padding: 4px 8px;
                    border-radius: 3px;
                    font-size: 12px;
                    cursor: pointer;
                    transition: all 0.2s;
                }
                
                .browser-btn:hover {
                    background: #444;
                    color: #fff;
                }
                
                .url-bar {
                    background: #1a1a1a;
                    border: 1px solid #333;
                    padding: 8px 12px;
                    border-radius: 20px;
                    font-size: 13px;
                    width: 100%;
                    transition: all 0.3s;
                }
                
                .url-bar:hover {
                    border-color: #007bff;
                    animation: glow 2s ease-in-out;
                }
                
                .browser-container {
                    position: relative;
                    width: 100%;
                    height: 600px;
                    overflow: hidden;
                    background: #000;
                    border-radius: 5px;
                }
                
                .browser-viewport {
                    width: 100%;
                    height: 100%;
                    border: none;
                    border-radius: 5px;
                    background: #fff;
                }
                
                /* Mouse cursor simulation */
                .mouse-container {
                    position: absolute;
                    top: 0;
                    left: 0;
                    width: 100%;
                    height: 100%;
                    pointer-events: none;
                    z-index: 1000;
                }
                
                .mouse-cursor {
                    position: absolute;
                    width: 20px;
                    height: 20px;
                    transition: all 0.1s ease-out;
                    pointer-events: none;
                }
                
                .mouse-cursor::before {
                    content: '';
                    position: absolute;
                    width: 0;
                    height: 0;
                    border-left: 8px solid transparent;
                    border-right: 8px solid transparent;
                    border-top: 16px solid #00ff00;
                    filter: drop-shadow(0 0 3px #00ff00);
                    transform: rotate(45deg);
                }
                
                .mouse-cursor.clicking::before {
                    border-top-color: #ff0;
                    transform: rotate(45deg) scale(0.8);
                }
                
                .mouse-trail {
                    position: absolute;
                    width: 100%;
                    height: 100%;
                    opacity: 0.3;
                }
                
                .mouse-trail::before,
                .mouse-trail::after {
                    content: '';
                    position: absolute;
                    width: 1px;
                    background: #00ff00;
                    opacity: 0.5;
                }
                
                .mouse-trail::before {
                    height: 100%;
                    left: var(--mouse-x, 50%);
                }
                
                .mouse-trail::after {
                    width: 100%;
                    top: var(--mouse-y, 50%);
                }
                
                /* Loading bar */
                .loading-container {
                    position: absolute;
                    top: 0;
                    left: 0;
                    width: 100%;
                    height: 3px;
                    background: rgba(0, 0, 0, 0.1);
                    z-index: 999;
                }
                
                .loading-bar {
                    height: 100%;
                    width: 0%;
                    background: linear-gradient(90deg, #007bff 0%, #00ff00 50%, #007bff 100%);
                    background-size: 200% 100%;
                    animation: loading 2s linear;
                    transition: width 2s ease-out;
                }
                
                @keyframes loading {
                    0% { background-position: 200% 0; }
                    100% { background-position: -200% 0; }
                }
                
                /* Scroll indicator */
                .scroll-indicator {
                    position: absolute;
                    right: 20px;
                    bottom: 20px;
                    font-size: 24px;
                    color: #00ff00;
                    animation: bounce 1s ease-in-out infinite;
                    z-index: 998;
                }
                
                @keyframes bounce {
                    0%, 100% { transform: translateY(0); }
                    50% { transform: translateY(10px); }
                }
                
                /* Activity log styling */
                .agent-log-container {
                    height: 400px;
                    overflow-y: auto;
                    padding: 10px;
                    font-family: 'Roboto Mono', monospace;
                    font-size: 13px;
                    scrollbar-width: thin;
                    scrollbar-color: #333 #0a0a0a;
                }
                
                .agent-log-container::-webkit-scrollbar {
                    width: 8px;
                }
                
                .agent-log-container::-webkit-scrollbar-track {
                    background: #0a0a0a;
                }
                
                .agent-log-container::-webkit-scrollbar-thumb {
                    background: #333;
                    border-radius: 4px;
                }
                
                .log-entry {
                    padding: 8px 12px;
                    margin-bottom: 8px;
                    border-left: 3px solid #2c3e50;
                    font-family: 'Roboto Mono', monospace;
                    animation: slideIn 0.3s ease-out;
                    transition: all 0.2s;
                }
                
                .log-entry:hover {
                    background-color: rgba(255, 255, 255, 0.05);
                    padding-left: 16px;
                }
                
                .log-timestamp {
                    color: #6c757d;
                    font-size: 11px;
                    margin-right: 12px;
                    font-weight: 300;
                }
                
                .log-navigation {
                    border-left-color: #17a2b8;
                    background-color: rgba(23, 162, 184, 0.1);
                }
                
                .log-thought {
                    border-left-color: #6f42c1;
                    background-color: rgba(111, 66, 193, 0.1);
                }
                
                .log-extraction {
                    border-left-color: #20c997;
                    background-color: rgba(32, 201, 151, 0.1);
                }
                
                .log-insight {
                    border-left-color: #ffc107;
                    background-color: rgba(255, 193, 7, 0.15);
                    font-weight: bold;
                    animation: pulse 2s ease-in-out;
                }
                
                .log-signal {
                    border-left-color: #dc3545;
                    background-color: rgba(220, 53, 69, 0.15);
                    font-weight: bold;
                    animation: glow 2s ease-in-out infinite;
                }
                
                .log-system {
                    border-left-color: #6c757d;
                    background-color: rgba(108, 117, 125, 0.1);
                }
                
                .log-interaction {
                    border-left-color: #00ff00;
                    background-color: rgba(0, 255, 0, 0.05);
                    font-style: italic;
                }
                
                /* Progress bars */
                .progress {
                    background-color: #1a1a1a;
                }
                
                /* Cards */
                .card {
                    transition: all 0.3s;
                    border: 1px solid transparent;
                }
                
                .card:hover {
                    transform: translateY(-2px);
                    border-color: #333;
                }
                
                /* Typography */
                h1, h2, h3, h4, h5, h6 {
                    font-weight: 700;
                    letter-spacing: 1px;
                }
                
                .font-monospace {
                    font-family: 'Roboto Mono', monospace;
                }
                
                /* Responsive */
                @media (max-width: 768px) {
                    .browser-container {
                        height: 400px;
                    }
                    
                    .agent-log-container {
                        height: 300px;
                    }
                }
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
    
    # Callbacks
    @app.callback(
        Output("simulation-active", "data"),
        [Input("update-interval", "n_intervals")],
        [State("simulation-active", "data")]
    )
    def manage_simulation(n_intervals, is_active):
        if n_intervals == 0 and not agent.is_active():
            agent.start()
            return True
        return is_active
    
    # Update browser frame with smooth transitions
    @app.callback(
        [Output("browser-iframe", "src"),
         Output("url-display", "children"),
         Output("site-description", "children"),
         Output("loading-bar", "style")],
        [Input("browser-update-interval", "n_intervals")]
    )
    def update_browser_frame(n_intervals):
        if n_intervals == 0:
            url_data = FINANCIAL_URLS[0]
            return url_data["url"], url_data["url"], url_data["description"], {"width": "0%"}
        
        url_data = agent.move_to_next_url()
        
        # Trigger loading animation
        loading_style = {"width": "100%", "transition": "width 2s ease-out"}
        
        return url_data["url"], url_data["url"], url_data["description"], loading_style
    
    # Update mouse position and scroll
    @app.callback(
        [Output("mouse-cursor", "style"),
         Output("mouse-trail", "style"),
         Output("browser-container", "style"),
         Output("scroll-indicator", "style")],
        [Input("update-interval", "n_intervals")]
    )
    def update_mouse_and_scroll(n_intervals):
        if not agent.is_active():
            return {}, {}, {}, {"display": "none"}
        
        # Get mouse position
        x, y = agent.get_mouse_position()
        scroll = agent.get_scroll_position()
        
        # Mouse cursor style
        cursor_class = "clicking" if agent.mouse.click_animation else ""
        cursor_style = {
            "left": f"{x}px",
            "top": f"{y}px",
            "transform": "translate(-50%, -50%)"
        }
        
        # Mouse trail crosshair
        trail_style = {
            "--mouse-x": f"{x}px",
            "--mouse-y": f"{y}px"
        }
        
        # Simulated scroll
        container_style = {
            "position": "relative",
            "width": "100%",
            "height": "600px",
            "overflow": "hidden",
            "background": "#000",
            "border-radius": "5px"
        }
        
        # Show scroll indicator when scrolling
        scroll_indicator_style = {
            "display": "block" if abs(agent.target_scroll - agent.current_scroll) > 50 else "none"
        }
        
        # Reset click animation
        if agent.mouse.click_animation:
            agent.mouse.click_animation = False
        
        return cursor_style, trail_style, container_style, scroll_indicator_style
    
    # Update activity log
    @app.callback(
        [Output("agent-log", "children"),
         Output("log-count", "children")],
        [Input("update-interval", "n_intervals")]
    )
    def update_agent_log(n_intervals):
        logs = agent.get_logs()
        
        log_entries = []
        for log in reversed(logs):
            icon = {
                "navigation": "fa-compass",
                "thought": "fa-brain",
                "extraction": "fa-database",
                "insight": "fa-lightbulb",
                "signal": "fa-exclamation-triangle",
                "system": "fa-cog",
                "interaction": "fa-mouse-pointer",
                "error": "fa-times-circle"
            }.get(log["type"], "fa-info-circle")
            
            log_entry = html.Div([
                html.I(className=f"fas {icon} me-2 text-muted"),
                html.Span(log["timestamp"], className="log-timestamp"),
                html.Span(log["message"])
            ], className=f"log-entry log-{log['type']}")
            
            log_entries.append(log_entry)
        
        return log_entries, len(logs)
    
    # Update metrics with animations
    @app.callback(
        [Output("sources-count", "children"),
         Output("sources-progress", "value"),
         Output("insights-count", "children"),
         Output("insights-progress", "value"),
         Output("signals-count", "children"),
         Output("signals-progress", "value")],
        [Input("slow-update-interval", "n_intervals")]
    )
    def update_metrics(n_intervals):
        logs = agent.get_logs()
        
        sources_count = sum(1 for log in logs if log["type"] == "navigation")
        insights_count = sum(1 for log in logs if log["type"] == "insight")
        signals_count = sum(1 for log in logs if log["type"] == "signal")
        
        # Progress values (max 10 for visual effect)
        sources_progress = min(sources_count * 10, 100)
        insights_progress = min(insights_count * 10, 100)
        signals_progress = min(signals_count * 20, 100)
        
        return (sources_count, sources_progress, 
                insights_count, insights_progress,
                signals_count, signals_progress)
    
    # Update session timer
    @app.callback(
        Output("session-timer", "children"),
        [Input("slow-update-interval", "n_intervals")]
    )
    def update_timer(n_intervals):
        minutes = n_intervals // 60
        seconds = n_intervals % 60
        return f"{minutes:02d}:{seconds:02d}"
    
    # Update sentiment graph for Natural Gas
    @app.callback(
        Output("sentiment-graph", "figure"),
        [Input("slow-update-interval", "n_intervals")]
    )
    def update_sentiment_graph(n_intervals):
        # Generate time series
        timestamps = pd.date_range(
            start=datetime.datetime.now() - datetime.timedelta(hours=4),
            end=datetime.datetime.now(),
            periods=40
        )
        
        # Natural Gas specific sentiment
        base_sentiment = [random.uniform(40, 60) for _ in range(40)]
        
        # Add realistic trends
        for i in range(1, len(base_sentiment)):
            base_sentiment[i] = 0.8 * base_sentiment[i-1] + 0.2 * base_sentiment[i]
            
            if random.random() < 0.1:
                base_sentiment[i] += random.uniform(-15, 15)
            
            base_sentiment[i] = max(0, min(100, base_sentiment[i]))
        
        # Create sentiment lines
        bullish = [min(100, s + random.uniform(10, 20)) for s in base_sentiment]
        bearish = [max(0, s - random.uniform(10, 20)) for s in base_sentiment]
        
        fig = go.Figure()
        
        # Bullish sentiment
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=bullish,
            mode='lines',
            name='Bullish',
            line=dict(color='#00ff00', width=2),
            fill='tonexty',
            fillcolor='rgba(0, 255, 0, 0.1)'
        ))
        
        # Neutral sentiment
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=base_sentiment,
            mode='lines',
            name='Neutral',
            line=dict(color='#17a2b8', width=2, dash='dash')
        ))
        
        # Bearish sentiment
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=bearish,
            mode='lines',
            name='Bearish',
            line=dict(color='#dc3545', width=2),
            fill='tozeroy',
            fillcolor='rgba(220, 53, 69, 0.1)'
        ))
        
        fig.update_layout(
            template="plotly_dark",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=10, r=10, t=30, b=10),
            height=250,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font=dict(size=10)
            ),
            xaxis=dict(
                showgrid=False,
                zeroline=False,
                tickfont=dict(size=10)
            ),
            yaxis=dict(
                title="Sentiment Score",
                showgrid=True,
                gridcolor='rgba(255,255,255,0.1)',
                zeroline=False,
                range=[0, 100],
                tickfont=dict(size=10)
            )
        )
        
        return fig
    
    # Update market indicators
    @app.callback(
        Output("market-indicators", "figure"),
        [Input("slow-update-interval", "n_intervals")]
    )
    def update_market_indicators(n_intervals):
        # Create gauge charts for Natural Gas indicators
        fig = go.Figure()
        
        # Generate realistic values
        volatility = random.uniform(15, 85)
        momentum = random.uniform(20, 80)
        volume = random.uniform(30, 90)
        
        # Volatility gauge
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=volatility,
            title={'text': "Volatility"},
            domain={'x': [0, 0.3], 'y': [0, 1]},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 30], 'color': "lightgray"},
                    {'range': [30, 70], 'color': "gray"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 70
                }
            }
        ))
        
        # Momentum gauge
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=momentum,
            title={'text': "Momentum"},
            domain={'x': [0.35, 0.65], 'y': [0, 1]},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "green"},
                'steps': [
                    {'range': [0, 30], 'color': "lightgray"},
                    {'range': [30, 70], 'color': "gray"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 80
                }
            }
        ))
        
        # Volume gauge
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=volume,
            title={'text': "Volume"},
            domain={'x': [0.7, 1], 'y': [0, 1]},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "orange"},
                'steps': [
                    {'range': [0, 30], 'color': "lightgray"},
                    {'range': [30, 70], 'color': "gray"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 85
                }
            }
        ))
        
        fig.update_layout(
            template="plotly_dark",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            height=250,
            margin=dict(l=10, r=10, t=10, b=10),
            font=dict(size=10)
        )
        
        return fig
    
    return app, agent

# Run the app
if __name__ == "__main__":
    print("Iniciando Agent de Inteligência Artificial - Foco: Gás Natural...")
    print("Abrindo navegador para visualizar a demonstração...")
    
    app, agent = create_agent_demo_app()
    
    # Open browser automatically
    webbrowser.open_new("http://127.0.0.1:8051/")
    
    # Run the app
    app.run(debug=False, port=8051)
    
    # Clean up
    print("\nFinalizando agent...")
    agent.stop()