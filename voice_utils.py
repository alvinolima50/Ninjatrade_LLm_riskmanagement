"""
Voice utilities for text-to-speech functionality
"""
import base64
import io
import os
import tempfile
import threading
from openai import OpenAI

# Para reprodução de áudio local
import pygame

# Global variable for selected voice
selected_voice = "alloy"  # Default voice

# Inicializar pygame para reprodução de áudio
pygame.mixer.init()

# Available voices with descriptions
AVAILABLE_VOICES = {
    "alloy": "Neutral and balanced",
    "echo": "Deep and resonant",
    "fable": "British accent, authoritative",
    "onyx": "Deep and professional",
    "nova": "Warm and friendly",
    "shimmer": "Clear and precise"
}

def set_voice(voice_name):
    """
    Set the voice to use for text-to-speech
    
    Args:
        voice_name (str): Name of the voice to use
    """
    global selected_voice
    if voice_name in AVAILABLE_VOICES:
        selected_voice = voice_name
        return True
    return False

def get_selected_voice():
    """
    Get the currently selected voice
    
    Returns:
        str: Name of the currently selected voice
    """
    global selected_voice
    return selected_voice

def get_available_voices():
    """
    Get a dictionary of available voices with descriptions
    
    Returns:
        dict: Dictionary of available voices
    """
    return AVAILABLE_VOICES

def generate_speech(text, max_length=4000):
    """
    Generate speech from text using OpenAI's text-to-speech API
    
    Args:
        text (str): Text to convert to speech
        max_length (int): Maximum length of text to process
        
    Returns:
        str: Base64 encoded audio data that can be embedded in HTML
    """
    global selected_voice
    
    try:
        # Limit text length to avoid API errors
        if len(text) > max_length:
            print(f"Text too long ({len(text)} chars), truncating to {max_length} chars")
            text = text[:max_length] + "... [Text truncated for audio]"
        
        # Initialize OpenAI client
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Generate speech using OpenAI's TTS API
        response = client.audio.speech.create(
            model="tts-1",  # You can use tts-1-hd for higher quality
            voice=selected_voice,  # Use globally selected voice
            input=text
        )
        
        # Get the audio content
        audio_data = response.content
        
        # Convert binary audio data to base64 for HTML embedding
        audio_base64 = base64.b64encode(audio_data).decode('utf-8')
        
        print(f"Speech generated successfully using voice: {selected_voice}")
        return audio_base64
    except Exception as e:
        print(f"Error generating speech: {e}")
        return None

def play_audio_locally(audio_data):
    """
    Reproduz o áudio localmente usando pygame
    
    Args:
        audio_data (bytes): Dados de áudio em formato binário
    """
    try:
        # Criar um arquivo temporário para o áudio
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_audio_file:
            temp_audio_file.write(audio_data)
            temp_filename = temp_audio_file.name
        
        # Reproduzir o áudio usando pygame
        pygame.mixer.music.load(temp_filename)
        pygame.mixer.music.play()
        
        # Remover o arquivo após a reprodução em uma thread separada
        def cleanup_temp_file():
            pygame.mixer.music.set_endevent(pygame.USEREVENT)
            while pygame.mixer.music.get_busy():
                pygame.time.wait(100)
            if os.path.exists(temp_filename):
                try:
                    os.unlink(temp_filename)
                except Exception as e:
                    print(f"Erro ao remover arquivo temporário: {e}")
        
        # Iniciar thread de limpeza
        cleanup_thread = threading.Thread(target=cleanup_temp_file)
        cleanup_thread.daemon = True
        cleanup_thread.start()
        
        print(f"Reproduzindo áudio localmente...")
        return True
    except Exception as e:
        print(f"Erro ao reproduzir áudio localmente: {e}")
        return False
        
def generate_speech_for_response(text):
    """
    Helper function to generate speech for chat responses with additional processing
    Reproduz o áudio automaticamente usando o player local, sem exibir o player no navegador
    
    Args:
        text (str): Text to convert to speech
        
    Returns:
        str: String vazia ou None se houver erro
    """
    try:
        # Generate speech from the response text
        audio_base64 = generate_speech(text)
        
        # If speech generation successful
        if audio_base64:
            # Decodificar o áudio de base64 para binário
            audio_data = base64.b64decode(audio_base64)
            
            # Reproduzir o áudio localmente
            play_audio_locally(audio_data)
            
            # Retornar string vazia para não mostrar o player no chat
            return ""
        else:
            return None
    except Exception as e:
        print(f"Erro em generate_speech_for_response: {e}")
        return None
    
def process_chat_query_with_voice(query, symbol, timeframe, llm_chain=None, indicator_history_buffer=None, h4_market_context_summary=None, get_current_candle_data=None, get_indicator_buffer_summary=None, get_recent_analysis_summary=None, initialize_llm_chain=None, token_usage=None):
    """
    Processa queries do chat com acesso completo aos dados dos indicadores
    e gera respostas por voz usando OpenAI Text-to-Speech
    
    Args:
        query (str): A consulta do usuário
        symbol (str): Símbolo de trading
        timeframe (str): Timeframe selecionado
        llm_chain: Cadeia LLM para processamento (opcional)
        indicator_history_buffer: Buffer de histórico de indicadores (opcional)
        h4_market_context_summary: Resumo do contexto H4 (opcional)
        get_current_candle_data: Função para obter dados do candle atual (opcional)
        get_indicator_buffer_summary: Função para obter resumo do buffer (opcional)
        get_recent_analysis_summary: Função para obter resumo da análise recente (opcional)
        initialize_llm_chain: Função para inicializar a cadeia LLM (opcional)
        token_usage: Dicionário de uso de tokens (opcional)
    
    Returns:
        tuple: (response_text, audio_html) - texto da resposta e HTML do áudio
    """
    try:
        from langchain_openai import ChatOpenAI
        from langchain.prompts import ChatPromptTemplate
        from langchain_community.callbacks.manager import get_openai_callback
        import os
        import traceback
        
        # Check if LLM is initialized
        if llm_chain is None and initialize_llm_chain is not None:
            llm_chain = initialize_llm_chain()
        
        # Obter dados atuais dos indicadores
        try:
            current_indicator_data = get_current_candle_data(symbol, timeframe) if get_current_candle_data else "Indicator data not available"
            
            # Se houver erro nos dados, usar dados básicos
            if isinstance(current_indicator_data, str) and ("Erro" in current_indicator_data or "insuficiente" in current_indicator_data):
                current_indicator_data = "Indicator data not currently available"
        except Exception as e:
            current_indicator_data = f"Erro ao obter dados: {str(e)}"
        
        # Obter contexto H4 se disponível
        h4_context = h4_market_context_summary or "H4 context not available"
        
        # Obter resumo do buffer de indicadores
        indicator_summary = get_indicator_buffer_summary() if get_indicator_buffer_summary else "Indicator buffer summary not available"
        
        # Obter informações da análise mais recente
        recent_analysis = get_recent_analysis_summary() if get_recent_analysis_summary else "No recent market analysis available"
        
        # Create a prompt specific for chat queries with full data access
        chat_prompt = ChatPromptTemplate.from_template("""
        You are an expert market analyst with full access to trading data for {symbol}.
        
        # CURRENT MARKET DATA & INDICATORS
        {current_data}
        
        # H4 TIMEFRAME CONTEXT
        {h4_context}
        
        # INDICATOR HISTORY SUMMARY
        {indicator_summary}
        
        # RECENT ANALYSIS
        {recent_analysis}
        
        # USER QUESTION
        {query}
        important SUMMARY YOUR ANSWERS with maximum 50 words.
        Based on ALL the available data above, provide a comprehensive but conversational response.
        
        Guidelines:
        - Use specific indicator values when available 
        - Reference the indicator progression/trends from the history
        - Relate current timeframe to H4 context when relevant
        - If asked about charts or visuals, describe what the data shows
        - Keep responses informative but accessible
        - Include confidence levels and reasoning when discussing market direction
        
        Respond in a natural, funny tone like you're explaining to a fellow trader.
        """)
        
        # Create a temporary chain for this query
        chat_chain = (
            {"symbol": lambda x: symbol,
             "current_data": lambda x: current_indicator_data,
             "h4_context": lambda x: h4_context,
             "indicator_summary": lambda x: indicator_summary,
             "recent_analysis": lambda x: recent_analysis,
             "query": lambda x: query}
            | chat_prompt
            | ChatOpenAI(
                temperature=0.3,
                api_key=os.getenv("OPENAI_API_KEY"),
                model_name="gpt-4.1-mini" 
            )
        )
        
        with get_openai_callback() as cb:
            response = chat_chain.invoke({})
            
            # Update token usage counters if provided
            if token_usage is not None:
                token_usage["total_tokens"] += cb.total_tokens
                token_usage["prompt_tokens"] += cb.prompt_tokens
                token_usage["completion_tokens"] += cb.completion_tokens
            
            print(f"Chat query tokens used: {cb.total_tokens}")
        
        response_text = response.content
        
        # Generate speech from the response text
        audio_html = generate_speech_for_response(response_text)
        
        # Return both the text response and the audio HTML
        return response_text, audio_html
        
    except Exception as e:
        print(f"Error processing chat query: {e}")
        traceback.print_exc()
        
        error_message = f"Sorry, I couldn't process your question. Error: {str(e)}"
        return error_message, None

def handle_voice_command_in_chat(query, timestamp, chat_messages, chat_history):
    """
    Processa comandos de voz no chat como 'change voice to'
    
    Args:
        query (str): A consulta do usuário
        timestamp (str): Timestamp formatado para o chat
        chat_messages (list): Lista atual de mensagens do chat
        chat_history (dict): Histórico do chat
    
    Returns:
        tuple: (foi_processado, resultados)
            - foi_processado: True se foi um comando de voz
            - resultados: Tuple com (mensagens finais, resultados visuais, estilo visual, histórico atualizado, valor de entrada, valor de voz)
    """
    from dash import html, no_update
    
    if query.lower().startswith("change voice to "):
        voice_name = query.lower().replace("change voice to ", "").strip()
        available_voices = get_available_voices()
        
        # Update the voice selector dropdown as well
        if voice_name in available_voices:
            set_voice(voice_name)
            response_text = f"Voice changed to {voice_name} - {available_voices[voice_name]}"
        else:
            voice_options = ", ".join(available_voices.keys())
            response_text = f"Voice '{voice_name}' not found. Available voices: {voice_options}"
            
        # Skip regular processing for voice commands
        assistant_message = html.Div([
            html.Div(response_text, className="assistant-message"),
            html.Div(timestamp, className="timestamp")
        ])
        
        updated_messages = (chat_messages or []) + [html.Div([
            html.Div(query, className="user-message"),
            html.Div(timestamp, className="timestamp")
        ])]
        
        final_messages = updated_messages + [assistant_message]
        
        # Update chat history with this command
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
        
        # Return updated voice selector value
        return True, (final_messages, [], {"display": "none"}, updated_chat_history, "", voice_name if voice_name in available_voices else no_update)
    
    return False, None

def create_voice_selector_with_tooltip():
    """
    Cria um seletor de voz com um ícone de informação e tooltip
    
    Returns:
        html.Div: Componente completo do seletor de voz
    """
    from dash import html, dcc
    import dash_bootstrap_components as dbc
    
    return html.Div([
        html.Div([
            html.Span("Voice: ", className="text-muted small me-2"),
            dcc.Dropdown(
                id="voice-selector",
                options=[
                    {"label": f"{voice} - {desc}", "value": voice} 
                    for voice, desc in get_available_voices().items()
                ],
                value=get_selected_voice(),  # Default é a voz atualmente selecionada
                clearable=False,
                style={                         
                    "width": "200px",
                    "color": "black",
                    "backgroundColor": "white",
                    "fontSize": "0.8rem",
                },
                className="voice-selector" 
            ),
            html.Span(html.I(className="fas fa-info-circle ms-2"), id="voice-commands-tooltip", style={"cursor": "pointer"}),
            dbc.Tooltip(
                "You can also type 'change voice to [voice name]' in the chat to change the voice.",
                target="voice-commands-tooltip",
                placement="top"
            )
        ], className="d-flex align-items-center")
    ])