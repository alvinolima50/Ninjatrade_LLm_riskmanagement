"""
Módulo para exportar decisões do LLM para o NinjaTrader
Versão melhorada com mapeamento de símbolos MT5 -> NinjaTrader
"""
import os
from datetime import datetime

# Mapeamento de símbolos MT5 -> NinjaTrader
SYMBOL_MAP = {
    # Futuros de Energia
    "NGEN25": "NG",    # Natural Gas
    "NGEQ25": "NG",    # Natural Gas
    "NGEM25": "NG",    # Natural Gas (outro mês)
    "CLM25": "CL",     # Crude Oil
    "CLQ25": "CL",     # Crude Oil (outro mês)
    
    # Índices
    "ESM25": "ES",     # S&P 500 E-mini
    "ESH25": "ES",     # S&P 500 E-mini (março)
    "NQM25": "NQ",     # Nasdaq E-mini
    "NQH25": "NQ",     # Nasdaq E-mini (março)
    "YMM25": "YM",     # Dow E-mini
    
    # Moedas (Forex)
    "EURUSD": "EURUSD",
    "GBPUSD": "GBPUSD",
    "USDJPY": "USDJPY",
    
    # Metais
    "GCM25": "GC",     # Gold
    "SIM25": "SI",     # Silver
    
    # Adicione mais mapeamentos conforme necessário
}

def get_ninja_symbol(mt5_symbol):
    """
    Converte símbolo MT5 para símbolo NinjaTrader
    
    Args:
        mt5_symbol (str): Símbolo do MetaTrader5
        
    Returns:
        str: Símbolo do NinjaTrader ou None se não encontrado
    """
    # Primeiro tenta match direto
    if mt5_symbol in SYMBOL_MAP:
        return SYMBOL_MAP[mt5_symbol]
    
    # Se não encontrar, tenta remover números do final (para futuros)
    # Ex: "NGEN25" -> "NG"
    symbol_base = ''.join([c for c in mt5_symbol if not c.isdigit()])
    
    # Procura no mapa por símbolos que começam com a mesma base
    for mt5_sym, ninja_sym in SYMBOL_MAP.items():
        if mt5_sym.startswith(symbol_base):
            return ninja_sym
    
    # Se ainda não encontrar, retorna None
    return None


def save_ninjatrader_command(action, contracts, symbol="NG", file_path=None):
    """
    Salva comando de trading em arquivo TXT para o NinjaTrader executar
    
    Args:
        action (str): "ADD_CONTRACTS" ou "REMOVE_CONTRACTS" 
        contracts (int): Número de contratos
        symbol (str): Símbolo do ativo no MT5
        file_path (str): Caminho do arquivo (se None, usa o padrão do NinjaScript)
        
    Returns:
        dict: Status da operação com detalhes
    """
    # Caminho padrão do arquivo que o NinjaScript está monitorando
    if file_path is None:
        file_path = r"C:\Users\sousa\Documents\DataH\NinjatradeV2\nt_ng_trade.txt"
    
    # Verificar se a ação requer salvamento
    if action == "WAIT" or contracts == 0:
        return {
            "status": "skipped",
            "reason": f"No action needed - Action: {action}, Contracts: {contracts}",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    
    # Converter símbolo MT5 para NinjaTrader
    ninja_symbol = get_ninja_symbol(symbol)
    
    if ninja_symbol is None:
        # Se não encontrar mapeamento, tenta usar o símbolo original
        print(f"⚠️ Símbolo '{symbol}' não mapeado. Usando símbolo original.")
        ninja_symbol = symbol
    else:
        print(f"✅ Símbolo mapeado: {symbol} -> {ninja_symbol}")
    
    # Converter ação para formato NinjaTrader
    if action == "ADD_CONTRACTS":
        nt_action = "BUY"
    elif action == "REMOVE_CONTRACTS":
        nt_action = "SELL"
    else:
        return {
            "status": "error",
            "reason": f"Invalid action: {action}",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    
    # Formato esperado pelo NinjaScript: ACTION,QUANTITY,SYMBOL
    # Alguns scripts podem esperar apenas ACTION,QUANTITY
    # Verifique o formato esperado pelo seu NinjaScript
    command = f"{nt_action},{int(contracts)}"
    
    try:
        # Criar diretório se não existir
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Salvar comando no arquivo (sobrescreve o anterior)
        with open(file_path, 'w') as f:
            f.write(command)
        
        # Log detalhado
        log_message = f"""
╔══════════════════════════════════════════════════════╗
║          NINJATRADER COMMAND EXPORTED                ║
╠══════════════════════════════════════════════════════╣
║ Timestamp: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}                      ║
║ MT5 Symbol: {symbol}                                 ║
║ Ninja Symbol: {ninja_symbol}                         ║
║ Action: {nt_action}                                  ║
║ Contracts: {contracts}                               ║
║ Command: {command}                                   ║
║ File: {os.path.basename(file_path)}                  ║
╚══════════════════════════════════════════════════════╝
        """
        print(log_message)
        
        # Salvar backup do comando com timestamp
        backup_dir = r"C:\Users\sousa\Documents\DataH\ninjatradeBot\ninja_commands"
        if not os.path.exists(backup_dir):
            os.makedirs(backup_dir)
        
        backup_filename = f"command_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{symbol}.txt"
        backup_path = os.path.join(backup_dir, backup_filename)
        
        with open(backup_path, 'w') as f:
            f.write(f"Command: {command}\n")
            f.write(f"MT5 Symbol: {symbol}\n")
            f.write(f"Ninja Symbol: {ninja_symbol}\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Original Action: {action}\n")
            f.write(f"Contracts: {contracts}\n")
        
        return {
            "status": "success",
            "command": command,
            "mt5_symbol": symbol,
            "ninja_symbol": ninja_symbol,
            "file_path": file_path,
            "backup_path": backup_path,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
    except Exception as e:
        error_msg = f"Error saving NinjaTrader command: {str(e)}"
        print(f"❌ {error_msg}")
        return {
            "status": "error",
            "reason": error_msg,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }


def clear_ninjatrader_command(file_path=None):
    """
    Limpa/deleta o arquivo de comando após execução
    
    Args:
        file_path (str): Caminho do arquivo
    """
    if file_path is None:
        file_path = r"C:\Users\sousa\Documents\DataH\NinjatradeV2\nt_ng_trade.txt"
    
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"✅ NinjaTrader command file cleared: {file_path}")
            return True
    except Exception as e:
        print(f"❌ Error clearing command file: {e}")
        return False


def get_command_history(symbol=None, days=7):
    """
    Recupera histórico de comandos enviados
    
    Args:
        symbol (str): Filtrar por símbolo (opcional)
        days (int): Número de dias para buscar
        
    Returns:
        list: Lista de comandos enviados
    """
    backup_dir = r"C:\Users\sousa\Documents\DataH\ninjatradeBot\ninja_commands"
    commands = []
    
    if not os.path.exists(backup_dir):
        return commands
    
    try:
        # Listar todos os arquivos de backup
        for filename in os.listdir(backup_dir):
            if filename.startswith("command_") and filename.endswith(".txt"):
                if symbol and f"_{symbol}.txt" not in filename:
                    continue
                
                file_path = os.path.join(backup_dir, filename)
                
                # Ler conteúdo do arquivo
                with open(file_path, 'r') as f:
                    content = f.read()
                    
                # Extrair informações
                command_data = {}
                for line in content.split('\n'):
                    if ':' in line:
                        key, value = line.split(':', 1)
                        command_data[key.strip()] = value.strip()
                
                commands.append(command_data)
        
        # Ordenar por timestamp (mais recente primeiro)
        commands.sort(key=lambda x: x.get('Timestamp', ''), reverse=True)
        
        return commands
        
    except Exception as e:
        print(f"Error reading command history: {e}")
        return []


# Teste da função
if __name__ == "__main__":
    # Testar mapeamento de símbolos
    print("Testando mapeamento de símbolos:")
    test_symbols = ["NGEN25", "NGEM25", "CLM25", "ESM25", "EURUSD", "XYZ123","NGEQ25"]
    for sym in test_symbols:
        ninja_sym = get_ninja_symbol(sym)
        print(f"  {sym} -> {ninja_sym if ninja_sym else 'NÃO MAPEADO'}")
    
    print("\n" + "="*50 + "\n")
    
    # Testar salvamento de comando
    print("Testando exportação:")
    result = save_ninjatrader_command("ADD_CONTRACTS", 2, "NGEN25")
    print(f"Resultado: {result['status']}")
    
    if result['status'] == 'success':
        print(f"Símbolo Ninja usado: {result['ninja_symbol']}")