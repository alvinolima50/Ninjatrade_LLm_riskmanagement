"""
Script para verificar símbolos disponíveis no MetaTrader5
"""

import MetaTrader5 as mt5
import pandas as pd

def check_mt5_symbols():
    """Verifica e lista todos os símbolos disponíveis no MT5"""
    
    print("🔍 Verificando conexão com MetaTrader5...")
    
    # Inicializar MT5
    if not mt5.initialize():
        print("❌ Erro: Não foi possível conectar ao MetaTrader5")
        print("   Certifique-se de que o MT5 está aberto e logado")
        return
    
    # Obter informações do terminal
    terminal_info = mt5.terminal_info()
    if terminal_info:
        print(f"✅ Conectado ao MT5: {terminal_info.name}")
        print(f"   Empresa: {terminal_info.company}")
        print(f"   Caminho: {terminal_info.path}")
    
    # Obter conta
    account_info = mt5.account_info()
    if account_info:
        print(f"\n📊 Conta: {account_info.login}")
        print(f"   Servidor: {account_info.server}")
        print(f"   Saldo: {account_info.balance} {account_info.currency}")
    
    # Listar símbolos disponíveis
    print("\n📈 Símbolos disponíveis:")
    print("-" * 50)
    
    symbols = mt5.symbols_get()
    if symbols:
        # Organizar por categoria
        forex_symbols = []
        index_symbols = []
        commodity_symbols = []
        crypto_symbols = []
        other_symbols = []
        
        for symbol in symbols:
            if symbol.visible:  # Apenas símbolos visíveis
                name = symbol.name
                
                # Categorizar
                if any(curr in name for curr in ["EUR", "USD", "GBP", "JPY", "CHF", "CAD", "AUD", "NZD"]):
                    forex_symbols.append(name)
                elif any(idx in name for idx in ["DAX", "SP500", "NASDAQ", "DOW", "FTSE"]):
                    index_symbols.append(name)
                elif any(comm in name for comm in ["GOLD", "SILVER", "OIL", "GAS", "COPPER"]):
                    commodity_symbols.append(name)
                elif any(crypto in name for crypto in ["BTC", "ETH", "CRYPTO"]):
                    crypto_symbols.append(name)
                else:
                    other_symbols.append(name)
        
        # Exibir por categoria
        if forex_symbols:
            print("\n💱 FOREX:")
            for i, symbol in enumerate(forex_symbols[:10]):  # Mostrar apenas os primeiros 10
                print(f"   {symbol}")
            if len(forex_symbols) > 10:
                print(f"   ... e mais {len(forex_symbols) - 10} símbolos")
        
        if index_symbols:
            print("\n📊 ÍNDICES:")
            for symbol in index_symbols[:10]:
                print(f"   {symbol}")
        
        if commodity_symbols:
            print("\n🛢️ COMMODITIES:")
            for symbol in commodity_symbols[:10]:
                print(f"   {symbol}")
        
        if crypto_symbols:
            print("\n🪙 CRYPTO:")
            for symbol in crypto_symbols[:10]:
                print(f"   {symbol}")
        
        if other_symbols:
            print("\n📌 OUTROS:")
            for symbol in other_symbols[:10]:
                print(f"   {symbol}")
        
        print(f"\n📊 Total de símbolos visíveis: {len([s for s in symbols if s.visible])}")
        
        # Testar um símbolo específico
        print("\n🧪 Testando obtenção de dados para EURUSD...")
        test_symbols = ["EURUSD", "EURUSD.", "EUR/USD", "EURUSD.a", "EURUSDm"]
        
        for test_symbol in test_symbols:
            rates = mt5.copy_rates_from_pos(test_symbol, mt5.TIMEFRAME_M5, 0, 10)
            if rates is not None and len(rates) > 0:
                print(f"✅ Sucesso com: {test_symbol} - {len(rates)} candles obtidos")
                print(f"   Último preço: {rates[-1]['close']}")
                break
            else:
                print(f"❌ Falhou: {test_symbol}")
    else:
        print("❌ Nenhum símbolo encontrado")
    
    # Desconectar
    mt5.shutdown()
    print("\n✅ Desconectado do MT5")

if __name__ == "__main__":
    check_mt5_symbols()
    input("\nPressione Enter para sair...")
