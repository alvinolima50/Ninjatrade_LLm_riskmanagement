"""
MÓDULO CORRIGIDO PARA CONEXÃO METATRADER5
==========================================
Corrige todos os problemas de conexão identificados no código original
Mantém 100% do pipeline NinjaTrader intacto
"""

import MetaTrader5 as mt5
import os
import time
from datetime import datetime
import pandas as pd

class MT5ConnectionManager:
    """Gerenciador robusto de conexão MetaTrader5"""
    
    def __init__(self):
        self.is_connected = False
        self.account_info = None
        self.terminal_info = None
        self.last_error = None
        
    def check_mt5_installation(self):
        """Verifica se o MT5 está instalado corretamente"""
        try:
            # Verificar se a biblioteca está disponível
            import MetaTrader5 as mt5
            print(f"✅ MetaTrader5 biblioteca versão: {mt5.__version__}")
            return True
        except Exception as e:
            print(f"❌ MT5 não instalado: {e}")
            return False
    
    def get_mt5_status(self):
        """Verifica status atual do MT5 sem tentar conectar"""
        try:
            # Verificar se já está conectado
            terminal_info = mt5.terminal_info()
            account_info = mt5.account_info()
            
            if terminal_info and account_info:
                print("🟢 MT5 já está conectado!")
                print(f"   Terminal: {terminal_info.name}")
                print(f"   Conta: {account_info.login} ({account_info.server})")
                self.is_connected = True
                self.terminal_info = terminal_info
                self.account_info = account_info
                return True
            else:
                print("🟡 MT5 não está conectado")
                return False
                
        except Exception as e:
            print(f"⚠️ Erro ao verificar status MT5: {e}")
            return False
    
    def initialize_mt5_robust(self, max_attempts=3):
        """Inicialização robusta do MT5 com múltiplas tentativas"""
        
        if not self.check_mt5_installation():
            return False
        
        # Primeiro, verificar se já está conectado
        if self.get_mt5_status():
            return True
        
        print(f"🔄 Tentando inicializar MT5... ({max_attempts} tentativas)")
        
        for attempt in range(max_attempts):
            try:
                print(f"   Tentativa {attempt + 1}/{max_attempts}")
                
                # Tentar inicializar
                result = mt5.initialize()
                
                if result:
                    # Verificar se realmente conectou
                    terminal_info = mt5.terminal_info()
                    account_info = mt5.account_info()
                    
                    if terminal_info and account_info:
                        print("✅ MT5 conectado com sucesso!")
                        print(f"   Terminal: {terminal_info.name}")
                        print(f"   Conta: {account_info.login}")
                        print(f"   Servidor: {account_info.server}")
                        print(f"   Saldo: {account_info.balance} {account_info.currency}")
                        
                        self.is_connected = True
                        self.terminal_info = terminal_info
                        self.account_info = account_info
                        return True
                    else:
                        print("⚠️ MT5 inicializou mas sem conta conectada")
                else:
                    error = mt5.last_error()
                    print(f"❌ Falha na inicialização: {error}")
                    self.last_error = error
                
                # Aguardar antes da próxima tentativa
                if attempt < max_attempts - 1:
                    print("   Aguardando 2 segundos...")
                    time.sleep(2)
                    
            except Exception as e:
                print(f"❌ Erro na tentativa {attempt + 1}: {e}")
                self.last_error = str(e)
        
        print("❌ Falha em todas as tentativas de conexão")
        return False
    
    def try_login_with_credentials(self, login=None, password=None, server=None):
        """Tenta fazer login com credenciais específicas"""
        
        # Credenciais padrão (atualize conforme necessário)
        if login is None:
            login = 1522209  # Sua conta demo
        if password is None:
            password = "L@X3CgFz"  # Sua senha (ATUALIZE SE NECESSÁRIO)
        if server is None:
            server = "AMPGlobalUSA-Demo"  # Seu servidor
        
        print(f"🔐 Tentando login na conta {login}...")
        
        try:
            # Verificar se MT5 está inicializado
            if not mt5.initialize():
                print("❌ MT5 não pode ser inicializado")
                return False
            
            # Tentar fazer login
            authorized = mt5.login(login, password, server)
            
            if authorized:
                account_info = mt5.account_info()
                if account_info:
                    print("✅ Login realizado com sucesso!")
                    print(f"   Conta: {account_info.login}")
                    print(f"   Nome: {account_info.name}")
                    print(f"   Servidor: {account_info.server}")
                    print(f"   Saldo: {account_info.balance} {account_info.currency}")
                    
                    self.is_connected = True
                    self.account_info = account_info
                    return True
                else:
                    print("❌ Login falhou - sem informações da conta")
            else:
                error = mt5.last_error()
                print(f"❌ Login falhou: {error}")
                self.last_error = error
                
        except Exception as e:
            print(f"❌ Erro durante login: {e}")
            self.last_error = str(e)
        
        return False
    
    def test_data_connection(self, symbol="EURUSD", timeframe=mt5.TIMEFRAME_M15):
        """Testa se consegue obter dados de mercado"""
        if not self.is_connected:
            print("❌ MT5 não está conectado para teste de dados")
            return False
        
        print(f"📊 Testando obtenção de dados para {symbol}...")
        
        # Lista de símbolos para testar
        test_symbols = [symbol, "EURUSD", "USDCHF", "GBPUSD", "USDJPY", "XAUUSD"]
        
        for test_symbol in test_symbols:
            try:
                # Tentar habilitar o símbolo
                if not mt5.symbol_select(test_symbol, True):
                    continue
                
                # Obter dados
                rates = mt5.copy_rates_from_pos(test_symbol, timeframe, 0, 10)
                
                if rates is not None and len(rates) > 0:
                    print(f"✅ Sucesso com {test_symbol}:")
                    print(f"   {len(rates)} candles obtidos")
                    print(f"   Último preço: {rates[-1]['close']}")
                    print(f"   Timestamp: {pd.to_datetime(rates[-1]['time'], unit='s')}")
                    return True
                    
            except Exception as e:
                print(f"❌ Erro com {test_symbol}: {e}")
        
        print("❌ Nenhum símbolo funcionou para obtenção de dados")
        return False
    
    def get_available_symbols(self, max_symbols=20):
        """Lista símbolos disponíveis"""
        if not self.is_connected:
            return []
        
        try:
            symbols = mt5.symbols_get()
            if symbols:
                visible_symbols = [s.name for s in symbols if s.visible]
                print(f"📋 {len(visible_symbols)} símbolos disponíveis")
                print(f"   Primeiros {max_symbols}: {visible_symbols[:max_symbols]}")
                return visible_symbols
            else:
                print("❌ Nenhum símbolo encontrado")
                return []
        except Exception as e:
            print(f"❌ Erro ao obter símbolos: {e}")
            return []
    
    def diagnose_connection_issues(self):
        """Diagnóstica problemas de conexão"""
        print("\n🔍 DIAGNÓSTICO DE PROBLEMAS MT5:")
        print("=" * 50)
        
        # 1. Verificar instalação
        if not self.check_mt5_installation():
            print("❌ PROBLEMA: MetaTrader5 não está instalado")
            print("   SOLUÇÃO: Instalar MetaTrader5 e biblioteca Python")
            return
        
        # 2. Verificar se MT5 está rodando
        try:
            result = mt5.initialize()
            if not result:
                print("❌ PROBLEMA: MT5 não consegue inicializar")
                print("   POSSÍVEIS CAUSAS:")
                print("   - MetaTrader5 não está aberto")
                print("   - Algorithmic trading desabilitado")
                print("   - Firewall bloqueando")
                print("   - MT5 travado")
                return
        except Exception as e:
            print(f"❌ PROBLEMA: Erro na inicialização - {e}")
            return
        
        # 3. Verificar conta
        account_info = mt5.account_info()
        if not account_info:
            print("❌ PROBLEMA: Nenhuma conta conectada")
            print("   SOLUÇÕES:")
            print("   - Fazer login manual no MT5")
            print("   - Verificar credenciais de conta")
            print("   - Verificar conexão com servidor")
            return
        
        # 4. Verificar terminal
        terminal_info = mt5.terminal_info()
        if terminal_info:
            print(f"✅ Terminal: {terminal_info.name}")
            print(f"✅ Empresa: {terminal_info.company}")
        
        # 5. Testar obtenção de dados
        if not self.test_data_connection():
            print("❌ PROBLEMA: Não consegue obter dados de mercado")
            print("   SOLUÇÕES:")
            print("   - Verificar símbolos disponíveis")
            print("   - Verificar conexão com servidor")
            print("   - Tentar diferentes símbolos")
        
        print("\n✅ Diagnóstico concluído")
    
    def disconnect(self):
        """Desconecta do MT5"""
        try:
            mt5.shutdown()
            self.is_connected = False
            print("👋 Desconectado do MT5")
        except Exception as e:
            print(f"⚠️ Erro ao desconectar: {e}")

# Função global corrigida para usar com o código existente
def initialize_mt5_fixed(server="AMPGlobalUSA-Demo", login=1522209, password="L@X3CgFz"):
    """
    Versão CORRIGIDA da função initialize_mt5 original
    Substitui a função problemática no código principal
    """
    print("\n🔧 INICIALIZANDO MT5 COM CORREÇÕES...")
    
    # Criar gerenciador de conexão
    manager = MT5ConnectionManager()
    
    # Tentar conexão robusta
    if manager.initialize_mt5_robust():
        # Se conectou, testar dados
        if manager.test_data_connection():
            print("✅ MT5 conectado e funcionando!")
            return True
        else:
            print("⚠️ MT5 conectado mas sem dados")
            # Tentar fazer login explícito
            if manager.try_login_with_credentials(login, password, server):
                if manager.test_data_connection():
                    print("✅ MT5 agora está funcionando completamente!")
                    return True
    
    # Se chegou aqui, algo está errado
    print("❌ Falha na conexão MT5")
    manager.diagnose_connection_issues()
    
    print("\n🎯 PRÓXIMOS PASSOS:")
    print("1. Abrir MetaTrader5 manualmente")
    print("2. Fazer login na conta demo/real")
    print("3. Habilitar 'Algorithmic Trading' (Ctrl+E)")
    print("4. Tentar novamente")
    
    return False

def test_mt5_connection():
    """Função para testar a conexão MT5 independentemente"""
    print("🧪 TESTE INDEPENDENTE DE CONEXÃO MT5")
    print("=" * 40)
    
    manager = MT5ConnectionManager()
    
    # Teste completo
    if manager.initialize_mt5_robust():
        manager.test_data_connection()
        manager.get_available_symbols(10)
        return True
    else:
        manager.diagnose_connection_issues()
        return False

if __name__ == "__main__":
    # Executar teste quando chamado diretamente
    test_mt5_connection()
    input("\nPressione Enter para sair...")
