"""
Módulo para sincronizar análises com novos candles
"""
import MetaTrader5 as mt5
from datetime import datetime
import time

class CandleSync:
    def __init__(self):
        self.last_candle_time = {}
        self.timeframe_seconds = {
            "M1": 60,
            "M5": 300,
            "M15": 900,
            "M30": 1800,
            "H1": 3600,
            "H4": 14400,
            "D1": 86400
        }
    
    def get_current_candle_time(self, symbol, timeframe):
        """Obtém o timestamp do candle atual"""
        try:
            timeframe_dict = {
                "M1": mt5.TIMEFRAME_M1,
                "M5": mt5.TIMEFRAME_M5,
                "M15": mt5.TIMEFRAME_M15,
                "M30": mt5.TIMEFRAME_M30,
                "H1": mt5.TIMEFRAME_H1,
                "H4": mt5.TIMEFRAME_H4,
                "D1": mt5.TIMEFRAME_D1,
            }
            
            mt5_timeframe = timeframe_dict.get(timeframe, mt5.TIMEFRAME_M15)
            rates = mt5.copy_rates_from_pos(symbol, mt5_timeframe, 0, 1)
            
            if rates is not None and len(rates) > 0:
                return rates[0]['time']
            return None
        except Exception as e:
            print(f"Erro ao obter tempo do candle: {e}")
            return None
    
    def is_new_candle(self, symbol, timeframe):
        """Verifica se um novo candle foi formado"""
        current_candle_time = self.get_current_candle_time(symbol, timeframe)
        if current_candle_time is None:
            return False
        
        key = f"{symbol}_{timeframe}"
        
        # Se é a primeira verificação
        if key not in self.last_candle_time:
            self.last_candle_time[key] = current_candle_time
            return True
        
        # Se o tempo do candle mudou
        if current_candle_time > self.last_candle_time[key]:
            self.last_candle_time[key] = current_candle_time
            return True
        
        return False
    
    def get_interval_for_timeframe(self, timeframe):
        """Retorna o intervalo apropriado em milissegundos para polling"""
        # Para timeframes pequenos, verificar mais frequentemente
        # Para timeframes grandes, verificar menos frequentemente
        seconds = self.timeframe_seconds.get(timeframe, 300)
        
        if seconds <= 60:  # M1
            return 5000  # 5 segundos
        elif seconds <= 300:  # M5
            return 10000  # 10 segundos
        elif seconds <= 900:  # M15
            return 30000  # 30 segundos
        elif seconds <= 1800:  # M30
            return 60000  # 1 minuto
        else:  # H1+
            return 120000  # 2 minutos

# Instância global
candle_sync = CandleSync()
