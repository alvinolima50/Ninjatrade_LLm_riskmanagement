############# ESTA TREINANDO TODOS OS INDICADORES BEM, NAO SEI FALAR AGORA MAS PARECE BOM SE LAPIDAR, VOU TENTAR ISOLAR CADA INDICADOR NAS PROXIMAS VERSOES

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import talib as ta
import joblib
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tensorflow.keras.models import Sequential, load_model, save_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import warnings
warnings.filterwarnings('ignore')

# Constantes
SYMBOL = "NG=F"  # Símbolo do Yahoo Finance para Gás Natural
WINDOW_SIZE = 30  # Reduzido para diminuir overfitting
TRAIN_SPLIT = 0.8  # Divisão treino/teste
PREDICTION_THRESHOLDS = {
    'Slope': 0.0,      # Limiar para considerar tendência positiva/negativa na inclinação
    'ATR': 0.0,        # Variação % no ATR
    'DirectionalEntropy': 0.0,  # Variação % na entropia
    'EMA14': 0.0       # Variação % na EMA14
}
MODELS_DIR = 'models/indicators'
RESULTS_DIR = 'results/indicators'

# Função para criar pastas necessárias
def create_directories():
    """Cria as pastas necessárias para salvar modelos e resultados"""
    import os
    os.makedirs('data', exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

# Função para download de dados
def download_data(symbol, period="2y", interval="1d"):
    """Download de dados do Yahoo Finance"""
    try:
        print(f"Baixando dados para {symbol}...")
        data = yf.download(symbol, period=period, interval=interval)
        print(f"Baixados {len(data)} registros para {symbol}")
        
        # Verificar se temos dados
        if len(data) == 0:
            print("Aviso: Nenhum dado baixado. O símbolo pode estar incorreto ou indisponível.")
            return None
        
        # Corrigir colunas se necessário (MultiIndex do Yahoo Finance)
        if isinstance(data.columns, pd.MultiIndex):
            print("Corrigindo colunas MultiIndex dos dados do Yahoo Finance...")
            if data.columns.nlevels == 2:
                data.columns = data.columns.get_level_values(0)
        
        print("Amostra dos dados baixados:")
        print(data.head())
        
        return data
    except Exception as e:
        print(f"Erro ao baixar dados: {e}")
        return None

# Função para carregar dados de um arquivo CSV
def load_data_from_csv(file_path):
    """Carrega dados de um arquivo CSV"""
    try:
        data = pd.read_csv(file_path, index_col=0, parse_dates=True)
        print(f"Carregados {len(data)} registros de {file_path}")
        return data
    except Exception as e:
        print(f"Erro ao carregar dados: {e}")
        return None

# Funções para calcular indicadores
def calculate_slope(series, period=14):
    """Calcula a inclinação (slope) usando regressão linear"""
    slopes = []
    for i in range(len(series)):
        if i < period:
            slopes.append(np.nan)
        else:
            y = series.iloc[i-period:i].values
            x = np.arange(period)
            slope, _ = np.polyfit(x, y, 1)
            slopes.append(slope)
    return pd.Series(slopes, index=series.index)

def calculate_directional_entropy(series, period=14):
    """Calcula a entropia direcional de uma série temporal"""
    entropy = []
    for i in range(len(series)):
        if i < period:
            entropy.append(np.nan)
        else:
            # Calcular mudanças de preço
            changes = np.diff(series.iloc[i-period:i].values)
            # Converter para movimentos para cima (1), para baixo (-1) ou laterais (0)
            directions = np.sign(changes)
            # Contar ocorrências de cada direção
            unique, counts = np.unique(directions, return_counts=True)
            # Calcular probabilidades
            probabilities = counts / np.sum(counts)
            # Calcular entropia
            h = -np.sum(probabilities * np.log2(probabilities + 1e-10))
            # Normalizar entropia
            h_norm = h / np.log2(len(unique))
            entropy.append(h_norm)
    return pd.Series(entropy, index=series.index)

def create_features(df):
    """Cria features para o modelo incluindo os indicadores solicitados"""
    print("Criando features...")
    
    # Fazer uma cópia do DataFrame
    data = df.copy()
    
    # Verificar se as colunas necessárias existem
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    print(f"Colunas disponíveis nos dados: {data.columns.tolist()}")
    
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        print(f"Erro: Colunas necessárias ausentes: {missing_columns}")
        # Tentar lidar com variações comuns de nomes de colunas
        column_map = {}
        
        for req_col in missing_columns:
            # Verificar versão minúscula
            if req_col.lower() in data.columns:
                column_map[req_col.lower()] = req_col
            # Verificar versão maiúscula
            elif req_col.upper() in data.columns:
                column_map[req_col.upper()] = req_col
            # Verificar versão capitalizada
            elif req_col.capitalize() in data.columns:
                column_map[req_col.capitalize()] = req_col
        
        if column_map:
            print(f"Renomeando colunas: {column_map}")
            data = data.rename(columns=column_map)
        
        # Verificar novamente após renomear
        still_missing = [col for col in required_columns if col not in data.columns]
        if still_missing:
            print(f"Erro: Ainda faltam colunas necessárias após renomear: {still_missing}")
            print("Certifique-se de que seus dados têm as colunas necessárias: Open, High, Low, Close, Volume")
            return None
    
    # Garantir que não há valores NaN nos dados de preço
    data = data.dropna(subset=required_columns)
    
    # Criar features básicas de preço
    data['HL_Pct'] = (data['High'] - data['Low']) / data['Close'] * 100.0
    data['Daily_Return'] = data['Close'].pct_change() * 100.0
    
    # Calcular os indicadores solicitados
    # 1. Slope (usando regressão linear nos preços de Close)
    data['Slope'] = calculate_slope(data['Close'], period=14)
    
    # 2. ATR (Average True Range) - medida de volatilidade
    try:
        # Converter para arrays numpy
        high = data['High'].values
        low = data['Low'].values
        close = data['Close'].values
        
        if len(high) == len(low) == len(close):
            data['ATR'] = ta.ATR(high, low, close, timeperiod=14)
        else:
            print("Erro: Arrays de preço têm comprimentos diferentes. Usando cálculo alternativo de ATR.")
            # Cálculo alternativo de ATR
            true_range = np.maximum(
                data['High'] - data['Low'],
                np.maximum(
                    np.abs(data['High'] - data['Close'].shift(1)),
                    np.abs(data['Low'] - data['Close'].shift(1))
                )
            )
            data['ATR'] = true_range.rolling(window=14).mean()
    except Exception as e:
        print(f"Erro ao calcular ATR com TA-Lib: {e}")
        print("Usando cálculo alternativo de ATR.")
        # Cálculo alternativo de ATR
        true_range = np.maximum(
            data['High'] - data['Low'],
            np.maximum(
                np.abs(data['High'] - data['Close'].shift(1)),
                np.abs(data['Low'] - data['Close'].shift(1))
            )
        )
        data['ATR'] = true_range.rolling(window=14).mean()
    
    # 3. Entropia Direcional - medida de aleatoriedade nos movimentos de preço
    data['DirectionalEntropy'] = calculate_directional_entropy(data['Close'], period=14)
    
    # 4. EMA14 (Média Móvel Exponencial com período 14)
    try:
        data['EMA14'] = ta.EMA(data['Close'].values, timeperiod=14)
    except Exception as e:
        print(f"Erro ao calcular EMA com TA-Lib: {e}")
        print("Usando cálculo EMA do pandas.")
        data['EMA14'] = data['Close'].ewm(span=14, adjust=False).mean()
    
    # Adicionar colunas alvo para cada indicador
    # Usamos as mudanças percentuais nos indicadores
    data['Slope_Next'] = data['Slope'].shift(-1)
    data['Slope_Change'] = (data['Slope_Next'] - data['Slope'])
    data['Slope_Target'] = (data['Slope_Change'] > PREDICTION_THRESHOLDS['Slope']).astype(int)
    
    data['ATR_Next'] = data['ATR'].shift(-1)
    data['ATR_Change'] = ((data['ATR_Next'] - data['ATR']) / data['ATR']) * 100
    data['ATR_Target'] = (data['ATR_Change'] > PREDICTION_THRESHOLDS['ATR']).astype(int)
    
    data['DirectionalEntropy_Next'] = data['DirectionalEntropy'].shift(-1)
    data['DirectionalEntropy_Change'] = ((data['DirectionalEntropy_Next'] - data['DirectionalEntropy']))
    data['DirectionalEntropy_Target'] = (data['DirectionalEntropy_Change'] > PREDICTION_THRESHOLDS['DirectionalEntropy']).astype(int)
    
    data['EMA14_Next'] = data['EMA14'].shift(-1)
    data['EMA14_Change'] = ((data['EMA14_Next'] - data['EMA14']) / data['EMA14']) * 100
    data['EMA14_Target'] = (data['EMA14_Change'] > PREDICTION_THRESHOLDS['EMA14']).astype(int)
    
    # Adicionar mais features derivadas para melhorar o modelo
    # Cruzamentos de EMA
    data['EMA5'] = data['Close'].ewm(span=5, adjust=False).mean()
    data['EMA20'] = data['Close'].ewm(span=20, adjust=False).mean()
    data['EMA5_EMA20_Cross'] = ((data['EMA5'] > data['EMA20']).astype(int) - 
                               (data['EMA5'].shift(1) > data['EMA20'].shift(1)).astype(int))
    
    # RSI - Índice de Força Relativa
    try:
        data['RSI'] = ta.RSI(data['Close'].values, timeperiod=14)
    except:
        # Cálculo manual do RSI
        delta = data['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        data['RSI'] = 100 - (100 / (1 + rs))
    
    # Bandas de Bollinger
    try:
        data['BB_Upper'], data['BB_Middle'], data['BB_Lower'] = ta.BBANDS(
            data['Close'].values, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
        data['BB_Width'] = (data['BB_Upper'] - data['BB_Lower']) / data['BB_Middle']
    except:
        # Cálculo manual das Bandas de Bollinger
        data['BB_Middle'] = data['Close'].rolling(window=20).mean()
        data['BB_Std'] = data['Close'].rolling(window=20).std()
        data['BB_Upper'] = data['BB_Middle'] + (data['BB_Std'] * 2)
        data['BB_Lower'] = data['BB_Middle'] - (data['BB_Std'] * 2)
        data['BB_Width'] = (data['BB_Upper'] - data['BB_Lower']) / data['BB_Middle']
    
    # Remover linhas com NaN
    data = data.dropna()
    print(f"Criação de features completa. Formato final: {data.shape}")
    
    # Verificar a distribuição dos alvos
    for target in ['Slope_Target', 'ATR_Target', 'DirectionalEntropy_Target', 'EMA14_Target']:
        class_counts = data[target].value_counts()
        print(f"Distribuição do alvo {target}: {class_counts.to_dict()}")
    
    return data

def prepare_sequences(data, feature_cols, target_col, window_size=WINDOW_SIZE):
    """Prepara sequências para o modelo LSTM"""
    X, y = [], []
    
    # Extrair e normalizar features
    features_data = data[feature_cols].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    features_scaled = scaler.fit_transform(features_data)
    
    # Extrair alvo
    target = data[target_col].values
    
    # Criar sequências
    for i in range(len(features_scaled) - window_size):
        X.append(features_scaled[i:i+window_size])
        y.append(target[i+window_size])
    
    X = np.array(X)
    y = np.array(y)
    
    return X, y, scaler

def split_data(X, y, train_split=TRAIN_SPLIT):
    """Divide os dados em conjuntos de treino e teste"""
    # Usar uma abordagem mais sofisticada para divisão dos dados
    # Garantir que a divisão mantenha a distribuição de classes
    
    n_samples = len(X)
    indices = np.arange(n_samples)
    
    # Verificar distribuição de classes
    classes, counts = np.unique(y, return_counts=True)
    print(f"Distribuição de classes nos dados completos: {dict(zip(classes, counts))}")
    
    # Dividir dados preservando a distribuição de classes
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, indices, 
        test_size=1-train_split, 
        random_state=42,
        stratify=y  # Garante que a distribuição de classes seja preservada
    )
    
    print(f"Treino: {len(X_train)} amostras, Teste: {len(X_test)} amostras")
    
    return X_train, X_test, y_train, y_test

def build_model(input_shape):
    """Constrói um modelo LSTM para classificação binária"""
    model = Sequential([
        # Reduzir a complexidade do modelo para evitar overfitting
        LSTM(32, return_sequences=True, input_shape=input_shape),
        Dropout(0.3),  # Aumentar o dropout para reduzir overfitting
        LSTM(16),
        Dropout(0.3),  # Aumentar o dropout para reduzir overfitting
        Dense(8, activation='relu'),
        # Adicionar regularização L2 para reduzir overfitting
        Dense(1, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l2(0.001))
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),  # Reduzir a taxa de aprendizado
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_indicator_model(data, indicator_name, feature_cols, target_col):
    """Treina um modelo para um indicador específico"""
    print(f"\n{'='*50}")
    print(f"Treinando modelo para o indicador: {indicator_name}")
    print(f"{'='*50}")
    
    # Preparar sequências
    X, y, scaler = prepare_sequences(data, feature_cols, target_col, WINDOW_SIZE)
    
    # Dividir dados
    X_train, X_test, y_train, y_test = split_data(X, y, TRAIN_SPLIT)
    
    print(f"Formato dos dados de treino: X={X_train.shape}, y={y_train.shape}")
    print(f"Formato dos dados de teste: X={X_test.shape}, y={y_test.shape}")
    
    # Verificar distribuição das classes
    train_class_dist = np.bincount(y_train.astype(int))
    test_class_dist = np.bincount(y_test.astype(int))
    
    print(f"Distribuição das classes (treino): {train_class_dist}")
    print(f"Distribuição das classes (teste): {test_class_dist}")
    
    # Calcular class_weight para lidar com desbalanceamento
    n_samples = len(y_train)
    n_classes = len(train_class_dist)
    class_weights = {}
    for i in range(n_classes):
        class_weights[i] = n_samples / (n_classes * train_class_dist[i])
    
    print(f"Pesos das classes: {class_weights}")
    
    # Construir modelo
    model = build_model(input_shape=(X_train.shape[1], X_train.shape[2]))
    
    # Callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=15,  # Aumentado para evitar parar prematuramente
        restore_best_weights=True,
        verbose=1
    )
    
    model_checkpoint = ModelCheckpoint(
        filepath=f"{MODELS_DIR}/{indicator_name}_model.h5",
        save_best_only=True,
        monitor='val_loss',
        mode='min',
        verbose=1
    )
    
    # Adicionar ReduceLROnPlateau para ajustar a taxa de aprendizado durante o treinamento
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=0.00001,
        verbose=1
    )
    
    # Treinar modelo
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=50,  # Reduzido para evitar overfitting
        batch_size=16,  # Batch size menor para melhor generalização
        callbacks=[early_stopping, model_checkpoint, reduce_lr],
        class_weight=class_weights,  # Usar pesos de classe para dados desbalanceados
        verbose=1
    )
    
    # Avaliar modelo
    y_pred_proba = model.predict(X_test)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nPrecisão no conjunto de teste: {accuracy:.4f}")
    
    # Relatório de classificação
    print("\nRelatório de Classificação:")
    print(classification_report(y_test, y_pred, target_names=['Diminuição', 'Aumento']))
    
    # Matriz de confusão
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Diminuição', 'Aumento'], 
                yticklabels=['Diminuição', 'Aumento'])
    plt.xlabel('Previsto')
    plt.ylabel('Real')
    plt.title(f'Matriz de Confusão - {indicator_name}')
    plt.savefig(f"{RESULTS_DIR}/{indicator_name}_confusion_matrix.png")
    plt.close()
    
    # Plotar histórico de treino
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Treino')
    plt.plot(history.history['val_loss'], label='Validação')
    plt.title(f'Loss - {indicator_name}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Treino')
    plt.plot(history.history['val_accuracy'], label='Validação')
    plt.title(f'Acurácia - {indicator_name}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/{indicator_name}_training_history.png")
    plt.close()
    
    # Plotar curva de aprendizado (loss por epoch)
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Treino', color='blue', linewidth=2)
    plt.plot(history.history['val_loss'], label='Validação', color='red', linewidth=2)
    
    # Adicionar linha para o epoch com o menor val_loss
    best_epoch = np.argmin(history.history['val_loss'])
    plt.axvline(x=best_epoch, color='green', linestyle='--', 
                label=f'Melhor epoch ({best_epoch})')
    
    plt.title(f'Curva de Aprendizado - {indicator_name}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{RESULTS_DIR}/{indicator_name}_learning_curve.png")
    plt.close()
    
    # Salvar o scaler
    joblib.dump(scaler, f"{MODELS_DIR}/{indicator_name}_scaler.pkl")
    
    # Salvar configuração do modelo
    model_config = {
        'feature_cols': feature_cols,
        'target_col': target_col,
        'window_size': WINDOW_SIZE,
        'accuracy': accuracy
    }
    joblib.dump(model_config, f"{MODELS_DIR}/{indicator_name}_config.pkl")
    
    return model, scaler, accuracy

def predict_next_values(data, models_info):
    """Faz previsões para o próximo valor de cada indicador"""
    print("\n" + "="*70)
    print("PREVISÕES PARA O PRÓXIMO CANDLE")
    print("="*70)
    
    predictions = {}
    latest_values = {}
    
    for indicator, info in models_info.items():
        model = info['model']
        scaler = info['scaler']
        feature_cols = info['feature_cols']
        window_size = info['window_size']
        
        # Obter dados recentes
        recent_data = data[feature_cols].values[-window_size:]
        
        # Normalizar
        recent_data_scaled = scaler.transform(recent_data)
        
        # Preparar para previsão
        X_pred = recent_data_scaled.reshape(1, window_size, len(feature_cols))
        
        # Fazer previsão
        prediction = model.predict(X_pred)[0][0]
        binary_prediction = int(prediction > 0.5)
        
        # Armazenar previsão e valor atual
        predictions[indicator] = {
            'probability': float(prediction),
            'binary': binary_prediction,
            'confidence': max(prediction, 1-prediction)
        }
        
        # Obter o valor atual do indicador
        latest_value = data[indicator].iloc[-1]
        latest_values[indicator] = latest_value
        
        # Texto para aumento/diminuição
        direction = "AUMENTO" if binary_prediction == 1 else "DIMINUIÇÃO"
        
        # Calcular nível de confiança
        confidence = predictions[indicator]['confidence']
        if confidence > 0.8:
            confidence_level = "ALTA"
        elif confidence > 0.65:
            confidence_level = "MÉDIA"
        else:
            confidence_level = "BAIXA"
        
        # Imprimir previsão
        print(f"\n📊 {indicator}:")
        print(f"  • Valor atual: {latest_value:.6f}")
        print(f"  • Previsão: {direction} com {confidence*100:.1f}% de probabilidade")
        print(f"  • Confiança: {confidence_level} ({confidence*100:.1f}%)")
        
        # Análise adicional específica por indicador
        if indicator == 'Slope':
            if latest_value > 0:
                current_trend = "positiva (tendência de alta)"
            elif latest_value < 0:
                current_trend = "negativa (tendência de baixa)"
            else:
                current_trend = "neutra"
                
            print(f"  • Inclinação atual é {current_trend}")
            if binary_prediction == 1 and latest_value <= 0:
                print("  • ALERTA: Possível reversão de tendência de baixa para alta")
            elif binary_prediction == 0 and latest_value >= 0:
                print("  • ALERTA: Possível reversão de tendência de alta para baixa")
            elif binary_prediction == 1 and latest_value > 0:
                print("  • Tendência de alta deve continuar fortalecendo")
            elif binary_prediction == 0 and latest_value < 0:
                print("  • Tendência de baixa deve continuar fortalecendo")
                
        elif indicator == 'ATR':
            if binary_prediction == 1:
                print("  • Expectativa de aumento na volatilidade")
                print(f"  • Prepare-se para movimentos maiores que o normal")
            else:
                print("  • Expectativa de diminuição na volatilidade")
                print(f"  • Mercado pode se acalmar, com movimentos menores")
                
        elif indicator == 'DirectionalEntropy':
            if binary_prediction == 1:
                print("  • Expectativa de aumento na aleatoriedade do mercado")
                print("  • O mercado pode se tornar menos previsível e mais caótico")
                if latest_value > 0.7:
                    print("  • ALERTA: Entropia já está alta, mercado pode ficar extremamente imprevisível")
            else:
                print("  • Expectativa de diminuição na aleatoriedade do mercado")
                print("  • O mercado pode se tornar mais ordenado e desenvolver uma tendência clara")
                
        elif indicator == 'EMA14':
            current_close = data['Close'].iloc[-1]
            ema_relation = (current_close - latest_value) / latest_value * 100
            
            if ema_relation > 0:
                print(f"  • Preço atual está {ema_relation:.2f}% ACIMA da EMA14 (sinal de alta)")
            else:
                print(f"  • Preço atual está {abs(ema_relation):.2f}% ABAIXO da EMA14 (sinal de baixa)")
                
            if binary_prediction == 1:
                print("  • EMA14 deve aumentar, indicando potencial momentum de alta")
            else:
                print("  • EMA14 deve diminuir, indicando potencial momentum de baixa")
    
    # Análise conjunta
    print("\n" + "="*70)
    print("ANÁLISE CONJUNTA DOS INDICADORES")
    print("="*70)
    
    # Contagem de sinais de alta vs. baixa
    bullish_signals = sum(1 for p in predictions.values() if p['binary'] == 1)
    bearish_signals = len(predictions) - bullish_signals
    
    print(f"\n📈 Sinais de ALTA: {bullish_signals}")
    print(f"📉 Sinais de BAIXA: {bearish_signals}")
    
    # Análise de tendência vs. volatilidade
    slope_bullish = predictions['Slope']['binary'] == 1
    entropy_increasing = predictions['DirectionalEntropy']['binary'] == 1
    volatility_increasing = predictions['ATR']['binary'] == 1
    ema_bullish = predictions['EMA14']['binary'] == 1
    
    print("\n🔍 INTERPRETAÇÃO GERAL:")
    
    # Cenários de tendência clara
    if slope_bullish and ema_bullish and not entropy_increasing:
        print("  • TENDÊNCIA DE ALTA forte e clara se desenvolvendo")
        print("  • Boa oportunidade para posições COMPRADAS")
    elif not slope_bullish and not ema_bullish and not entropy_increasing:
        print("  • TENDÊNCIA DE BAIXA forte e clara se desenvolvendo")
        print("  • Boa oportunidade para posições VENDIDAS")
    
    # Cenários de possível reversão
    elif slope_bullish and not ema_bullish:
        print("  • Possível REVERSÃO de baixa para alta em formação")
        print("  • Considere aguardar confirmação antes de entrar comprado")
    elif not slope_bullish and ema_bullish:
        print("  • Possível REVERSÃO de alta para baixa em formação")
        print("  • Considere aguardar confirmação antes de entrar vendido")
    
    # Cenários de alta volatilidade/entropia
    if entropy_increasing and volatility_increasing:
        print("  • ALERTA: Alta volatilidade e imprevisibilidade esperadas")
        print("  • Recomenda-se cautela e redução no tamanho das posições")
    
    # Coerência entre os indicadores
    if bullish_signals >= 3:
        print("  • CONSENSO DE ALTA: Maioria dos indicadores aponta para movimento de alta")
    elif bearish_signals >= 3:
        print("  • CONSENSO DE BAIXA: Maioria dos indicadores aponta para movimento de baixa")
    else:
        print("  • SINAIS MISTOS: Não há consenso claro entre os indicadores")
        print("  • Recomenda-se cautela ou aguardar sinais mais claros")
    
    return predictions, latest_values

def plot_indicators_history(data):
    """Plota o histórico recente dos indicadores"""
    # Mostrar apenas os últimos 100 candles
    recent_data = data.iloc[-100:].copy()
    
    # Criar um gráfico com 4 subplots
    fig, axes = plt.subplots(4, 1, figsize=(15, 16), sharex=True)
    
    # Plot 1: Preço e EMA14
    axes[0].plot(recent_data.index, recent_data['Close'], label='Close', color='blue')
    axes[0].plot(recent_data.index, recent_data['EMA14'], label='EMA14', color='red')
    axes[0].set_title('Preço e EMA14')
    axes[0].set_ylabel('Preço')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Slope
    axes[1].plot(recent_data.index, recent_data['Slope'], label='Slope', color='green')
    axes[1].axhline(y=0, color='gray', linestyle='--')
    axes[1].set_title('Slope (Inclinação)')
    axes[1].set_ylabel('Valor')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: ATR
    axes[2].plot(recent_data.index, recent_data['ATR'], label='ATR', color='purple')
    axes[2].set_title('ATR (Volatilidade)')
    axes[2].set_ylabel('Valor')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    # Plot 4: Entropia Direcional
    axes[3].plot(recent_data.index, recent_data['DirectionalEntropy'], label='Entropia Direcional', color='orange')
    axes[3].set_title('Entropia Direcional')
    axes[3].set_ylabel('Valor')
    axes[3].set_xlabel('Data')
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/indicators_history.png")
    plt.close()
    
    # Adicionalmente, criar um gráfico de candlestick
    try:
        import mplfinance as mpf
        
        # Preparar dados para mplfinance
        df_mpf = recent_data[['Open', 'High', 'Low', 'Close']].copy()
        
        # Adicionar EMA como overlay
        apd = [mpf.make_addplot(recent_data['EMA14'], color='red')]
        
        # Plotar gráfico de candlestick
        mpf.plot(
            df_mpf,
            type='candle',
            style='yahoo',
            title=f'{SYMBOL} Gráfico de Candlestick com EMA14',
            ylabel='Preço',
            volume=False,
            addplot=apd,
            savefig=f"{RESULTS_DIR}/candlestick_chart.png"
        )
    except ImportError:
        print("mplfinance não instalado. Pulando gráfico de candlestick.")

def main():
    """Função principal"""
    # Criar diretórios
    create_directories()
    
    # Baixar ou carregar dados
    # Opção 1: Baixar dados do Yahoo Finance
    data = download_data(SYMBOL, period="2y", interval="1d")
    if data is not None:
        data.to_csv(f'data/{SYMBOL}_data.csv')
    
    # Opção 2: Carregar dados de um arquivo CSV
    # data = load_data_from_csv('data/seu_arquivo.csv')
    
    if data is None:
        print("Erro: Nenhum dado disponível. Verifique sua fonte de dados.")
        return
    
    # Criar features
    data = create_features(data)
    if data is None:
        print("Erro durante a criação de features. Saindo.")
        return
    
    # Salvar dados processados
    data.to_csv(f'data/{SYMBOL}_processed.csv')
    
    # Definir colunas de features básicas para todos os modelos
    base_features = ['Open', 'High', 'Low', 'Close', 'Volume', 'HL_Pct', 'Daily_Return']
    
    # Configurar modelos para cada indicador
    indicator_configs = {
        'Slope': {
            'feature_cols': base_features + ['Slope', 'ATR', 'DirectionalEntropy', 'EMA14', 
                                           'EMA5', 'EMA20', 'EMA5_EMA20_Cross', 'RSI', 
                                           'BB_Upper', 'BB_Middle', 'BB_Lower', 'BB_Width'],
            'target_col': 'Slope_Target'
        },
        'ATR': {
            'feature_cols': base_features + ['Slope', 'ATR', 'DirectionalEntropy', 'EMA14', 
                                           'EMA5', 'EMA20', 'EMA5_EMA20_Cross', 'RSI', 
                                           'BB_Upper', 'BB_Middle', 'BB_Lower', 'BB_Width'],
            'target_col': 'ATR_Target'
        },
        'DirectionalEntropy': {
            'feature_cols': base_features + ['Slope', 'ATR', 'DirectionalEntropy', 'EMA14', 
                                           'EMA5', 'EMA20', 'EMA5_EMA20_Cross', 'RSI', 
                                           'BB_Upper', 'BB_Middle', 'BB_Lower', 'BB_Width'],
            'target_col': 'DirectionalEntropy_Target'
        },
        'EMA14': {
            'feature_cols': base_features + ['Slope', 'ATR', 'DirectionalEntropy', 'EMA14', 
                                           'EMA5', 'EMA20', 'EMA5_EMA20_Cross', 'RSI', 
                                           'BB_Upper', 'BB_Middle', 'BB_Lower', 'BB_Width'],
            'target_col': 'EMA14_Target'
        }
    }
    
    # Treinar modelos para cada indicador
    models_info = {}
    for indicator, config in indicator_configs.items():
        model, scaler, accuracy = train_indicator_model(
            data,
            indicator,
            config['feature_cols'],
            config['target_col']
        )
        
        models_info[indicator] = {
            'model': model,
            'scaler': scaler,
            'feature_cols': config['feature_cols'],
            'window_size': WINDOW_SIZE,
            'accuracy': accuracy
        }
    
    # Plotar histórico dos indicadores
    plot_indicators_history(data)
    
    # Fazer previsões para o próximo candle
    predictions, latest_values = predict_next_values(data, models_info)
    
    print("\nTreinamento e avaliação completos!")
    print(f"Modelos salvos no diretório '{MODELS_DIR}'")
    print(f"Resultados salvos no diretório '{RESULTS_DIR}'")
    
    return models_info, predictions, latest_values

def load_models_and_predict(data_path=None):
    """Carrega modelos salvos e faz previsões com novos dados"""
    import os
    
    # Verificar se os modelos existem
    if not os.path.exists(MODELS_DIR):
        print(f"Erro: Diretório de modelos '{MODELS_DIR}' não encontrado.")
        return None
    
    # Carregar dados
    if data_path:
        data = load_data_from_csv(data_path)
    else:
        # Baixar dados mais recentes
        data = download_data(SYMBOL, period="60d", interval="1d")
    
    if data is None:
        print("Erro: Nenhum dado disponível. Verifique sua fonte de dados.")
        return None
    
    # Criar features
    data = create_features(data)
    if data is None:
        print("Erro durante a criação de features. Saindo.")
        return None
    
    # Carregar modelos e fazer previsões
    models_info = {}
    indicators = ['Slope', 'ATR', 'DirectionalEntropy', 'EMA14']
    
    for indicator in indicators:
        model_path = f"{MODELS_DIR}/{indicator}_model.h5"
        scaler_path = f"{MODELS_DIR}/{indicator}_scaler.pkl"
        config_path = f"{MODELS_DIR}/{indicator}_config.pkl"
        
        if not os.path.exists(model_path) or not os.path.exists(scaler_path) or not os.path.exists(config_path):
            print(f"Erro: Arquivos do modelo para {indicator} não encontrados.")
            continue
        
        try:
            model = load_model(model_path)
            scaler = joblib.load(scaler_path)
            config = joblib.load(config_path)
            
            models_info[indicator] = {
                'model': model,
                'scaler': scaler,
                'feature_cols': config['feature_cols'],
                'window_size': config['window_size']
            }
            
            print(f"Modelo para {indicator} carregado com sucesso.")
        except Exception as e:
            print(f"Erro ao carregar modelo para {indicator}: {e}")
    
    if not models_info:
        print("Nenhum modelo pôde ser carregado. Saindo.")
        return None
    
    # Plotar histórico dos indicadores
    plot_indicators_history(data)
    
    # Fazer previsões
    predictions, latest_values = predict_next_values(data, models_info)
    
    return predictions, latest_values

if __name__ == "__main__":
    main()
    # Para carregar modelos salvos e fazer previsões:
    # load_models_and_predict()